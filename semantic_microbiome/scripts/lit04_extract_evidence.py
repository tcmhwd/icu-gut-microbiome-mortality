"""
lit04_extract_evidence.py — Phase I / Gate 1, step 4: structured,
passage-grounded evidence extraction from every fetched paper (spec sec.
11, 14, 15).

For each PMID, one LLM call (gpt-4o-mini) receives the ENTIRE available
source text for that paper (full text if lit02 got it, else abstract
only -- concatenated from lit03's per-PMID chunks, so this reuses the
same chunk boundaries, not a fresh re-read) and returns:
  - study-level fields (country, design, N_total/N_deaths, population,
    specimen, sequencing method, mortality endpoint, etc.)
  - inclusion/exclusion signal (direct_human_mortality_study bool +
    exclusion_flags list, per spec sec. 11-12)
  - a list of feature-level evidence rows (spec sec. 14 schema), each
    requiring an `evidence_passage` that must be an actual quote from
    the source text.

"NO UNSUPPORTED LLM EXTRACTION" (spec sec. 15): after the LLM call, every
`evidence_passage` is checked against the ACTUAL source text
(normalized-whitespace substring match). Rows that don't verify are kept
(nothing is silently dropped, sec. 16) but flagged
`passage_verified=False` and `manual_review_required=True` --
downstream (lit06) must not treat them as accepted Tier A/B evidence.

hospital_mortality (Gangnam) is never read by this script -- this is
pure literature text extraction.

Outputs:
  literature_prior/extraction/evidence_raw.jsonl   (one JSON object per paper, study-level + nested features)
  literature_prior/extraction/evidence_flat.csv     (one row per feature-evidence pair, study fields repeated)
  literature_prior/extraction/extraction_manifest.json
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('OPENAI_API_KEY', '').strip():
    raise SystemExit('OPENAI_API_KEY must be set (see README).')

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

from common import SEMANTIC, today_str

ABSTRACTS_DIR = SEMANTIC / "literature_prior" / "abstracts"
FULLTEXT_DIR = SEMANTIC / "literature_prior" / "fulltext"
EXTRACTION_DIR = SEMANTIC / "literature_prior" / "extraction"
EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)

def _openai_api_key() -> str:  # [PUBLIC RELEASE] required env var replaces the original local key-file path
    k = __import__("os").environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        raise SystemExit("OPENAI_API_KEY must be set (see README).")
    return k
GEN_MODEL = "gpt-4o-mini"
N_WORKERS = 6
MAX_CHARS = 60000  # ~15-18k tokens; truncate extreme outliers rather than drop the paper

ENDPOINT_TYPES = ["ICU mortality", "in-hospital mortality", "28-day mortality", "30-day mortality",
                  "60-day mortality", "90-day mortality", "180-day mortality", "other/unspecified"]

SYSTEM_PROMPT = f"""You are extracting STRUCTURED evidence from a single biomedical paper about gut/intestinal microbiome and critically ill (ICU/sepsis/critical illness) patients. You will be given the paper's available text (full text or abstract) with its PMID.

Follow these rules strictly:
1. Only extract information that is actually stated in the given text. Never invent numbers, countries, or findings not present.
2. Every feature-level finding MUST include an `evidence_passage` that is an EXACT (or near-exact) quote copied from the given text -- not a paraphrase. If you cannot find a supporting passage, do not include that finding.
3. Preserve the EXACT mortality endpoint terminology used by the paper (one of: {ENDPOINT_TYPES}) -- do not treat different endpoints (e.g. 28-day vs 90-day mortality) as the same.
4. If the paper is not about (a) human patients, (b) ICU or clearly critically-ill population, (c) gut/fecal/rectal microbiome, (d) a mortality/survival endpoint -- set direct_human_mortality_study=false and explain why in exclusion_flags. Still extract any features it reports (they will be used as lower-tier mechanistic evidence).
5. If a feature is evaluated but the paper reports NO significant association, set direction="null" and still include it (do not omit null findings -- publication-bias avoidance is required).

Return ONLY a single JSON object (no other text) with this exact shape:
{{
  "study_id": "<short readable id, e.g. FirstAuthorYear>",
  "country": "<string or null>",
  "center_type": "<'academic'/'community'/'mixed'/null>",
  "single_or_multicenter": "<'single'/'multi'/null>",
  "population": "<'adult'/'pediatric'/'mixed'/null>",
  "ICU_type": "<e.g. 'general ICU','surgical ICU','medical ICU', or null>",
  "study_design": "<'prospective_cohort'/'retrospective_cohort'/'case_control'/'review_or_meta_analysis'/'animal'/'in_vitro'/'other'>",
  "N_total": <int or null>,
  "N_deaths": <int or null>,
  "sampling_time": "<string or null>",
  "specimen_type": "<'stool'/'fecal'/'rectal_swab'/other or null>",
  "sequencing_method": "<string or null>",
  "16S_or_shotgun": "<'16S'/'shotgun'/'both'/null>",
  "analysis_method": "<string or null>",
  "mortality_endpoint": "<one of {ENDPOINT_TYPES} or null>",
  "followup_duration": "<string or null>",
  "direct_human_mortality_study": <true/false>,
  "exclusion_flags": [<subset of: "non_ICU_cohort","healthy_vs_ICU_no_mortality","severity_only","infection_acquisition_only","LOS_only","mechanical_ventilation_only","composite_outcome_cannot_separate_mortality","animal_study","cell_study","non_gut_microbiome">],
  "llm_confidence": "<'high'/'medium'/'low'>",
  "features": [
    {{
      "feature_view": "<'taxonomy'/'function'/'resistome'/'genome_resolved'/'diversity'/'ratio'/'other'>",
      "feature_name_as_reported": "<string>",
      "normalized_feature_name": "<your best-guess canonical name>",
      "taxonomic_level": "<'species'/'genus'/'family'/'order'/'phylum'/null>",
      "database_identifier_if_available": "<string or null>",
      "direction": "<'higher_in_death'/'lower_in_death'/'higher_in_survival'/'lower_in_survival'/'null'>",
      "effect_measure": "<'OR'/'HR'/'RR'/'p-value_only'/'descriptive' or null>",
      "effect_size": <number or null>,
      "CI_low": <number or null>,
      "CI_high": <number or null>,
      "p_value": <number or null>,
      "adjusted_analysis": <true/false/null>,
      "adjustment_variables": "<string or null>",
      "direct_mortality_evidence": <true/false>,
      "evidence_passage": "<exact quote from the given text>",
      "section": "<'abstract'/'results'/'discussion'/'table'/'figure' or null>",
      "table_or_figure_reference": "<string or null>"
    }}
  ]
}}
If NO extractable features are found, return "features": []."""


def load_source_text(pmid):
    parts = []
    ab_path = ABSTRACTS_DIR / f"{pmid}.txt"
    ft_path = FULLTEXT_DIR / f"{pmid}.txt"
    if ab_path.exists():
        parts.append("=== ABSTRACT ===\n" + ab_path.read_text())
    if ft_path.exists():
        parts.append("=== FULL TEXT ===\n" + ft_path.read_text())
    text = "\n\n".join(parts)
    truncated = len(text) > MAX_CHARS
    return text[:MAX_CHARS], truncated, bool(ft_path.exists())


def normalize_ws(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


def verify_passage(passage, source_text_norm):
    p = normalize_ws(passage)
    if len(p) < 15:
        return False
    if p in source_text_norm:
        return True
    # fuzzy fallback: require a long enough contiguous substring to match
    words = p.split()
    if len(words) < 6:
        return False
    for span in (12, 8, 6):
        if len(words) >= span:
            candidate = " ".join(words[:span])
            if candidate in source_text_norm:
                return True
    return False


def extract_one(client, pmid, title):
    text, truncated, has_fulltext = load_source_text(pmid)
    if not text.strip():
        return None
    user_msg = f"PMID: {pmid}\nTitle: {title}\n\n{text}"
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}],
        temperature=0.0, max_tokens=4000, response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else None
    if parsed is None:
        return None

    source_norm = normalize_ws(text)
    for feat in parsed.get("features", []):
        feat["passage_verified"] = verify_passage(feat.get("evidence_passage", ""), source_norm)
        feat["manual_review_required"] = (not feat["passage_verified"]) or parsed.get("llm_confidence") == "low"

    parsed["pmid"] = pmid
    parsed["full_text_or_abstract"] = "full_text" if has_fulltext else "abstract"
    parsed["source_text_truncated"] = truncated
    return parsed


def main():
    meta = pd.read_csv(SEMANTIC / "literature_prior" / "abstracts" / "articles_metadata.csv", dtype={"pmid": str})
    meta = meta[meta["has_abstract"] | meta["full_text_available"]]
    print(f"[lit04] extracting structured evidence for {len(meta)} PMIDs (has abstract or full text)")

    client = OpenAI(api_key=_openai_api_key())
    results = {}
    errors = []

    def worker(row):
        pmid, title = row["pmid"], row["title"]
        for attempt in range(2):
            try:
                return pmid, extract_one(client, pmid, title)
            except Exception as e:
                if attempt == 0:
                    time.sleep(2)
                    continue
                return pmid, {"pmid": pmid, "error": str(e)}

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, row): row["pmid"] for _, row in meta.iterrows()}
        done = 0
        for fut in as_completed(futures):
            pmid, res = fut.result()
            if res is not None:
                results[pmid] = res
                if "error" in res:
                    errors.append(pmid)
            done += 1
            if done % 25 == 0:
                print(f"[lit04]   {done}/{len(meta)} processed ({len(errors)} errors so far)")

    with open(EXTRACTION_DIR / "evidence_raw.jsonl", "w") as f:
        for pmid, rec in results.items():
            f.write(json.dumps(rec) + "\n")

    flat_rows = []
    for pmid, rec in results.items():
        if "error" in rec:
            continue
        study_fields = {k: v for k, v in rec.items() if k != "features"}
        for feat in rec.get("features", []):
            row = dict(study_fields)
            row.update(feat)
            flat_rows.append(row)
    flat_df = pd.DataFrame(flat_rows)
    flat_df.to_csv(EXTRACTION_DIR / "evidence_flat.csv", index=False)

    n_verified = int(flat_df["passage_verified"].sum()) if len(flat_df) else 0
    manifest = {
        "generated": today_str(), "model": GEN_MODEL, "n_papers_attempted": len(meta),
        "n_papers_succeeded": len(results) - len(errors), "n_papers_errored": len(errors),
        "n_feature_rows": len(flat_df), "n_passage_verified": n_verified,
        "n_passage_unverified": len(flat_df) - n_verified,
        "n_direct_human_mortality_studies": int(sum(1 for r in results.values() if r.get("direct_human_mortality_study"))),
    }
    with open(EXTRACTION_DIR / "extraction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[lit04] {manifest}")
    print(f"[lit04] wrote {EXTRACTION_DIR}/")


if __name__ == "__main__":
    main()
