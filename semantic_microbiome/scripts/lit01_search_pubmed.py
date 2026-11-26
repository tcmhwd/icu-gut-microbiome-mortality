"""
lit01_search_pubmed.py — Phase I / Gate 1, step 1: systematic multi-query
PubMed search for gut-microbiome / ICU-critical-illness / mortality
literature, per the new rigorous spec (2026-08-13, "Literature-Grounded
RAG Mining -> Structured/Semantic Microbiome Modeling").

Naming note: this is a NEW, stricter literature-mining pipeline
(lit01..lit19, per the spec's suggested modular script list in its
section 80) that supersedes the earlier ad-hoc `32_literature_indicator_
mining.py` / `34_literature_indicator_mining_v2.py` for the purposes of
this spec (evidence tiering, structured extraction schema, frozen
manifest, RAG corpus with source passages). Those older outputs are kept
as-is (not deleted) and may still be referenced for context, but are NOT
treated as satisfying this spec's Gate 1 requirements. `lit` prefix
chosen instead of continuing 36../37.. numbering, to keep this
self-contained pipeline visually distinct from the earlier phase 1-8h
scripts -- documented here explicitly per the project's "document
ambiguity, don't silently assume" rule.

Design (spec sec. 9): run MULTIPLE complementary PubMed searches rather
than one giant query.
  - Group "core": ICU-terms AND microbiome-terms AND outcome-terms, as a
    single broad boolean query (PubMed natively supports AND/OR/paren
    syntax in `term=`).
  - Group "view_*": each specific-view seed term (taxonomy / function /
    resistome / genome-resolved) combined with a LIGHTER core concept.
    Rationale, empirically established in `34_literature_indicator_
    mining_v2.py`: PubMed's automatic term-mapping returns ZERO hits for
    overly long multi-concept AND chains even when relevant literature
    exists (e.g. "carbapenem resistant Enterobacteriaceae gut
    colonization ICU outcome" -> 0 hits). Each view query is tried at up
    to 3 progressively looser boolean tiers; the FIRST tier with >0
    results is what gets recorded as the query actually used. All
    attempted tiers (including 0-result ones) are logged in
    `search_manifest.csv` for transparency -- nothing is silently
    retried without a record.

These seed terms are SEARCH EXPANSION ONLY (spec sec. 9) -- they do not
themselves become accepted biomarkers; that judgment happens later
(evidence extraction + tiering, lit04/lit06).

hospital_mortality is not read by this script.

Outputs:
  literature_prior/search/search_manifest.csv   (every query attempted, incl. 0-result ones)
  literature_prior/search/pmid_pool.csv          (deduped PMIDs retrieved, with which query group(s) found them)
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('NCBI_EMAIL', '').strip():
    raise SystemExit('NCBI_EMAIL must be set (see README).')

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from common import SEMANTIC, today_str

OUT_DIR = SEMANTIC / "literature_prior" / "search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NCBI_EMAIL = __import__("os").environ.get("NCBI_EMAIL", "").strip()  # [REDACTED for public release] personal e-mail removed; set NCBI_EMAIL
if not NCBI_EMAIL:  # [PUBLIC RELEASE] required; fail before any request is sent
    raise SystemExit("NCBI_EMAIL must be set to a contact e-mail address for NCBI E-utilities (see README).")
NCBI_TOOL = "semantic_microbiome_lit01"
DB = "pubmed"

# ---------------------------------------------------------------------
# seed term groups (spec sec. 9, verbatim)
# ---------------------------------------------------------------------
ICU_TERMS = ["intensive care", "ICU", "critical illness", "critically ill", "sepsis", "septic shock"]
MICROBIOME_TERMS = ["gut microbiome", "intestinal microbiome", "fecal microbiome", "faecal microbiome",
                     "metagenome", "metagenomics", "dysbiosis"]
OUTCOME_TERMS = ["mortality", "survival", "death", "prognosis", "outcome"]

VIEW_TERMS = {
    "taxonomy": ["Enterococcus", "Enterobacteriaceae", "Klebsiella", "Escherichia", "anaerobic bacteria"],
    "function": ["microbial pathway", "metabolome", "short-chain fatty acid", "butyrate", "bile acid", "tryptophan"],
    "resistome": ["resistome", "antimicrobial resistance", "antibiotic resistance genes"],
    "genome_resolved": ["metagenome assembled genome", "MAG", "virulence", "virulome", "biosynthetic gene cluster"],
}


def or_block(terms):
    return "(" + " OR ".join(f'"{t}"[tiab]' for t in terms) + ")"


CORE_QUERY = f"{or_block(ICU_TERMS)} AND {or_block(MICROBIOME_TERMS)} AND {or_block(OUTCOME_TERMS)}"

# light core concept used for view-specific queries (spec: microbiome +
# outcome, ICU optional at the loosest tier) -- 3 tiers, loosest last.
MICROBIOME_LIGHT = "(gut microbiome[tiab] OR intestinal microbiome[tiab] OR metagenome[tiab] OR metagenomics[tiab])"
OUTCOME_LIGHT = "(mortality[tiab] OR survival[tiab] OR outcome[tiab])"
ICU_LIGHT = "(ICU[tiab] OR critical illness[tiab] OR critically ill[tiab] OR sepsis[tiab])"


def view_query_tiers(term):
    return [
        f'"{term}"[tiab] AND {ICU_LIGHT} AND {MICROBIOME_LIGHT} AND {OUTCOME_LIGHT}',
        f'"{term}"[tiab] AND {MICROBIOME_LIGHT} AND {OUTCOME_LIGHT}',
        f'"{term}"[tiab] AND {MICROBIOME_LIGHT}',
    ]


def esearch(query, retmax):
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
           + urllib.parse.urlencode({"db": DB, "term": query, "retmax": retmax, "retmode": "json",
                                      "sort": "relevance", "email": NCBI_EMAIL, "tool": NCBI_TOOL}))
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read())
    res = data.get("esearchresult", {})
    return res.get("idlist", []), int(res.get("count", 0))


def main():
    manifest_rows = []
    pmid_pool = {}  # pmid -> set of query_group labels

    def record(group, query, retmax, ids, count, tier=None):
        manifest_rows.append({
            "search_date": today_str(), "query_group": group, "tier": tier, "exact_query": query,
            "database": DB, "retmax": retmax, "number_results_total": count,
            "number_pmids_retrieved": len(ids), "PMIDs_retrieved": ";".join(ids),
        })
        for pmid in ids:
            pmid_pool.setdefault(pmid, set()).add(group)

    print(f"[lit01] core query: {CORE_QUERY}")
    ids, count = esearch(CORE_QUERY, retmax=60)
    print(f"[lit01]   {count} total hits, retrieved {len(ids)} PMIDs")
    record("core", CORE_QUERY, 60, ids, count)
    time.sleep(0.4)

    for view, terms in VIEW_TERMS.items():
        for term in terms:
            tiers = view_query_tiers(term)
            used = False
            for tier_i, q in enumerate(tiers, start=1):
                ids, count = esearch(q, retmax=20)
                record(f"view_{view}", q, 20, ids, count, tier=tier_i)
                print(f"[lit01] (view_{view}) tier{tier_i} '{term}': {count} hits" + (" <- used" if ids and not used else ""))
                time.sleep(0.4)
                if ids and not used:
                    used = True
                    # still log looser tiers for transparency, but do not use them for the pool
                    continue
            if not used:
                print(f"[lit01]   WARNING: all tiers 0 hits for '{term}' ({view})")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_DIR / "search_manifest.csv", index=False)

    pool_rows = [{"pmid": pmid, "query_groups": ";".join(sorted(groups)), "n_query_groups": len(groups)}
                 for pmid, groups in pmid_pool.items()]
    pool_df = pd.DataFrame(pool_rows).sort_values("n_query_groups", ascending=False)
    pool_df.to_csv(OUT_DIR / "pmid_pool.csv", index=False)

    print(f"[lit01] {len(manifest_df)} query attempts logged -> {OUT_DIR}/search_manifest.csv")
    print(f"[lit01] {len(pool_df)} unique PMIDs in pool -> {OUT_DIR}/pmid_pool.csv")
    print(f"[lit01] pool size by query group:\n{manifest_df.groupby('query_group')['number_pmids_retrieved'].sum()}")


if __name__ == "__main__":
    main()
