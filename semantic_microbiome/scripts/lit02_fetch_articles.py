"""
lit02_fetch_articles.py — Phase I / Gate 1, step 2: fetch metadata +
abstracts (+ PMC full text where legally/programmatically available) for
every PMID in the lit01 search pool.

Spec sec. 8: PubMed/NCBI E-utilities primary; PMC for legally accessible
full text; never circumvent paywalls -- if only an abstract is available,
use the abstract and record full_text_available=false (never pretend full
text was reviewed).

Method:
  1. batch efetch (db=pubmed, retmode=xml) -> title/journal/year/abstract/
     DOI/PMCID-if-already-linked, chunks of 150 PMIDs.
  2. for PMIDs with no PMCID yet, batch elink (pubmed -> pmc) to find one.
  3. for every PMID with a PMCID, attempt efetch (db=pmc, retmode=xml) and
     extract <body> paragraph text. If PMC returns no body (embargoed /
     not actually OA), full_text_available stays false -- no fallback
     scraping, no pretending.

hospital_mortality is not read by this script.

Outputs:
  literature_prior/abstracts/articles_metadata.csv   (PMID, PMCID, DOI, title, journal, year, full_text_available)
  literature_prior/abstracts/{pmid}.txt               (abstract text, one file per PMID)
  literature_prior/fulltext/{pmid}.txt                (full text, only written when available)
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('NCBI_EMAIL', '').strip():
    raise SystemExit('NCBI_EMAIL must be set (see README).')

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from common import SEMANTIC

SEARCH_DIR = SEMANTIC / "literature_prior" / "search"
ABSTRACTS_DIR = SEMANTIC / "literature_prior" / "abstracts"
FULLTEXT_DIR = SEMANTIC / "literature_prior" / "fulltext"
ABSTRACTS_DIR.mkdir(parents=True, exist_ok=True)
FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)

NCBI_EMAIL = __import__("os").environ.get("NCBI_EMAIL", "").strip()  # [REDACTED for public release] personal e-mail removed; set NCBI_EMAIL
if not NCBI_EMAIL:  # [PUBLIC RELEASE] required; fail before any request is sent
    raise SystemExit("NCBI_EMAIL must be set to a contact e-mail address for NCBI E-utilities (see README).")
NCBI_TOOL = "semantic_microbiome_lit02"
CHUNK = 150


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def efetch_pubmed_xml(pmids):
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
                                      "email": NCBI_EMAIL, "tool": NCBI_TOOL}))
    with urllib.request.urlopen(url, timeout=60) as r:
        return ET.fromstring(r.read())


def parse_pubmed_article(art):
    pmid_el = art.find(".//PMID")
    pmid = pmid_el.text if pmid_el is not None else None

    title_el = art.find(".//ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else None

    journal_el = art.find(".//Journal/Title")
    journal = journal_el.text if journal_el is not None else None

    year = None
    for path in [".//JournalIssue/PubDate/Year", ".//JournalIssue/PubDate/MedlineDate"]:
        y_el = art.find(path)
        if y_el is not None and y_el.text:
            year = y_el.text[:4]
            break

    abstract_parts = []
    for ab in art.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        text = "".join(ab.itertext()).strip()
        if text:
            abstract_parts.append(f"{label}: {text}" if label else text)
    abstract = "\n".join(abstract_parts) if abstract_parts else None

    doi, pmcid = None, None
    for aid in art.findall(".//ArticleIdList/ArticleId"):
        if aid.get("IdType") == "doi":
            doi = aid.text
        elif aid.get("IdType") == "pmc":
            pmcid = aid.text
    if doi is None:
        for eloc in art.findall(".//ELocationID"):
            if eloc.get("EIdType") == "doi":
                doi = eloc.text

    return {"pmid": pmid, "title": title, "journal": journal, "year": year,
            "abstract": abstract, "doi": doi, "pmcid": pmcid}


def elink_pubmed_to_pmc(pmids):
    """Returns dict pmid -> pmcid for whichever PMIDs have a linked PMC record."""
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
           + urllib.parse.urlencode({"dbfrom": "pubmed", "db": "pmc", "id": pmids, "retmode": "xml",
                                      "email": NCBI_EMAIL, "tool": NCBI_TOOL}, doseq=True))
    # elink needs repeated id= params, not comma-joined, for reliable per-id linksets
    qs = "&".join(f"id={p}" for p in pmids)
    full_url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
                f"dbfrom=pubmed&db=pmc&retmode=xml&email={NCBI_EMAIL}&tool={NCBI_TOOL}&{qs}")
    with urllib.request.urlopen(full_url, timeout=60) as r:
        root = ET.fromstring(r.read())
    out = {}
    for linkset in root.findall("LinkSet"):
        src_pmid_el = linkset.find("IdList/Id")
        if src_pmid_el is None:
            continue
        src_pmid = src_pmid_el.text
        for linksetdb in linkset.findall("LinkSetDb"):
            for link in linksetdb.findall("Link/Id"):
                out[src_pmid] = link.text
                break
            break
    return out


def fetch_pmc_fulltext(pmcid):
    pmc_num = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
           + urllib.parse.urlencode({"db": "pmc", "id": pmc_num, "retmode": "xml",
                                      "email": NCBI_EMAIL, "tool": NCBI_TOOL}))
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    body = root.find(".//body")
    if body is None:
        return None
    paras = ["".join(p.itertext()).strip() for p in body.findall(".//p")]
    paras = [p for p in paras if p]
    return "\n\n".join(paras) if paras else None


def main():
    pool = pd.read_csv(SEARCH_DIR / "pmid_pool.csv", dtype={"pmid": str})
    pmids = pool["pmid"].tolist()
    print(f"[lit02] fetching metadata+abstracts for {len(pmids)} PMIDs")

    records = {}
    for i, chunk in enumerate(chunks(pmids, CHUNK)):
        root = efetch_pubmed_xml(chunk)
        n_this = 0
        for art in root.findall(".//PubmedArticle"):
            rec = parse_pubmed_article(art)
            if rec["pmid"]:
                records[rec["pmid"]] = rec
                n_this += 1
        print(f"[lit02]   batch {i+1}: {n_this} articles parsed")
        time.sleep(0.4)

    missing_pmcid = [p for p in pmids if p in records and not records[p].get("pmcid")]
    print(f"[lit02] {len(missing_pmcid)}/{len(pmids)} PMIDs have no PMCID from PubMed XML yet -- trying elink")
    for chunk in chunks(missing_pmcid, 100):
        try:
            found = elink_pubmed_to_pmc(chunk)
            for pmid, pmcid in found.items():
                if pmid in records:
                    records[pmid]["pmcid"] = pmcid
            print(f"[lit02]   elink batch: {len(found)}/{len(chunk)} resolved to PMCID")
        except Exception as e:
            print(f"[lit02]   WARNING: elink batch failed: {e}")
        time.sleep(0.4)

    n_fulltext = 0
    for pmid, rec in records.items():
        # always write abstract (even if None -> empty file skipped)
        if rec.get("abstract"):
            (ABSTRACTS_DIR / f"{pmid}.txt").write_text(rec["abstract"])

        rec["full_text_available"] = False
        if rec.get("pmcid"):
            try:
                fulltext = fetch_pmc_fulltext(rec["pmcid"])
                time.sleep(0.4)
                if fulltext and len(fulltext) > 500:  # sanity floor: a real body, not just a stub
                    (FULLTEXT_DIR / f"{pmid}.txt").write_text(fulltext)
                    rec["full_text_available"] = True
                    n_fulltext += 1
            except Exception as e:
                pass  # PMC record exists but not retrievable (embargoed etc.) -> stays False, no fallback

    print(f"[lit02] {n_fulltext}/{len(records)} PMIDs have usable PMC full text")

    meta_rows = []
    for pmid in pmids:
        rec = records.get(pmid, {"pmid": pmid})
        meta_rows.append({
            "pmid": pmid, "pmcid": rec.get("pmcid"), "doi": rec.get("doi"),
            "title": rec.get("title"), "journal": rec.get("journal"), "year": rec.get("year"),
            "has_abstract": bool(rec.get("abstract")), "full_text_available": rec.get("full_text_available", False),
        })
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(ABSTRACTS_DIR / "articles_metadata.csv", index=False)
    print(f"[lit02] {meta_df['has_abstract'].sum()}/{len(meta_df)} PMIDs have an abstract")
    print(f"[lit02] wrote {ABSTRACTS_DIR}/articles_metadata.csv")


if __name__ == "__main__":
    main()
