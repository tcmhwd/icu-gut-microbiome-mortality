"""
lit03_build_rag_corpus.py — Phase I / Gate 1, step 3: chunk every fetched
abstract/full-text into a local RAG corpus, then embed each chunk
(cached by SHA-256 so nothing is ever re-embedded), per spec sec. 17-18.

Chunking (spec: "semantically appropriate", "do not mix passages from
multiple articles into one chunk", "store the exact original text"):
  - abstract: each PubMed-labelled subsection (BACKGROUND/METHODS/
    RESULTS/CONCLUSIONS, or the whole abstract if unlabelled) is its own
    chunk; a subsection longer than ~1800 chars is split further on
    sentence boundaries.
  - full text: paragraphs (already one-per-line from lit02's <p>
    extraction) are greedily merged up to ~1500 chars per chunk, never
    split mid-paragraph unless a single paragraph alone exceeds ~2500
    chars (rare; split on sentence boundaries then).
  - every chunk is tagged with its single source PMID -- never merged
    across articles.

Embedding (spec sec. 18): backend/model configurable via env vars,
default OpenAI text-embedding-3-large. Key read ONLY from the
OPENAI_API_KEY environment variable (public release; the original read
a local key file) -- never hard-coded, never
printed/logged). Cached by SHA-256(text) in rag_index/embedding_cache/
so re-running this script (e.g. after adding more PMIDs later) never
re-embeds identical text.

hospital_mortality is not read by this script.

Outputs:
  literature_prior/rag_index/corpus_chunks.csv     (pmid, pmcid, doi, title, year, journal, source, section, chunk_index, sha256, n_chars)
  literature_prior/rag_index/embeddings.npy         (N x dim, row order == corpus_chunks.csv row order)
  literature_prior/rag_index/embedding_cache/*.npy  (per-chunk cache, keyed by sha256)
  literature_prior/rag_index/embedding_manifest.json
"""
from __future__ import annotations
import os as _pre_os  # [PUBLIC RELEASE] preflight: validate required env vars before any import with side effects, input read or output write
if not _pre_os.environ.get('OPENAI_API_KEY', '').strip():
    raise SystemExit('OPENAI_API_KEY must be set (see README).')

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

from common import SEMANTIC, today_str

ABSTRACTS_DIR = SEMANTIC / "literature_prior" / "abstracts"
FULLTEXT_DIR = SEMANTIC / "literature_prior" / "fulltext"
RAG_DIR = SEMANTIC / "literature_prior" / "rag_index"
CACHE_DIR = RAG_DIR / "embedding_cache"
RAG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _openai_api_key() -> str:  # [PUBLIC RELEASE] required env var replaces the original local key-file path
    k = __import__("os").environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        raise SystemExit("OPENAI_API_KEY must be set (see README).")
    return k
EMBED_MODEL = "text-embedding-3-large"  # configurable default per spec sec. 18
EMBED_DIM = 1024  # matches this project's earlier embedding scripts (12/12b/12c)

MAX_CHUNK_CHARS = 1500
HARD_SPLIT_CHARS = 2500


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text)


def chunk_long_text(text, target=MAX_CHUNK_CHARS, hard_cap=HARD_SPLIT_CHARS):
    if len(text) <= hard_cap:
        return [text]
    sents = sentence_split(text)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > target:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur.strip())
    return out


def chunk_abstract(text):
    """One chunk per labelled subsection (or the whole thing if unlabelled)."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return []
    chunks = []
    for line in lines:
        chunks.extend([("abstract", c) for c in chunk_long_text(line)])
    return chunks


def chunk_fulltext(text):
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(p) > HARD_SPLIT_CHARS:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.extend(chunk_long_text(p))
            continue
        if cur and len(cur) + len(p) + 2 > MAX_CHUNK_CHARS:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + "\n\n" + p).strip() if cur else p
    if cur:
        chunks.append(cur.strip())
    return [("fulltext", c) for c in chunks]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_batch(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def main():
    meta = pd.read_csv(ABSTRACTS_DIR / "articles_metadata.csv", dtype={"pmid": str})
    print(f"[lit03] chunking corpus for {len(meta)} PMIDs")

    rows = []
    texts_to_embed = []
    for _, m in meta.iterrows():
        pmid = m["pmid"]
        ab_path = ABSTRACTS_DIR / f"{pmid}.txt"
        ft_path = FULLTEXT_DIR / f"{pmid}.txt"

        article_chunks = []
        if ab_path.exists():
            article_chunks.extend(chunk_abstract(ab_path.read_text()))
        if ft_path.exists():
            article_chunks.extend(chunk_fulltext(ft_path.read_text()))

        for idx, (section, text) in enumerate(article_chunks):
            h = sha256(text)
            rows.append({
                "pmid": pmid, "pmcid": m["pmcid"], "doi": m["doi"], "title": m["title"],
                "year": m["year"], "journal": m["journal"], "source": section,
                "chunk_index": idx, "sha256": h, "n_chars": len(text),
            })
            (RAG_DIR / "chunk_text").mkdir(exist_ok=True)
            chunk_text_path = RAG_DIR / "chunk_text" / f"{h}.txt"
            if not chunk_text_path.exists():
                chunk_text_path.write_text(text)
            if not (CACHE_DIR / f"{h}.npy").exists():
                texts_to_embed.append((h, text))

    corpus_df = pd.DataFrame(rows)
    corpus_df.to_csv(RAG_DIR / "corpus_chunks.csv", index=False)
    print(f"[lit03] {len(corpus_df)} chunks from {corpus_df.pmid.nunique()} PMIDs "
          f"({(corpus_df.source == 'abstract').sum()} abstract chunks, {(corpus_df.source == 'fulltext').sum()} fulltext chunks)")

    # dedupe texts_to_embed by hash (same chunk text can recur, e.g. identical
    # short abstract labels across different articles -- rare but possible)
    unique_new = {h: t for h, t in texts_to_embed}
    print(f"[lit03] {len(unique_new)} NEW unique chunks to embed (cache already covers the rest)")

    if unique_new:
        client = OpenAI(api_key=_openai_api_key())
        items = list(unique_new.items())
        batch_size = 100
        n_embedded = 0
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            hashes, texts = zip(*batch)
            vecs = embed_batch(client, list(texts))
            for h, v in zip(hashes, vecs):
                np.save(CACHE_DIR / f"{h}.npy", v)
            n_embedded += len(batch)
            print(f"[lit03]   embedded {n_embedded}/{len(items)}")

    # assemble final embeddings.npy in corpus_df row order, reading from cache
    embeddings = np.zeros((len(corpus_df), EMBED_DIM), dtype=np.float32)
    missing = 0
    for i, h in enumerate(corpus_df["sha256"]):
        p = CACHE_DIR / f"{h}.npy"
        if p.exists():
            embeddings[i] = np.load(p)
        else:
            missing += 1
    np.save(RAG_DIR / "embeddings.npy", embeddings)
    print(f"[lit03] wrote embeddings.npy ({embeddings.shape}), {missing} missing (should be 0)")

    manifest = {
        "generated": today_str(), "embedding_model": EMBED_MODEL, "embedding_dimension": EMBED_DIM,
        "n_chunks": len(corpus_df), "n_pmids": int(corpus_df.pmid.nunique()),
        "n_abstract_chunks": int((corpus_df.source == "abstract").sum()),
        "n_fulltext_chunks": int((corpus_df.source == "fulltext").sum()),
        "n_newly_embedded_this_run": len(unique_new),
    }
    with open(RAG_DIR / "embedding_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[lit03] wrote {RAG_DIR}/")


if __name__ == "__main__":
    main()
