"""
lit05_normalize_features.py — Phase II / ontology normalization (spec
sec. 21-30): map every unique literature-reported feature (from lit04's
evidence_flat.csv) onto this project's ACTUAL data products --
MetaPhlAn, HUMAnN3, CARD/RGI (contig- and MAG-level), MAG taxonomy
(GTDB), eggNOG (gene-symbol level), antiSMASH, VFDB.

Mapping classes (spec sec. 21), assigned independently per target
ontology -- a feature can be EXACT in one ontology and UNMAPPABLE in
another:
  EXACT                  - literal / near-literal name match at the
                            correct rank/category in our data
  SYNONYM                - matched via a documented synonym (e.g.
                            "vancomycin" -> CARD drug class "glycopeptide")
  PARENT_CHILD            - literature reports a different taxonomic rank
                            than what matched (e.g. genus reported, only
                            species-level column available or vice versa)
  FUNCTIONAL_APPROXIMATION - literature concept is broader/vaguer than the
                            specific data column used as a proxy (e.g.
                            "SCFA production" -> a specific MetaCyc SCFA
                            pathway)
  MECHANISTIC_ONLY        - biologically relevant but no corresponding
                            data column exists in our project at all
  UNMAPPABLE              - no reasonable mapping found

This is DATA-ONLY normalization -- no hospital_mortality is read, no
selection/filtering by evidence tier or performance happens here (that is
lit06). A literature feature mentioned by only 1 paper is mapped exactly
the same way as one mentioned by 20.

Outputs:
  literature_prior/ontology/feature_ontology_mapping.csv   (one row per unique (feature_view, normalized_feature_name))
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import SEMANTIC, ICU_PUBLIC, METAPHLAN_DIR, HUMANN_DIR

EXTRACTION_DIR = SEMANTIC / "literature_prior" / "extraction"
ONTOLOGY_DIR = SEMANTIC / "literature_prior" / "ontology"
ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

MERGE_OUTPUT = ICU_PUBLIC / "merge" / "output"
MAG_SUMMARY = MERGE_OUTPUT / "mag_summary"

# ---------------------------------------------------------------------
# reference vocabularies
# ---------------------------------------------------------------------
TAX_LEVELS = {}
for level in ["species", "genus", "family", "order", "phylum"]:
    cols = pd.read_csv(METAPHLAN_DIR / f"metaphlan3_{level}_merged.tsv", sep="\t", nrows=0).columns.tolist()
    TAX_LEVELS[level] = [c for c in cols if c != "clade_name"]
# need actual clade names, not sample ids -- read clade_name column instead
TAX_LEVELS = {}
for level in ["species", "genus", "family", "order", "phylum"]:
    df = pd.read_csv(METAPHLAN_DIR / f"metaphlan3_{level}_merged.tsv", sep="\t", usecols=[0])
    TAX_LEVELS[level] = df.iloc[:, 0].tolist()

pathway_names = pd.read_csv(HUMANN_DIR / "humann_pathabundance_wide.unstratified.tsv", sep="\t", usecols=[0]).iloc[:, 0].tolist()
pathway_names = [p for p in pathway_names if p not in ("UNMAPPED", "UNINTEGRATED")]

rgi_mag_rows = pd.read_csv(MAG_SUMMARY / "rgi" / "rgi_drug_class_per_sample.tsv", sep="\t", usecols=["feature"])["feature"].tolist()
rgi_contig_rows = pd.read_csv(MERGE_OUTPUT / "R_ready" if False else MAG_SUMMARY / "rgi_contig" / "rgi_contig_drug_class_per_sample.tsv", sep="\t", usecols=["feature"])["feature"].tolist()
vfdb_rows = pd.read_csv(MAG_SUMMARY / "vfdb" / "vfdb_category_per_sample.tsv", sep="\t", usecols=["feature"])["feature"].tolist()
bgc_rows = pd.read_csv(MAG_SUMMARY / "antismash" / "antismash_bgc_types_per_sample.tsv", sep="\t", usecols=["feature"])["feature"].tolist()

gtdb = pd.read_csv(MAG_SUMMARY / "gtdb" / "gtdb_mag_taxonomy.tsv", sep="\t")
gtdb_genus_counts = gtdb["genus"].dropna().value_counts()
gtdb_species_counts = gtdb["species"].dropna().value_counts()

eggnog_gene_symbols = set()
try:
    for chunk in pd.read_csv(SEMANTIC / "inventory" / "eggnog_gene_pfam_by_mag.tsv.gz", sep="\t",
                              usecols=["type", "value"], chunksize=500000):
        eggnog_gene_symbols.update(chunk.loc[chunk.type == "gene_symbol", "value"].dropna().unique())
except FileNotFoundError:
    pass
print(f"[lit05] reference vocab loaded: {sum(len(v) for v in TAX_LEVELS.values())} taxa, {len(pathway_names)} pathways, "
      f"{len(rgi_mag_rows)} MAG-RGI classes, {len(vfdb_rows)} VFDB categories, {len(bgc_rows)} BGC types, "
      f"{len(eggnog_gene_symbols)} eggNOG gene symbols, {gtdb_genus_counts.shape[0]} GTDB genera")

DRUGCLASS_SYNONYMS = {
    "vancomycin": "glycopeptide", "teicoplanin": "glycopeptide",
    "carbapenem-resistant": "carbapenem", "carbapenemase": "carbapenem",
    "esbl": "cephalosporin", "extended-spectrum beta-lactamase": "cephalosporin",
    "colistin": "peptide", "polymyxin": "peptide",
    "methicillin": "penicillin", "oxacillin": "penicillin",
    "erythromycin": "macrolide", "azithromycin": "macrolide", "clarithromycin": "macrolide",
    "ciprofloxacin": "fluoroquinolone", "levofloxacin": "fluoroquinolone", "quinolone": "fluoroquinolone",
    "rifampin": "rifamycin", "rifampicin": "rifamycin", "linezolid": "oxazolidinone",
    "trimethoprim": "diaminopyrimidine", "sulfamethoxazole": "diaminopyrimidine",
    "gentamicin": "aminoglycoside", "amikacin": "aminoglycoside", "tobramycin": "aminoglycoside",
    "tetracycline": "tetracycline", "doxycycline": "tetracycline", "minocycline": "tetracycline",
    "clindamycin": "lincosamide", "chloramphenicol": "phenicol",
}
BGC_SYNONYMS = {"bacteriocin": "RiPP-like", "lanthipeptide": "lanthipeptide-class-i", "polyketide": "T1PKS"}
STOPWORDS = {"gene", "genes", "resistant", "resistance", "antibiotic", "antibiotics", "bacteria", "bacterial",
             "infection", "infections", "pathogen", "pathogens", "multidrug", "mdr", "cluster", "clusters",
             "system", "factor", "factors", "of", "the", "and", "gut", "colonization", "expression", "burden",
             "production", "producing", "producer", "producers", "related", "associated"}


def keywords(name):
    return [w for w in re.split(r"[\s\-()/,]+", name.lower()) if len(w) > 3 and w not in STOPWORDS]


def keyword_match(name_lower, row_names, synonyms=None):
    text = name_lower
    for syn, repl in (synonyms or {}).items():
        if syn in text:
            text += " " + repl
    for row in row_names:
        rl = row.lower()
        if text in rl or rl in text:
            return row, "EXACT" if text == rl else "SYNONYM" if synonyms and any(s in text for s in synonyms) else "FUNCTIONAL_APPROXIMATION"
    kws = keywords(text)
    for row in row_names:
        rl = row.lower()
        if any(kw in rl for kw in kws):
            cls = "SYNONYM" if synonyms and any(s in name_lower for s in synonyms) else "FUNCTIONAL_APPROXIMATION"
            return row, cls
    return None, None


def map_taxon(name):
    key = name.strip().lower()
    result = {"MetaPhlAn_mapping": "UNMAPPABLE", "MetaPhlAn_target": None,
              "MAG_mapping": "UNMAPPABLE", "MAG_target": None}
    matched_level = None
    for level in ["species", "genus", "family", "order", "phylum"]:
        for clade in TAX_LEVELS[level]:
            leaf = clade.lower().split("|")[-1].replace("_", " ")
            if key == leaf.split("__")[-1] if "__" in leaf else key == leaf:
                result["MetaPhlAn_mapping"], result["MetaPhlAn_target"] = "EXACT", f"{level}:{clade}"
                matched_level = level
                break
            if key in leaf:
                result["MetaPhlAn_mapping"], result["MetaPhlAn_target"] = "PARENT_CHILD", f"{level}:{clade}"
                matched_level = level
        if matched_level:
            break
    # GTDB MAG-level presence (genus/species)
    if key.istitle() or " " in key:
        hits = gtdb_species_counts[gtdb_species_counts.index.str.lower().str.contains(re.escape(key), na=False)]
        if len(hits):
            result["MAG_mapping"], result["MAG_target"] = "EXACT", f"species:{hits.index[0]} (n_MAGs={int(hits.iloc[0])})"
    if result["MAG_mapping"] == "UNMAPPABLE":
        hits = gtdb_genus_counts[gtdb_genus_counts.index.str.lower() == key]
        if len(hits):
            result["MAG_mapping"], result["MAG_target"] = "EXACT", f"genus:{hits.index[0]} (n_MAGs={int(hits.iloc[0])})"
        else:
            partial = gtdb_genus_counts[gtdb_genus_counts.index.str.lower().str.contains(re.escape(key), na=False)] if len(key) > 3 else pd.Series(dtype=int)
            if len(partial):
                result["MAG_mapping"], result["MAG_target"] = "PARENT_CHILD", f"genus:{partial.index[0]} (n_MAGs={int(partial.iloc[0])})"
    return result


def map_function(name):
    key = name.strip().lower()
    result = {"HUMAnN_mapping": "UNMAPPABLE", "HUMAnN_target": None,
              "eggNOG_mapping": "UNMAPPABLE", "eggNOG_target": None}
    hit, cls = keyword_match(key, pathway_names)
    if hit:
        result["HUMAnN_mapping"], result["HUMAnN_target"] = cls, hit
    # gene-symbol-level eggNOG match (exact case-insensitive)
    for sym in eggnog_gene_symbols:
        if key == sym.lower():
            result["eggNOG_mapping"], result["eggNOG_target"] = "EXACT", f"gene_symbol:{sym}"
            break
    if result["HUMAnN_mapping"] == "UNMAPPABLE" and result["eggNOG_mapping"] == "UNMAPPABLE":
        result["HUMAnN_mapping"] = "MECHANISTIC_ONLY"
    return result


def map_resistome(name):
    key = name.strip().lower()
    result = {"RGI_mapping": "UNMAPPABLE", "RGI_target": None}
    hit, cls = keyword_match(key, rgi_mag_rows, DRUGCLASS_SYNONYMS)
    if hit:
        result["RGI_mapping"], result["RGI_target"] = cls, f"MAG-level:{hit}"
    else:
        hit, cls = keyword_match(key, rgi_contig_rows, DRUGCLASS_SYNONYMS)
        if hit:
            result["RGI_mapping"], result["RGI_target"] = cls, f"contig-level:{hit}"
    return result


def map_genome_resolved(name):
    key = name.strip().lower()
    result = {"VFDB_mapping": "UNMAPPABLE", "VFDB_target": None,
              "antiSMASH_mapping": "UNMAPPABLE", "antiSMASH_target": None,
              "RGI_mapping": "UNMAPPABLE", "RGI_target": None}
    hit, cls = keyword_match(key, vfdb_rows)
    if hit:
        result["VFDB_mapping"], result["VFDB_target"] = cls, hit
    hit, cls = keyword_match(key, bgc_rows, BGC_SYNONYMS)
    if hit:
        result["antiSMASH_mapping"], result["antiSMASH_target"] = cls, hit
    hit, cls = keyword_match(key, rgi_mag_rows, DRUGCLASS_SYNONYMS)
    if hit:
        result["RGI_mapping"], result["RGI_target"] = cls, f"MAG-level:{hit}"
    if all(result[k] == "UNMAPPABLE" for k in ["VFDB_mapping", "antiSMASH_mapping", "RGI_mapping"]):
        result["VFDB_mapping"] = "MECHANISTIC_ONLY"
    return result


def main():
    flat = pd.read_csv(EXTRACTION_DIR / "evidence_flat.csv", dtype={"pmid": str})
    grouped = flat.groupby(["feature_view", "normalized_feature_name"]).agg(
        n_mentions=("pmid", "nunique"),
        n_papers_list=("pmid", lambda s: ";".join(sorted(set(s)))),
        example_reported_name=("feature_name_as_reported", "first"),
        directions=("direction", lambda s: ";".join(sorted(set(str(d) for d in s)))),
    ).reset_index()
    print(f"[lit05] {len(grouped)} unique (view, normalized_feature_name) candidates from {flat.pmid.nunique()} papers")

    rows = []
    for _, r in grouped.iterrows():
        view, name = r["feature_view"], r["normalized_feature_name"]
        base = {"feature_view": view, "normalized_feature_name": name, "n_mentions": r["n_mentions"],
                "example_reported_name": r["example_reported_name"], "directions_reported": r["directions"],
                "MetaPhlAn_mapping": "UNMAPPABLE", "MetaPhlAn_target": None,
                "HUMAnN_mapping": "UNMAPPABLE", "HUMAnN_target": None,
                "RGI_mapping": "UNMAPPABLE", "RGI_target": None,
                "MAG_mapping": "UNMAPPABLE", "MAG_target": None,
                "eggNOG_mapping": "UNMAPPABLE", "eggNOG_target": None,
                "antiSMASH_mapping": "UNMAPPABLE", "antiSMASH_target": None,
                "VFDB_mapping": "UNMAPPABLE", "VFDB_target": None}
        if not isinstance(name, str) or not name.strip():
            rows.append(base)
            continue
        if view == "taxonomy":
            base.update(map_taxon(name))
        elif view == "function":
            base.update(map_function(name))
        elif view == "resistome":
            base.update(map_resistome(name))
        elif view == "genome_resolved":
            base.update(map_genome_resolved(name))
        elif view == "diversity":
            base["MetaPhlAn_mapping"], base["MetaPhlAn_target"] = "EXACT", "computed_directly (Shannon/Simpson/richness on species table)"
        elif view == "ratio":
            key = name.lower()
            if "firmicutes" in key and "bacteroidetes" in key:
                base["MetaPhlAn_mapping"], base["MetaPhlAn_target"] = "EXACT", "computed_directly (phylum ratio)"
        else:
            base["MetaPhlAn_mapping"] = "UNMAPPABLE"
        rows.append(base)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(ONTOLOGY_DIR / "feature_ontology_mapping.csv", index=False)

    any_mapped = out_df[[c for c in out_df.columns if c.endswith("_mapping")]].apply(
        lambda row: any(v not in ("UNMAPPABLE",) for v in row), axis=1)
    print(f"[lit05] {any_mapped.sum()}/{len(out_df)} unique features have >=1 non-UNMAPPABLE ontology mapping")
    for col in [c for c in out_df.columns if c.endswith("_mapping")]:
        print(f"[lit05]   {col}: {out_df[col].value_counts().to_dict()}")
    print(f"[lit05] wrote {ONTOLOGY_DIR}/feature_ontology_mapping.csv")


if __name__ == "__main__":
    main()
