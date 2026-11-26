#!/usr/bin/env python3
"""62_phase_x_vre_vana_vanb_split.py -- review-triggered analysis PR-2 (POST_HOC_EXPLORATORY).
Pre-registered in semantic_microbiome/reports/paper1/ROUND1_PREREGISTRATION.md BEFORE this ran (trigger: reviewer R3-01).

The reported VRE result pools vanA and vanB (contig-level RGI/CARD gene presence; 416 patients pre-negative for VRE with resistome data,
44 gene-positive, 11 later culture-positive). Published rectal-swab studies report much lower PPV for vanB than vanA (a non-enterococcal faecal
anaerobe reservoir carries vanB). This script splits the exposure: vanA-any, vanB-any, vanA-only, vanB-only, both.

Reuses, unchanged, script 53's population definition (pre-negative, resistome-joined, SOFA-complete), outcome (post.VRE.1), broad-spectrum antibiotic
covariate and helper conventions (Wilson CIs; closed-form 2x2 OR with Haldane-Anscombe correction; Firth via R logistf for the adjusted OR).
Reproduction gate: pooled group must reproduce N=416, 44 gene-positive, 11 events, 6/44 vs 5/372 before any split row is written.
Not done: taxonomic assignment of the contigs carrying each gene (no contig-to-taxon table is used here; stated as a limitation).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config, mdro_data as M  # noqa: E402

OUT_CSV = os.path.join(config.DIR_REPORTS, "phase_x_vre_vana_vanb_split.csv")
OUT_MD = os.path.join(config.DIR_REPORTS, "phase_x_vre_vana_vanb_split.md")
R_FIRTH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "run_firth_logistf.R")
ABX = ["Carbapenem", "Glycopeptides", "cephems_3rd", "cephems_4th", "BLBLI", "CEF_AVI", "CEF_TAZ"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, (c - a) / den), min(1.0, (c + a) / den))


def or_2x2(x, y):
    a = int(((x == 1) & (y == 1)).sum()); b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum()); d = int(((x == 0) & (y == 0)).sum())
    corr = 0 in (a, b, c, d)
    if corr:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    lo = np.log(a * d / (b * c)); se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return np.exp(lo), np.exp(lo - 1.96 * se), np.exp(lo + 1.96 * se), 2 * (1 - norm.cdf(abs(lo / se))), corr


def firth(y, x, sofa, abx):
    df = pd.DataFrame({"y": y.astype(int), "VRE_gene": x.astype(float), "SOFA": sofa, "abx_broad": abx})
    fd, tmp = tempfile.mkstemp(suffix=".csv"); os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        p = subprocess.run([os.environ.get("RSCRIPT_BIN", "Rscript"), R_FIRTH_SCRIPT, tmp], capture_output=True, text=True, timeout=240)
    finally:
        os.remove(tmp)
    if p.returncode != 0 or "FIRTH_FIT_FAILED" in p.stdout:
        return None
    for line in p.stdout.splitlines():
        parts = line.strip().split(",")
        if parts[0] == "COEF" and parts[1] == "VRE_gene":
            return dict(OR=float(parts[3]), lo=float(parts[4]), hi=float(parts[5]), p=float(parts[6]))
    return None


mc = M.load_clinical().set_index("sample_list", drop=False)
aro, _ = M.load_resistome_tables("contig")
vanA = (aro["vanA"] > 0).astype(int).reindex(mc.index) if "vanA" in aro.columns else pd.Series(np.nan, index=mc.index)
vanB = (aro["vanB"] > 0).astype(int).reindex(mc.index) if "vanB" in aro.columns else pd.Series(np.nan, index=mc.index)
print("vanA column present:", "vanA" in aro.columns, "| vanB column present:", "vanB" in aro.columns)
pre_c, post_c, _ = M.ORGANISM_PREPOST["VRE"]
pre = pd.to_numeric(mc[pre_c], errors="coerce"); post = pd.to_numeric(mc[post_c], errors="coerce")
sofa = pd.to_numeric(mc["SOFA"], errors="coerce")
abx = (mc[ABX].apply(pd.to_numeric, errors="coerce") == 1).any(axis=1).astype(float)
pooled = ((vanA + vanB) > 0).astype(float).where(vanA.notna() & vanB.notna())
valid = (pre == 0) & pooled.notna() & post.notna() & sofa.notna()
y = post[valid].astype(int).values; n = len(y)
assert n == 416 and int(pooled[valid].sum()) == 44 and int(y.sum()) == 11, (n, int(pooled[valid].sum()), int(y.sum()))
xp = pooled[valid].astype(int).values
assert int(((xp == 1) & (y == 1)).sum()) == 6 and int(((xp == 0) & (y == 1)).sum()) == 5, "pooled 6/44 vs 5/372 not reproduced"
print("REPRODUCTION GATE PASSED: N=416, 44 gene-positive, 11 events, 6/44 vs 5/372")

a_, b_ = vanA[valid].astype(int).values, vanB[valid].astype(int).values
groups = {"pooled vanA-or-vanB (reproduces reported)": xp, "vanA present (any)": a_, "vanB present (any)": b_,
          "vanA only (vanB absent)": ((a_ == 1) & (b_ == 0)).astype(int), "vanB only (vanA absent)": ((b_ == 1) & (a_ == 0)).astype(int),
          "both vanA and vanB": ((a_ == 1) & (b_ == 1)).astype(int)}
rows = []
for name, x in groups.items():
    pos = int(x.sum()); tp = int(((x == 1) & (y == 1)).sum()); fn = int(((x == 0) & (y == 1)).sum())
    tn = int(((x == 0) & (y == 0)).sum()); fp = int(((x == 1) & (y == 0)).sum())
    row = dict(group=name, N=n, gene_positive=pos, events_in_positive=tp, events_in_negative=fn, PPV=round(tp / pos, 4) if pos else np.nan,
               PPV_ci_low=round(wilson(tp, pos)[0], 4) if pos else np.nan, PPV_ci_high=round(wilson(tp, pos)[1], 4) if pos else np.nan,
               NPV=round(tn / (tn + fn), 4), sensitivity=round(tp / (tp + fn), 4), specificity=round(tn / (tn + fp), 4))
    if 0 < pos < n:
        o = or_2x2(x, y)
        row.update(unadj_OR=round(o[0], 2), unadj_ci_low=round(o[1], 2), unadj_ci_high=round(o[2], 2), unadj_p=round(o[3], 4), haldane=o[4])
        if tp >= 1:
            f = firth(y, x, sofa[valid].values, abx[valid].values)
            if f:
                row.update(firth_adj_OR=round(f["OR"], 2), firth_ci_low=round(f["lo"], 2), firth_ci_high=round(f["hi"], 2), firth_p=round(f["p"], 4))
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(OUT_CSV, index=False)
md = ["# PR-2: vanA / vanB separation (POST_HOC_EXPLORATORY; pre-registered before running)", "",
      "N = 416 patients pre-negative for VRE with contig-level resistome data and complete SOFA; outcome = later VRE culture positivity (broad any-Enterococcus definition, 11 events).",
      "Firth adjusted for SOFA + broad-spectrum antibiotic exposure. Wilson intervals. Contig taxonomy not assessed.", "", res.to_markdown(index=False)]
open(OUT_MD, "w").write("\n".join(md))
print(res.to_string(index=False))
