"""Post hoc Holm adjustment across the three Table 2 clinical-model contrasts.

Written for the public code release (v1.0.1); it was not part of the original
analysis tree. In the original analysis the three adjusted P values reported in
Table 2 (0.440, 0.135, 0.230) were obtained by hand from the nominal paired
bootstrap P values written by
`clinical metadata/scripts/45_phase_x_c1_c2_nested_cv_corrected_v4.py --aggregate`.
This script reproduces that arithmetic from the same file.

Rows used (preprocessing_variant, model) -> Table 2 row:
  V0 (= REF), C1_clinical_plus_M1_LiteratureCore -> literature-derived score
  V4_STRICT,  C2a_clinical_plus_M3_LR            -> data-driven logistic regression
  V3,         C2b_clinical_plus_M3_CatBoost      -> data-driven CatBoost
Nominal P values: 0.44, 0.045, 0.115.
"""
import os

import numpy as np
import pandas as pd

BASE = "/home/tcmhwd/microbiome/ICU_public"
BOOT_CSV = os.path.join(BASE, "clinical metadata", "clinical_ai", "reports",
                        "phase_x_c1_c2_nested_bootstrap_delta_v4.csv")

ROWS = [
    ("V0", "C1_clinical_plus_M1_LiteratureCore"),
    ("V4_STRICT", "C2a_clinical_plus_M3_LR"),
    ("V3", "C2b_clinical_plus_M3_CatBoost"),
]
EXPECTED_ADJ = {ROWS[0]: 0.440, ROWS[1]: 0.135, ROWS[2]: 0.230}


def holm(p):
    """Holm-Bonferroni step-down adjusted P values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main():
    boot = pd.read_csv(BOOT_CSV)
    sel = []
    for variant, model in ROWS:
        hit = boot[(boot["preprocessing_variant"] == variant) & (boot["model"] == model)]
        if len(hit) != 1:
            raise SystemExit(f"expected exactly one row for {variant}/{model} in {BOOT_CSV}, found {len(hit)}")
        sel.append(hit.iloc[0])
    out = pd.DataFrame({
        "preprocessing_variant": [r["preprocessing_variant"] for r in sel],
        "model": [r["model"] for r in sel],
        "delta_AUROC": [r["delta_AUROC"] for r in sel],
        "nominal_boot_p": [float(r["delta_AUROC_boot_p"]) for r in sel],
    })
    out["holm_adjusted_p"] = holm(out["nominal_boot_p"].values)
    print(out.to_string(index=False))
    for (variant, model), exp in EXPECTED_ADJ.items():
        got = float(out.loc[(out.preprocessing_variant == variant) & (out.model == model), "holm_adjusted_p"].iloc[0])
        assert abs(got - exp) < 5e-4, f"{variant}/{model}: Holm P {got:.4f} != reported {exp:.3f}"
    print("Holm-adjusted P values match Table 2 (0.440, 0.135, 0.230).")


if __name__ == "__main__":
    main()
