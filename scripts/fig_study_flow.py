#!/usr/bin/env python
"""Render the study-flow figure (Figure 1) as a nested screening funnel, in the
same order as Table 1: adult ICU admissions, diagnosis-based exclusions, computable
baseline creatinine, KDIGO AKI estimation cohort, classifiable MAKE-H (confirmatory
analysis population). Both cohorts side by side.

Reads outputs/{mimic,eicu}_screening_log.csv and the classifiable counts.
Writes outputs/figures/fig5_study_flow.png and .pdf.

Usage:
    python scripts/fig_study_flow.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# classifiable MAKE-H (confirmatory analysis population); from outcome labelling
CLASSIFIABLE = {"mimic": 34889, "eicu": 56135}
MIMIC = "#1b4965"
EICU = "#9e2a2b"


def log(cohort: str) -> dict:
    d = {}
    with open(OUT / f"{cohort}_screening_log.csv", newline="") as f:
        for row in csv.DictReader(f):
            d[row["step"]] = int(row["n"])
    return d


def panel(ax, cohort: str, title: str, color: str):
    s = log(cohort)
    main = [
        ("Adult ICU admissions", s["adult_icu_admissions"]),
        ("With computable baseline\ncreatinine", s["with_computable_baseline"]),
        ("KDIGO AKI\n(estimation cohort)", s["kdigo_aki_creatinine"]),
        ("Classifiable MAKE-H\n(confirmatory analysis)", CLASSIFIABLE[cohort]),
    ]
    excl_n = s["excluded_esrd_dialysis_transplant"]

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", color=color)

    ys = [8.6, 6.2, 3.8, 1.4]
    xc, w, h = 4.0, 5.2, 1.5
    for (label, n), y in zip(main, ys):
        box = FancyBboxPatch((xc - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.4, edgecolor=color, facecolor="white")
        ax.add_patch(box)
        ax.text(xc, y, f"{label}\nn = {n:,}", ha="center", va="center", fontsize=9)
    # vertical arrows
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((xc, y0 - h / 2), (xc, y1 + h / 2),
                                     arrowstyle="-|>", mutation_scale=14, lw=1.3, color="0.35"))
    # exclusion side box between admissions and baseline
    ymid = (ys[0] + ys[1]) / 2
    ex_w, ex_h = 3.4, 1.0
    exb = FancyBboxPatch((8.0 - ex_w / 2, ymid - ex_h / 2), ex_w, ex_h,
                         boxstyle="round,pad=0.06,rounding_size=0.1",
                         linewidth=1.1, edgecolor="0.5", facecolor="#f4f4f4")
    ax.add_patch(exb)
    ax.text(8.0, ymid, f"Excluded ESRD, dialysis,\nor transplant\nn = {excl_n:,}",
            ha="center", va="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((xc + w / 2, ymid), (8.0 - ex_w / 2, ymid),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.0, color="0.5"))


fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 7))
panel(axL, "mimic", "MIMIC-IV (development)", MIMIC)
panel(axR, "eicu", "eICU-CRD (external validation)", EICU)
fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.03, wspace=0.06)
fig.savefig(FIG / "fig5_study_flow.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig5_study_flow.pdf", bbox_inches="tight")
print("Wrote", FIG / "fig5_study_flow.png")
