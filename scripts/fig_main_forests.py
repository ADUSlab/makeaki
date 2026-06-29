#!/usr/bin/env python
"""Regenerate the three main forest figures in a consistent, readable style:

  fig2_aki_gradient.png      nephrotoxin-policy benefit by KDIGO AKI stage
  fig4_components_forest.png nephrotoxin-policy effect on each MAKE-H component
  fig7_vpt_forest.png        well-defined VPT vs comparator, with direction labels

All are point-and-interval forests (no bar or line charts). Reads result CSVs in
outputs/. Writes PNG and PDF to outputs/figures/.

Usage:
    python scripts/fig_main_forests.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MIMIC = "#1b4965"
EICU = "#9e2a2b"
PP = 100.0


def rows(path):
    with open(OUT / path, newline="") as f:
        return list(csv.DictReader(f))


def forest(ax, items, xlabel, title):
    """items: list of (y, x, lo, hi, color, marker, filled)."""
    for y, x, lo, hi, col, mk, filled in items:
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt=mk,
                    mfc=(col if filled else "white"), mec=col, ecolor=col,
                    ms=7, capsize=3, lw=1.6, zorder=3)
    ax.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def legend_cohorts(ax):
    h1 = ax.plot([], [], "o", color=MIMIC, label="MIMIC-IV")[0]
    h2 = ax.plot([], [], "o", color=EICU, label="eICU-CRD")[0]
    ax.legend(handles=[h1, h2], loc="best", fontsize=8.5, frameon=False)


# ---- Figure 2: AKI-stage gradient ------------------------------------------
def fig_gradient():
    m = {r["subgroup"]: r for r in rows("mimic_hetero_subgroups.csv")}
    e = {r["subgroup"]: r for r in rows("eicu_hetero_subgroups.csv")}
    stages = ["AKI stage 1", "AKI stage 2", "AKI stage 3"]
    ys = [2, 1, 0]
    items = []
    for s, y in zip(stages, ys):
        rm, re = m[s], e[s]
        items.append((y + 0.16, float(rm["benefit"]) * PP, float(rm["ci_low"]) * PP, float(rm["ci_high"]) * PP, MIMIC, "o", True))
        items.append((y - 0.16, float(re["benefit"]) * PP, float(re["ci_low"]) * PP, float(re["ci_high"]) * PP, EICU, "o", True))
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    forest(ax, items, "Policy benefit, percentage-point reduction in MAKE-H (right = greater benefit)",
           "Benefit by KDIGO AKI stage")
    ax.set_yticks(ys); ax.set_yticklabels(stages, fontsize=10)
    ax.set_ylim(-0.6, 2.6)
    legend_cohorts(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_aki_gradient.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig2_aki_gradient.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 6 (file fig4): components ---------------------------------------
def fig_components():
    data = rows("make_components_secondary.csv")
    labels = [("make_h", "MAKE-H (composite)"), ("death_h", "In-hospital death"),
              ("rrt_h", "Renal-replacement dependence"), ("prd_h", "Persistent renal dysfunction (renal component)")]
    ys = [3, 2, 1, 0]
    items = []
    for (oc, _), y in zip(labels, ys):
        for coh, col, off in (("mimic", MIMIC, 0.16), ("eicu", EICU, -0.16)):
            r = next(x for x in data if x["cohort"] == coh and x["outcome"] == oc)
            e, lo, hi = float(r["benefit_exposed"]) * PP, float(r["ci_low"]) * PP, float(r["ci_high"]) * PP
            filled = not (lo < 0 < hi)
            items.append((y + off, e, lo, hi, col, "o", filled))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    forest(ax, items, "Policy benefit, percentage-point reduction (right = greater benefit)",
           "Effect on the components of MAKE-H")
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in labels], fontsize=9.5)
    ax.set_ylim(-0.6, 3.6)
    legend_cohorts(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_components_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig4_components_forest.pdf", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 3 (file fig7): VPT comparison with direction labels --------------
def fig_vpt():
    data = rows("vpt_results.csv")
    labels = [("make_h", "MAKE-H (composite)"), ("prd_h", "Persistent renal dysfunction (renal component)")]
    ys = [1, 0]
    items = []
    for (oc, _), y in zip(labels, ys):
        for coh, col, off in (("mimic", MIMIC, 0.14), ("eicu", EICU, -0.14)):
            r = next(x for x in data if x["cohort"] == coh and x["outcome"] == oc)
            e, lo, hi = float(r["ate_vpt_minus_comparator"]) * PP, float(r["ci_low"]) * PP, float(r["ci_high"]) * PP
            filled = not (lo < 0 < hi)
            items.append((y + off, e, lo, hi, col, "o", filled))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    forest(ax, items, "Risk difference, percentage points",
           "Vancomycin + piperacillin-tazobactam vs vancomycin + alternative beta-lactam")
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in labels], fontsize=9.5)
    ax.set_ylim(-0.6, 1.7)
    legend_cohorts(ax)
    # direction annotations
    xlo, xhi = ax.get_xlim()
    ax.text(xhi * 0.98, 1.62, "VPT worse →", ha="right", va="center", fontsize=8.5, color="0.3")
    ax.text(xlo * 0.98, 1.62, "← comparator worse", ha="left", va="center", fontsize=8.5, color="0.3")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_vpt_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig7_vpt_forest.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_gradient()
    fig_components()
    fig_vpt()
    print("Wrote fig2_aki_gradient, fig4_components_forest, fig7_vpt_forest")
