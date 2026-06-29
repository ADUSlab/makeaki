#!/usr/bin/env python
"""Render the VPT sensitivity figure (forest plot, no bar/line charts).

Panel A: grace-period robustness. The well-defined VPT-versus-comparator effect
on MAKE-H and persistent renal dysfunction (PRD) estimated under the main 72-hour
window and a 48-hour window, in both cohorts.

Panel B: dose-response. Effect on PRD of >=2 versus 1 day of piperacillin-
tazobactam among VPT-exposed patients, in both cohorts.

Reads outputs/vpt_results.csv (72h main) and outputs/vpt_sensitivity_results.csv.
Writes outputs/figures/fig9_vpt_sensitivity.png and .pdf.

Usage:
    python scripts/fig_vpt_sensitivity.py
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

MIMIC = "#1b4965"   # dark blue
EICU = "#9e2a2b"    # muted red
PP = 100.0          # proportion -> percentage points


def load(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get(rows, **kw):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def trip(r, eff="ate_vpt_minus_comparator"):
    return (float(r[eff]) * PP, float(r["ci_low"]) * PP, float(r["ci_high"]) * PP)


main = load(OUT / "vpt_results.csv")
sens = load(OUT / "vpt_sensitivity_results.csv")

# Panel A rows: (label, cohort, outcome, color). Harm contrasts at top.
rowsA = [
    ("eICU-CRD: persistent renal dysfunction", "eicu", "prd_h", EICU),
    ("eICU-CRD: MAKE-H", "eicu", "make_h", EICU),
    ("MIMIC-IV: persistent renal dysfunction", "mimic", "prd_h", MIMIC),
    ("MIMIC-IV: MAKE-H", "mimic", "make_h", MIMIC),
]

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(7.6, 6.6), gridspec_kw={"height_ratios": [4, 2.2], "hspace": 0.55}
)

# ---- Panel A: grace period 72h vs 48h --------------------------------------
yA = list(range(len(rowsA)))[::-1]
for y, (label, coh, oc, col) in zip(yA, rowsA):
    e72, lo72, hi72 = trip(get(main, cohort=coh, outcome=oc), "ate_vpt_minus_comparator")
    r48 = get(sens, analysis="grace_48h", cohort=coh, outcome=oc)
    e48, lo48, hi48 = trip(r48, "effect")
    # 72h: open marker above; 48h: filled marker below
    axA.errorbar(e72, y + 0.16, xerr=[[e72 - lo72], [hi72 - e72]], fmt="o",
                 mfc="white", mec=col, ecolor=col, ms=7, capsize=3, lw=1.6, zorder=3)
    axA.errorbar(e48, y - 0.16, xerr=[[e48 - lo48], [hi48 - e48]], fmt="o",
                 mfc=col, mec=col, ecolor=col, ms=7, capsize=3, lw=1.6, zorder=3)
axA.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
axA.set_yticks(yA)
axA.set_yticklabels([r[0] for r in rowsA], fontsize=9.5)
axA.set_ylim(-0.6, len(rowsA) - 0.4)
axA.set_xlabel("Risk difference, percentage points (positive favours comparator, i.e. VPT worse)", fontsize=9)
axA.set_title("A  Robustness to the grace period (72h vs 48h)", loc="left", fontsize=11, fontweight="bold")
# legend proxies
h72 = axA.plot([], [], "o", mfc="white", mec="0.25", ms=7, label="72-hour window (main)")[0]
h48 = axA.plot([], [], "o", mfc="0.25", mec="0.25", ms=7, label="48-hour window")[0]
axA.legend(handles=[h72, h48], loc="lower right", fontsize=8.5, frameon=False)
axA.grid(axis="x", color="0.9", lw=0.6)
axA.set_axisbelow(True)

# ---- Panel B: dose-response (>=2 vs 1 day), PRD -----------------------------
rowsB = [("eICU-CRD", "eicu", EICU), ("MIMIC-IV", "mimic", MIMIC)]
yB = list(range(len(rowsB)))[::-1]
for y, (label, coh, col) in zip(yB, rowsB):
    r = get(sens, analysis="dose_ge2_vs_1", cohort=coh, outcome="prd_h")
    e, lo, hi = trip(r, "effect")
    axB.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt="s", mfc=col, mec=col,
                 ecolor=col, ms=7, capsize=3, lw=1.6, zorder=3)
axB.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
axB.set_yticks(yB)
axB.set_yticklabels([r[0] for r in rowsB], fontsize=9.5)
axB.set_ylim(-0.6, len(rowsB) - 0.4)
axB.set_xlabel("Difference in PRD, percentage points (positive = more pip-tazo days worse)", fontsize=9)
axB.set_title("B  Dose-response: ≥2 vs 1 day of piperacillin-tazobactam (PRD)", loc="left", fontsize=11, fontweight="bold")
axB.grid(axis="x", color="0.9", lw=0.6)
axB.set_axisbelow(True)

fig.savefig(FIG / "fig9_vpt_sensitivity.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig9_vpt_sensitivity.pdf", bbox_inches="tight")
print("Wrote", FIG / "fig9_vpt_sensitivity.png")
