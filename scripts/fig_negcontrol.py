#!/usr/bin/env python
"""Render the negative-control falsification figure (forest plot), applied to both
the composite (MAKE-H) and the persistent-renal-dysfunction component (PRD).

Reads outputs/negcontrol_results.csv (columns: cohort, outcome, negative_control,
apparent_effect, ci_low, ci_high). Writes outputs/figures/fig8_negcontrol.png/.pdf.

Usage:
    python scripts/fig_negcontrol.py
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

rows = list(csv.DictReader(open(OUT / "negcontrol_results.csv", newline="")))


def get(cohort, outcome, drug):
    for r in rows:
        if r["cohort"] == cohort and r["outcome"] == outcome and r["negative_control"] == drug:
            return (float(r["apparent_effect"]) * PP, float(r["ci_low"]) * PP, float(r["ci_high"]) * PP)
    return None


# rows top-to-bottom within each panel
spec = [
    ("MIMIC-IV, acetaminophen", "mimic", "acetaminophen", MIMIC, "o"),
    ("eICU-CRD, acetaminophen", "eicu", "acetaminophen", EICU, "o"),
    ("MIMIC-IV, pantoprazole", "mimic", "pantoprazole", MIMIC, "s"),
    ("eICU-CRD, pantoprazole", "eicu", "pantoprazole", EICU, "s"),
]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
panels = [("make_h", "A  Composite (MAKE-H)"), ("prd_h", "B  Persistent renal dysfunction")]
ys = list(range(len(spec)))[::-1]

for ax, (oc, title) in zip(axes, panels):
    for y, (label, coh, drug, col, mk) in zip(ys, spec):
        e, lo, hi = get(coh, oc, drug)
        sig = not (lo < 0 < hi)
        ax.errorbar(e, y, xerr=[[e - lo], [hi - e]], fmt=mk, mfc=(col if sig else "white"),
                    mec=col, ecolor=col, ms=7, capsize=3, lw=1.6, zorder=3)
    ax.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Apparent effect, percentage points (negative = spurious benefit)", fontsize=8.5)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)

axes[0].set_yticks(ys)
axes[0].set_yticklabels([s[0] for s in spec], fontsize=9)
axes[0].set_ylim(-0.6, len(spec) - 0.4)
# legend for filled vs open
h1 = axes[1].plot([], [], "o", mfc="0.25", mec="0.25", ms=7, label="CI excludes zero")[0]
h2 = axes[1].plot([], [], "o", mfc="white", mec="0.25", ms=7, label="CI includes zero")[0]
axes[1].legend(handles=[h1, h2], loc="lower right", fontsize=8, frameon=False)

fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
fig.savefig(FIG / "fig8_negcontrol.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "fig8_negcontrol.pdf", bbox_inches="tight")
print("Wrote", FIG / "fig8_negcontrol.png")
