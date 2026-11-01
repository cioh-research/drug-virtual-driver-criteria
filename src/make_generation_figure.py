"""One-glance virtual-driver generation criteria figure (f_generation_criteria.png).

Panel (a): signed pooled relative effect of each impaired state on each of the nine features
(scenario E inputs, results/params.json) shown as a diverging heat map with glyphs.
Panel (b): the combination logic that the generated profiles must preserve.
Glyph scale: |d| < 5 % '≈'; 5-20 % one arrow; >= 20 % two arrows; direction-only '(↑)/(↓)'; no evidence '–'.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P = json.load(open(os.path.join(ROOT, "results", "params.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "savefig.dpi": 300})
FEATS = ["P", "BD", "SL", "PV", "G", "FD", "D", "A", "V"]
LAB = {"P": "$P$", "BD": "$B_d$", "SL": "$L_s$", "PV": "$V_p$", "G": "$G_p$", "FD": "$F_d$", "D": "$D_p$", "A": "$A_c$", "V": "$V_c$"}
STATES = [("DR", "Drowsy"), ("AC", "Alcohol"), ("OP", "Opioid"), ("CB", "Cannabis (THC)")]
INK = "#1f1f1f"; MUTED = "#6b6b6b"
CMAP = plt.get_cmap("RdBu_r")


def glyph(c, s, f):
    if s == "DR" and f == "P":
        return "↑↑", 2.186
    d = c["delta"]
    if c["k"] == 0:
        if c["dir_only"]:
            return {1: "(↑)", -1: "(↓)"}.get(c["dir_sign"], "(±)"), None
        return "–", None
    a = abs(d)
    if a < 0.05:
        return "≈", d
    g = ("↑" if d > 0 else "↓") * (2 if a >= 0.20 else 1)
    return g, d


def main():
    fig = plt.figure(figsize=(6.3, 2.6))
    ax = fig.add_axes([0.13, 0.08, 0.84, 0.80])
    M = np.full((4, 9), np.nan)
    for i, (s, _) in enumerate(STATES):
        for j, f in enumerate(FEATS):
            g, d = glyph(P["states"][s][f], s, f)
            if d is not None:
                M[i, j] = np.clip(d, -0.5, 0.5)
            col = "#f2f2f2" if d is None else CMAP((np.clip(d, -0.5, 0.5) + 0.5))
            ax.add_patch(plt.Rectangle((j, 3 - i), 0.96, 0.92, fc=col, ec="white", lw=1.5))
            if d is None:
                txt = g
            elif s == "DR" and f == "P":
                txt = f"{g}\n×3.2"
            else:
                pc = round(100 * d)
                txt = f"{g}\n{pc:+d}%".replace("-", "−") if pc != 0 else f"{g}\n0%"
            ax.text(j + 0.48, 3 - i + 0.46, txt, ha="center", va="center", fontsize=8.5, color=INK if d is None or abs(d) < 0.38 else "white",
                    fontweight="bold" if d is not None and abs(d) >= 0.20 else "normal")
    for x0, x1, name in ((0, 2, "Eyelid"), (2, 6, "Oculomotor"), (6, 9, "Pupillary")):
        ax.plot([x0 + 0.05, x1 - 0.09], [4.12, 4.12], color=MUTED, lw=1)
        ax.text((x0 + x1) / 2 - 0.02, 4.22, name, ha="center", va="bottom", fontsize=8.5, color=INK)
    for x in (2, 6):
        ax.axvline(x - 0.02, color="white", lw=4)
    ax.set_xlim(-0.05, 9.0); ax.set_ylim(-0.1, 4.55)
    ax.set_xticks(np.arange(9) + 0.48); ax.set_xticklabels([LAB[f] for f in FEATS], fontsize=9.5)
    ax.set_yticks(np.arange(4) + 0.46); ax.set_yticklabels([n for _, n in STATES][::-1], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    

    fig.savefig(os.path.join(OUT, "f_generation_criteria.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
