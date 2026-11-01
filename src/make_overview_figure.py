"""Overview figure for the repository front page: pairwise verdict matrix and the main verification numbers.
Reads results/results3.json and results/results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(BASE, "results"); F = os.path.join(BASE, "figures")
R3 = json.load(open(os.path.join(RD, "results3.json"), encoding="utf-8"))
R = json.load(open(os.path.join(RD, "results.json"), encoding="utf-8"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10,
                     "savefig.dpi": 200, "axes.titleweight": "bold"})
ST = ["AL", "DR", "AC", "OP", "CB"]
NAME = {"AL": "Alert", "DR": "Drowsy", "AC": "Alcohol", "OP": "Opioid", "CB": "Cannabis"}
CAT = {"abs": ("Absolute values", "#3c3c3c"), "base": ("Only with a baseline", "#bdbdbd"), "none": ("Not established", "#ffffff")}


def verdict(pr):
    a = R3["population"]["pairs"][pr]["accall"]; b = R3["operational"]["pairs"][pr]["acc2_det"]
    se = (a * (1 - a) / (2 * 200000)) ** 0.5
    return ("abs" if a >= 0.95 + 2 * se else ("base" if b >= 0.95 else "none")), a, b


def key(i, j):
    k = f"{ST[i]}-{ST[j]}"
    return k if k in R3["population"]["pairs"] else f"{ST[j]}-{ST[i]}"


fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
for i in range(5):
    for j in range(5):
        if i == j:
            ax.add_patch(Rectangle((j, 4 - i), 1, 1, fc="white", ec="#cccccc")); continue
        v, a, b = verdict(key(i, j))
        ax.add_patch(Rectangle((j, 4 - i), 1, 1, fc=CAT[v][1], ec="#777777", lw=0.8))
        txt = f"{a:.3f}" if i > j else f"{b:.3f}"
        ax.text(j + 0.5, 4 - i + 0.5, txt, ha="center", va="center", fontsize=9.5, color="white" if v == "abs" else "#1f1f1f")
ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")
ax.set_xticks(np.arange(5) + 0.5); ax.set_xticklabels([NAME[s] for s in ST])
ax.set_yticks(np.arange(5) + 0.5); ax.set_yticklabels([NAME[s] for s in ST[::-1]])
ax.tick_params(length=0)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title("Can the two states be told apart from the eyes?", fontsize=10.5)
ax.text(0, -0.55, "Lower triangle: all-feature accuracy without a baseline.  Upper: two-feature accuracy with a baseline.", fontsize=8, color="#444444")
for n, (k, (lab, c)) in enumerate(CAT.items()):
    ax.add_patch(Rectangle((n * 1.95, -1.05), 0.25, 0.25, fc=c, ec="#999999", lw=0.5, clip_on=False))
    ax.text(n * 1.95 + 0.33, -0.93, lab, fontsize=8.5, va="center")

m = R["pop_E"]["main"]; o = R["op_E"]["main"]
rows = [("Random forest, no baseline", m["Random Forest"]["f1"][0], m["Random Forest"]["alert_fpr"][0]),
        ("Random forest, with baseline", o["Random Forest"]["f1"][0], o["Random Forest"]["alert_fpr"][0]),
        ("HMID rules, with baseline", o["HMID"]["f1"][0], o["HMID"]["alert_fpr"][0])]
y = np.arange(len(rows))[::-1]
bx.barh(y + 0.18, [r[1] for r in rows], 0.34, color="#3c3c3c", edgecolor="black", lw=0.5, label="Five-state macro-F1")
bx.barh(y - 0.18, [r[2] for r in rows], 0.34, color="#ffffff", edgecolor="black", lw=0.8, label="Alert false-alarm rate")
for yy, r in zip(y, rows):
    bx.text(r[1] + 0.01, yy + 0.18, f"{r[1]:.3f}", va="center", fontsize=9)
    bx.text(r[2] + 0.01, yy - 0.18, f"{100 * r[2]:.1f}%", va="center", fontsize=9)
bx.set_yticks(y); bx.set_yticklabels([r[0] for r in rows]); bx.set_xlim(0, 1.12)
bx.spines["top"].set_visible(False); bx.spines["right"].set_visible(False)
bx.legend(frameon=False, loc="lower right", fontsize=8.5)
bx.set_title("Five-state discrimination on evidence-derived samples", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(F, "f_overview.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)
print("overview figure ok")
