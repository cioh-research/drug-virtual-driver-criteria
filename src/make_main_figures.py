"""Main-text data figures (evidence matrix, separability, operational condition, camera requirements),
drawn at their printed width (6.3 in) so that every label is at least 8 pt."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings; warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "figures"); RD = os.path.join(BASE, "results")
P = json.load(open(os.path.join(RD, "params.json"), encoding="utf-8")); R = json.load(open(os.path.join(RD, "results.json"), encoding="utf-8"))
R2 = json.load(open(os.path.join(RD, "results2.json"), encoding="utf-8"))
FS = 9.0          # base font size (pt) at printed size
MIN = 8.0         # smallest font used anywhere
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": FS,
                     "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": MIN + 0.5, "ytick.labelsize": MIN + 0.5,
                     "legend.fontsize": MIN + 0.5, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.titleweight": "bold", "mathtext.fontset": "stix"})
STATES = ["AL", "DR", "AC", "OP", "CB"]
LAB = {"AL": "Alert", "DR": "Drowsy", "AC": "Alcohol", "OP": "Opioid", "CB": "Cannabis"}
COL = {"AL": "#0072B2", "DR": "#E69F00", "AC": "#009E73", "OP": "#D55E00", "CB": "#CC79A7"}
INK = "#1f1f1f"
FEATS = P["features"]
FN = {"P": "Eye closure (%)", "BD": "Blink duration", "SL": "Saccade latency", "PV": "Saccade peak velocity",
      "G": "Pursuit gain", "FD": "Fixation duration", "D": "Pupil diameter", "A": "PLR amplitude", "V": "PLR velocity"}
SYM = {"P": "$P$", "BD": "$B_d$", "SL": "$L_s$", "PV": "$V_p$", "G": "$G_p$", "FD": "$F_d$", "D": "$D_p$", "A": "$A_c$", "V": "$V_c$"}
seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fb", "#9ecae1", "#2b6ca3", "#08306b"])
W = 6.3


def save(fig, n):
    fig.savefig(os.path.join(F, n), facecolor="white"); plt.close(fig)


def clean(ax):
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)


# ---------------- Fig: evidence matrix
fig, ax = plt.subplots(figsize=(W, 4.6))
M = np.zeros((len(FEATS), 4)); T = [["" for _ in range(4)] for _ in FEATS]
for j, s in enumerate(STATES[1:]):
    for i, f in enumerate(FEATS):
        c = P["states"][s][f]
        if s == "DR" and f == "P":
            M[i, j] = 1.0; T[i][j] = "×3.2 (k=1)"
        elif c["k"] > 0:
            v = 100 * c["delta"]; M[i, j] = np.clip(v / 50, -1, 1); T[i][j] = f"{v:+.0f}% (k={c['k']})".replace("-", "−")
        elif c["dir_only"]:
            T[i][j] = "mixed (n.s.)" if c["dir_sign"] == 0 else ("↑ dir. only" if c["dir_sign"] > 0 else "↓ dir. only")
        else:
            T[i][j] = "no data"
cmap = LinearSegmentedColormap.from_list("div", ["#2166ac", "#67a9cf", "#d1e5f0", "#f2f2f2", "#fbe3d6", "#ef8a62", "#b2182b"])
ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
for i in range(len(FEATS)):
    for j in range(4):
        ax.text(j, i, T[i][j], ha="center", va="center", fontsize=MIN + 0.5, color="white" if abs(M[i, j]) > 0.7 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels([LAB[s] for s in STATES[1:]], fontsize=FS)
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels([f"{FN[f]} {SYM[f]}" for f in FEATS], fontsize=FS)
ax.xaxis.tick_top()
ax.set_xticks(np.arange(-.5, 4), minor=True); ax.set_yticks(np.arange(-.5, len(FEATS)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0); clean(ax)
fig.subplots_adjust(left=0.30, right=0.99, top=0.93, bottom=0.02)
save(fig, "f_evidence_matrix.png")

# ---------------- Fig: separability (a, b on top; c full width below)
fig = plt.figure(figsize=(W, 6.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.45, wspace=0.35, left=0.08, right=0.99, top=0.95, bottom=0.10)
for k, (key, ttl) in enumerate((("pop_E", "(a) $D_B$, population"), ("op_E", "(b) $D_B$, operational"))):
    ax = fig.add_subplot(gs[0, k]); DB = np.array(R[key]["DB"]); ax.imshow(np.log10(DB + 0.05), cmap=seq, vmin=-1, vmax=2)
    for a in range(5):
        for b in range(5):
            ax.text(b, a, "–" if a == b else (f"{DB[a, b]:.1f}" if DB[a, b] < 100 else f"{DB[a, b]:.0f}"), ha="center", va="center",
                    fontsize=MIN + 0.5, color="white" if DB[a, b] > 10 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(STATES); ax.set_yticklabels(STATES); ax.set_title(ttl); clean(ax)
ax = fig.add_subplot(gs[1, :]); fis = R["pop_E"]["fisher"]; prs = list(fis.keys()); J = np.array([fis[p] for p in prs]).T
ax.imshow(np.log10(J + 1e-3), cmap=seq, vmin=-2, vmax=1.5, aspect="auto")
for a in range(J.shape[0]):
    for b in range(J.shape[1]):
        ax.text(b, a, f"{J[a, b]:.2f}" if J[a, b] >= 0.01 else "0", ha="center", va="center", fontsize=MIN, color="white" if J[a, b] > 3 else INK)
ax.set_xticks(range(len(prs))); ax.set_xticklabels([p.replace("-", "–") for p in prs], rotation=0)
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels([SYM[f] for f in FEATS]); ax.set_title("(c) Fisher ratio $J_k$, population"); clean(ax)
save(fig, "f_separability.png")

# ---------------- Fig: n-epoch curve + ablation
fig = plt.figure(figsize=(W, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.32, left=0.09, right=0.99, top=0.90, bottom=0.30)
ax = fig.add_subplot(gs[0, 0]); ns = [int(k) for k in R["n_curve"]]
for m, mk, c, lab in (("HMID", "s", "#2b6ca3", "HMID"), ("Random Forest", "^", "#08306b", "RF")):
    ax.plot(ns, [R["n_curve"][str(n)][m]["f1"][0] for n in ns], marker=mk, ms=4, color=c, lw=1.6, label=lab + " F1")
    ax.plot(ns, [R["n_curve"][str(n)][m]["alert_fpr"][0] for n in ns], marker=mk, ms=4, color=c, lw=1.1, ls="--", label=lab + " FAR")
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels(ns); ax.set_xlabel("Averaged 1-min epochs"); ax.set_ylim(0, 1.02)
ax.set_title("(a) Baseline + averaging"); ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25), handlelength=1.5, columnspacing=0.8)
ax = fig.add_subplot(gs[0, 1]); ab = R["ablation"]; keys = list(ab.keys()); short = ["All", "E", "E+O", "E+U", "O+U"]; w = 0.15
for j, s in enumerate(STATES):
    ax.bar(np.arange(5) + (j - 2) * w, [ab[k][f"rec_{s}"][0] for k in keys], w, color=COL[s], label=s, edgecolor="white", lw=0.4)
ax.set_xticks(range(5)); ax.set_xticklabels(short); ax.set_ylabel("Per-class recall"); ax.set_ylim(0, 1.05)
ax.set_title("(b) Channel ablation (RF)"); ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), handlelength=1, columnspacing=0.8)
save(fig, "f_operational.png")

# ---------------- Fig: camera requirements (a, b on top; c below)
FPS, RES = R2["FPS"], R2["RES"]; G = R2["grid"]; ref = R2["ref"]["pop"]["RF"]
fl = ref["f1"] - np.array([[G[f"{f}|{r}"]["RF"]["f1"] for r in RES] for f in FPS])
ol = ref["rec_OP"] - np.array([[G[f"{f}|{r}"]["RF"]["rec_OP"] for r in RES] for f in FPS])
ok = (fl <= 0.03) & (ol <= 0.03)
fig = plt.figure(figsize=(W, 6.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.45, wspace=0.30, left=0.10, right=0.98, top=0.95, bottom=0.08)
for k, (M, ttl) in enumerate(((fl, "(a) Macro-F1 loss"), (ol, "(b) Opioid-recall loss"))):
    ax = fig.add_subplot(gs[0, k]); ax.imshow(M, cmap=seq.reversed(), aspect="auto", origin="lower", vmin=0, vmax=0.08)
    for a in range(len(FPS)):
        for b in range(len(RES)):
            v = M[a, b]
            ax.text(b, a, f"{v:.3f}", ha="center", va="center", fontsize=MIN, fontweight="bold" if v <= 0.03 else "normal", color="white" if v < 0.03 else INK)
            if ok[a, b]:
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec="#D55E00", lw=1.5))
    ax.set_xticks(range(len(RES))); ax.set_xticklabels([f"{r:.2f}" for r in RES]); ax.set_yticks(range(len(FPS))); ax.set_yticklabels(FPS)
    ax.set_xlabel("Pupil resolution (mm)"); ax.set_ylabel("Frame rate (fps)"); ax.set_title(ttl); clean(ax)
ax = fig.add_subplot(gs[1, :]); fov = np.linspace(30, 130, 200)
for rr, ls in ((0.30, ":"), (0.20, "-"), (0.15, "--"), (0.10, "-.")):
    ax.plot(fov, 0.5 * 2 * 0.65 * np.tan(np.radians(fov) / 2) * 1000 / rr, ls, color="#08306b", lw=1.5, label=f"r = {rr:.2f} mm")
for name, Hs, c in (("1.0 MP (1280 px)", 1280, "#6b6b6b"), ("1.5 MP (1124 px)", 1124, "#9a9a9a"), ("5 MP (2560–2592 px)", 2592, "#D55E00")):
    ax.axhline(Hs, color=c, lw=1.0, ls=(0, (4, 2))); ax.text(127, Hs * (1.07 if Hs == 1280 else 0.88), name, fontsize=MIN + 0.5, ha="right", va="center", color=c)
ax.set_yscale("log"); ax.set_ylim(500, 15000); ax.set_xlim(30, 130); ax.set_xlabel("Horizontal field of view (deg)"); ax.set_ylabel("Required horizontal pixels")
ax.set_title("(c) Pixel budget (d = 0.65 m, κ = 0.5)"); ax.legend(frameon=False, loc="upper left", ncol=2)
save(fig, "f_requirement.png")
print("main figures ok")
