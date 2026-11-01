"""Supplementary figures (state parameters, signal-level dynamics, severity templates, confusion matrices).
Main-text figures are drawn by make_main_figures.py, make_generation_figure.py and make_block_diagrams.py."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings; warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "figures"); os.makedirs(F, exist_ok=True)
RD = os.path.join(BASE, "results")
P = json.load(open(os.path.join(RD, "params.json"), encoding="utf-8")); R = json.load(open(os.path.join(RD, "results.json"), encoding="utf-8"))
R2 = json.load(open(os.path.join(RD, "results2.json"), encoding="utf-8"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "axes.titleweight": "bold",
                     "axes.titlesize": 10.5, "mathtext.fontset": "stix"})
STATES = ["AL", "DR", "AC", "OP", "CB"]
LAB = {"AL": "Alert", "DR": "Drowsy", "AC": "Alcohol", "OP": "Opioid", "CB": "Cannabis (THC)"}
COL = {"AL": "#0072B2", "DR": "#E69F00", "AC": "#009E73", "OP": "#D55E00", "CB": "#CC79A7"}
INK, MUTED = "#1f1f1f", "#6b6b6b"
FEATS = P["features"]
FN = {"P": "Eye closure (%)", "BD": "Blink duration (ms)", "SL": "Saccade latency (ms)", "PV": "Saccade peak velocity (deg/s)",
      "G": "Smooth-pursuit gain", "FD": "Fixation duration (ms)", "D": "Pupil diameter (mm)", "A": "PLR amplitude (%)", "V": "PLR velocity (mm/s)"}
seq = LinearSegmentedColormap.from_list("seq", ["#f4f8fb", "#9ecae1", "#2b6ca3", "#08306b"])


def save(fig, n):
    fig.savefig(os.path.join(F, n), bbox_inches="tight", facecolor="white"); plt.close(fig)



# ---------------- Fig: archetypes (mean ± SD, population) with HMID thresholds (operational, relative)
MU = {s: np.array(v) for s, v in P["MU_E"].items()}; SD = {s: np.array(v) for s, v in P["SD_E"].items()}
fig, axs = plt.subplots(3, 3, figsize=(7.6, 6.4))
for k, ax in enumerate(axs.flat):
    for j, s in enumerate(STATES):
        ax.bar(j, MU[s][k], width=0.72, color=COL[s], edgecolor="white")
        ax.errorbar(j, MU[s][k], yerr=SD[s][k], color=INK, capsize=2.5, lw=0.9)
    ax.set_xticks(range(5)); ax.set_xticklabels(STATES, fontsize=8); ax.set_title(FN[FEATS[k]], fontsize=8.5); ax.tick_params(labelsize=8)
fig.tight_layout(); save(fig, "f_archetypes.png")

# ---------------- Fig: signal-level dynamics (PLR and saccade) calibrated to the archetype parameters
# PLR: constriction from D_pre to D_min = D_pre (1 - A_c/100) along a smoothstep whose mean slope over the constriction
# phase equals V_c (the slope operator of Eq. feat3), a 0.15-s plateau at D_min, then exponential re-dilation.
# Saccade: 15-deg saccade; the velocity profile is scaled so that its peak equals V_p and its time integral equals 15 deg.
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
t = np.linspace(0, 3.0, 3000); t_on = 0.5; lat = 0.25; i = {f: n for n, f in enumerate(FEATS)}
for s in STATES:
    d0 = MU[s][i["D"]]; amp = d0 * MU[s][i["A"]] / 100; tc = amp / MU[s][i["V"]]
    tt = t - t_on - lat; u = np.clip(tt / tc, 0, 1)
    y = d0 - amp * (3 * u ** 2 - 2 * u ** 3)
    rec = np.where(tt > tc + 0.15, amp * (1 - np.exp(-(tt - tc - 0.15) / 1.0)), 0)
    axs[0].plot(t, y + rec, color=COL[s], lw=2, label=LAB[s], ls="--" if s == "OP" else "-")
axs[0].axvline(t_on, color=MUTED, lw=0.8, ls=":"); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Pupil diameter (mm)"); axs[0].set_title("(a) Pupillary light reflex")
tm = np.linspace(0, 400, 4000); dt = (tm[1] - tm[0]) / 1000; AMP = {}
xs = np.linspace(0, 1, 4001); g0 = xs ** 1.5 * (1 - xs) ** 2.2; gmax = g0.max(); gs = g0 / gmax; gint = np.trapezoid(gs, xs)
for s in STATES:
    sl, vp = MU[s][i["SL"]], MU[s][i["PV"]]; T = 15.0 / (vp * gint)   # duration (s) giving a 15-deg integral
    x = (tm - sl) / (1000 * T)
    v = np.where((x > 0) & (x < 1), np.clip(x, 0, 1) ** 1.5 * (1 - np.clip(x, 0, 1)) ** 2.2, 0) / gmax * vp
    AMP[s] = (np.trapezoid(v, tm) / 1000, v.max())
    axs[1].plot(tm, v, color=COL[s], lw=2, ls="--" if s == "OP" else "-")
axs[1].set_xlabel("Time from target onset (ms)"); axs[1].set_ylabel("Eye angular velocity (deg/s)"); axs[1].set_title("(b) Saccadic velocity profile (15°)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "f_dynamics.png")
print("saccade integral (deg), peak (deg/s):", {k: (round(a, 2), round(p, 1)) for k, (a, p) in AMP.items()})
# ---------------- Fig: trajectories (assumed templates)
tt = np.linspace(0, 180, 600)
def bateman(t, ka, ke):
    h = t / 60; y = np.exp(-ke * h) - np.exp(-ka * h); return y / y.max()
traj = {"DR": 1 / (1 + np.exp(-(tt - 60) / 12)), "AC": bateman(tt, 3.0, 0.3), "OP": 1 - np.exp(-tt / 10), "CB": bateman(tt, 2.0, 0.8)}
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for s, y in traj.items():
    ax.plot(tt, y, color=COL[s], lw=2, label=LAB[s], ls="--" if s == "OP" else "-")
ax.set_xlabel("Time since onset of exposure / drive (min)"); ax.set_ylabel(r"Severity template $\lambda_s(\tau)$"); ax.set_ylim(0, 1.08)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.2))
save(fig, "f_trajectory.png")

# ---------------- Fig: confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9))
for ax, (key, m, ttl) in zip(axs, (("op_E", "HMID", "(a) HMID, operational"), ("op_E", "Random Forest", "(b) RF, operational"), ("op_D", "Random Forest", "(c) RF, operational, scenario D"))):
    C = np.array(R[key]["cm"][m])
    ax.imshow(C, cmap=seq, vmin=0, vmax=1)
    for a in range(5):
        for b in range(5):
            ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if C[a, b] > 0.55 else INK)
    ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(STATES, fontsize=8); ax.set_yticklabels(STATES, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8.5); ax.set_title(ttl, fontsize=9.5)
    for s_ in ax.spines.values(): s_.set_visible(False)
axs[0].set_ylabel("True", fontsize=8.5)
fig.tight_layout(); save(fig, "f_confusion.png")

print("supplementary figures ok")
