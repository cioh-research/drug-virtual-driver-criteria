"""Experiment 1 - statistical verification of the evidence-derived criteria.
Samples are drawn from the evidence-derived distributions (results/params.json); they are NOT rendered virtual drivers.
A. Population condition: single 1-min epoch, population thresholds (between-subject SDs).
B. Operational condition: individual alert baseline (n_cal epochs) + averaging of n_ep epochs (hierarchical variance model).
Scenario E = evidence-quantified effects only; scenario D = direction-only cells set to 0.5 between-subject SD (sensitivity).
Output: results/results.json"""
import json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import STATES, FEATS, models, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
warnings.filterwarnings("ignore")
IDX = {f: i for i, f in enumerate(FEATS)}


def setup(scenario="E", operational=True, n_ep=10, n_cal=10, rho_b=0.5, tier="Recommended"):
    core.set_scenario(scenario)
    if operational:
        core.use_effective(n_ep, n_cal, rho_b, core.noise_sd(tier))
    noise = None if operational else core.noise_sd(tier)
    return noise


def cv_eval(make, n=1000, seeds=(0, 1, 2), cols=None, return_cm=False, noise=None):
    res, cms = [], np.zeros((5, 5))
    for sd in seeds:
        rng = np.random.default_rng(sd)
        X, y = core.sample_eff(n, rng)
        if noise is not None:
            X = core.add_noise(X, None, rng, vec=noise, y=y)
        Xf = X if cols is None else X[:, cols]
        yp = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(Xf, y):
            if make is core.HMID:
                yp[te] = core.hmid(X[te])
            else:
                yp[te] = make().fit(Xf[tr], y[tr]).predict(Xf[te])
        res.append(metrics(y, yp)); cms += confusion_matrix(y, yp, labels=range(5))
    agg = {k: (float(np.nanmean([r[k] for r in res])), float(np.nanstd([r[k] for r in res]))) for k in res[0]}
    return (agg, (cms / cms.sum(1, keepdims=True)).tolist()) if return_cm else agg


def separability():
    DB = np.zeros((5, 5)); pairs = []; fis = {}
    for i, a in enumerate(STATES):
        for j, b in enumerate(STATES):
            v1, v2 = core.SD[a] ** 2, core.SD[b] ** 2; v = (v1 + v2) / 2; d = core.MU[a] - core.MU[b]
            if i != j:
                DB[i, j] = float((d ** 2 / v).sum() / 8 + 0.5 * np.log(v / np.sqrt(v1 * v2)).sum())
            if j > i:
                J = d ** 2 / (v1 + v2); o = np.argsort(-J); fis[f"{a}-{b}"] = J.tolist()
                pairs.append(dict(pair=f"{a}-{b}", DB=DB[i, j], bayes_ub=0.5 * np.exp(-DB[i, j]), top1=FEATS[o[0]], J1=float(J[o[0]]), top2=FEATS[o[1]], J2=float(J[o[1]])))
    return DB.tolist(), pairs, fis


R = {}
# ---------- A. population, single epoch
for sc in ("E", "D"):
    nz = setup(sc, operational=False)
    S = {"theta": core.THETA, "theta_mid": dict(core.THETA_MID)}
    S["DB"], S["pairs"], S["fisher"] = separability()
    S["main"] = {m: cv_eval(mk, noise=nz) for m, mk in models().items()}
    R[f"pop_{sc}"] = S
    print("pop", sc, {m: round(v["f1"][0], 3) for m, v in S["main"].items()})

# ---------- B. operational condition
for sc in ("E", "D"):
    setup(sc, operational=True)
    S = {"theta": core.THETA, "theta_mid": dict(core.THETA_MID)}
    S["DB"], S["pairs"], S["fisher"] = separability()
    S["main"], S["cm"] = {}, {}
    for m, mk in models().items():
        S["main"][m], S["cm"][m] = cv_eval(mk, return_cm=True)
    R[f"op_{sc}"] = S
    print("op", sc, {m: round(v["f1"][0], 3) for m, v in S["main"].items()})

# ---------- C. oral-dose opioid sensitivity (scenario W)
for cond in ("pop", "op"):
    nz = setup("W", operational=(cond == "op"))
    S = {"theta": core.THETA, "theta_mid": dict(core.THETA_MID)}
    S["DB"], S["pairs"], S["fisher"] = separability()
    S["main"] = {m: cv_eval(models()[m], noise=nz) for m in ("HMID", "Random Forest")}
    R[f"{cond}_W"] = S
    print(cond, "W", {m: round(v["f1"][0], 3) for m, v in S["main"].items()})

R["n_curve"] = {}
for n in (1, 3, 5, 10, 20, 30):
    setup("E", True, n_ep=n)
    R["n_curve"][str(n)] = {m: cv_eval(models()[m]) for m in ("HMID", "Random Forest")}
R["c_resp"] = {}
for cr in (0.0, 0.25, 0.5):
    core.set_scenario("E", c_resp=cr); core.use_effective(10, 10, 0.5, core.noise_sd("Recommended"))
    R["c_resp"][str(cr)] = {m: cv_eval(models()[m]) for m in ("HMID", "Random Forest")}
core.set_scenario("E", c_resp=float(os.environ.get("C_RESP", "0.25")))
R["rho_b"] = {}
for rb in (0.3, 0.5, 0.7):
    setup("E", True, rho_b=rb)
    R["rho_b"][str(rb)] = {m: cv_eval(models()[m]) for m in ("HMID", "Random Forest")}
print("curves done")

setup("E", True)
groups = {"All channels (E+O+U)": FEATS, "Eyelid only (E)": ["P", "BD"], "Eyelid + Oculomotor (E+O)": ["P", "BD", "SL", "PV", "G", "FD"],
          "Eyelid + Pupillary (E+U)": ["P", "BD", "D", "A", "V"], "Oculomotor + Pupillary (O+U)": ["SL", "PV", "G", "FD", "D", "A", "V"]}
R["ablation"] = {g: cv_eval(models()["Random Forest"], cols=[IDX[f] for f in fs]) for g, fs in groups.items()}
R["single"] = {f: cv_eval(models()["Gaussian NB"], cols=[IDX[f]])["f1"] for f in FEATS}
R["tiers"] = {}
for t in core.TIERS:
    setup("E", True, tier=t)
    R["tiers"][t] = {m: cv_eval(models()[m]) for m in ("HMID", "Random Forest")}
R["noise_sd"] = {t: {s: v.tolist() for s, v in core.noise_sd(t).items()} for t in core.TIERS}
print("ablation/tiers done")

# ---------- failure-mode probes (operational condition, n_ep = 10)
setup("E", True)
rng = np.random.default_rng(11)
X, y = core.sample_eff(1000, rng)
rf = models()["Random Forest"]().fit(X, y)
MU, SD = core.MU, core.SD
Dl = {s: MU[s] - MU["AL"] for s in STATES}
probes = {"Alcohol + Drowsiness (AC+DR)": (MU["AL"] + Dl["AC"] + Dl["DR"], np.maximum(SD["AC"], SD["DR"])),
          "Cannabis + Alcohol (CB+AC)": (MU["AL"] + Dl["CB"] + Dl["AC"], np.maximum(SD["CB"], SD["AC"])),
          "Opioid + Drowsiness (OP+DR)": (MU["AL"] + Dl["OP"] + Dl["DR"], np.maximum(SD["OP"], SD["DR"]))}
lum = MU["AL"].copy(); lum[IDX["D"]] = MU["OP"][IDX["D"]]  # bright-light miosis to opioid-like size, PLR velocity intact
probes["Alert, high-luminance miosis (PLR intact)"] = (lum, SD["AL"])
lum2 = lum.copy(); lum2[IDX["V"]] = MU["AL"][IDX["V"]] * MU["OP"][IDX["D"]] / MU["AL"][IDX["D"]]   # constriction velocity scaled with diameter
probes["Alert, high-luminance miosis (Vc scaled with diameter)"] = (lum2, SD["AL"])
R["probes"] = {}
for name, (mu, sd) in probes.items():
    Xp = np.clip(mu + rng.standard_normal((2000, len(FEATS))) * sd, core.LO, core.HI)
    R["probes"][name] = {"HMID": np.bincount(core.hmid(Xp), minlength=5).tolist(), "RF": np.bincount(rf.predict(Xp), minlength=5).tolist()}
json.dump(R, open(os.path.join(core.RESULTS_DIR, "results.json"), "w", encoding="utf-8"), indent=1, default=float)
print("saved")
