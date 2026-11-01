"""Experiment 2 - camera requirements.
Main: population condition (single epoch, no individual baseline; absolute indicators), measurement noise added per epoch.
Upper-bound comparison: operational condition (assumed individual baseline, 10-epoch averaging).
Criterion (main): loss of random-forest macro-F1 and of random-forest opioid recall relative to noise-free measurement
under the same condition and sampling (results/results2.json: grid = population, grid_op = operational, ref = noise-free).
Output: results/results2.json"""
import json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import STATES, hmid, metrics, noise_vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
core.set_scenario("E")

FPS = [30, 45, 60, 90, 120]
RESL = [0.40, 0.30, 0.20, 0.15, 0.10]
R = {"FPS": FPS, "RES": RESL, "grid": {}, "grid_op": {}, "ref": {}}


def run(f, r, cond):
    acc = {"RF": [], "HMID": []}
    for seed in (0, 1):
        rng = np.random.default_rng(100 + seed)
        core.set_scenario("E")
        nz = None if f is None else noise_vec(f, r)
        if cond == "op":
            core.use_effective(10, 10, 0.5, nz)
            X, y = core.sample_eff(800, rng)
        else:
            X, y = core.sample_eff(800, rng)
            if nz is not None:
                X = core.add_noise(X, None, rng, vec=nz, y=y)
        yp = np.zeros_like(y)
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
            yp[te] = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(X[tr], y[tr]).predict(X[te])
        acc["RF"].append(metrics(y, yp)); acc["HMID"].append(metrics(y, hmid(X)))
    return {m: {k: float(np.nanmean([d[k] for d in v])) for k in v[0]} for m, v in acc.items()}


for cond, key in (("pop", "grid"), ("op", "grid_op")):
    R["ref"][cond] = run(None, None, cond)
    for f in FPS:
        for r in RESL:
            R[key][f"{f}|{r}"] = run(f, r, cond)
        print(cond, "fps", f, flush=True)


def pupil_res(H, fov, d=0.65, kappa=0.5):
    p = 2 * d * np.tan(np.radians(fov) / 2) / H * 1000
    return p, kappa * p


def H_required(r_req, fov, d=0.65, kappa=0.5):
    return kappa * 2 * d * np.tan(np.radians(fov) / 2) * 1000 / r_req


sensors = {"OV9284 (1280 × 800, GS)": 1280, "VB56G4A (1124 × 1364, GS)": 1124, "VB1940 (2560 × 1984, RGB-IR)": 2560, "OX05B1S (2592 × 1944, RGB-IR GS)": 2592}
R["geo"] = {f"{n}|{fov}|{d}": dict(zip(("p", "r"), pupil_res(H, fov, d))) for n, H in sensors.items() for fov in (50, 120) for d in (0.5, 0.65, 0.8)}
R["req"] = {f"{rr}|{fov}|{d}": H_required(rr, fov, d) for rr in (0.30, 0.20, 0.15, 0.10) for fov in (40, 50, 60, 120) for d in (0.5, 0.65, 0.8)}
json.dump(R, open(os.path.join(core.RESULTS_DIR, "results2.json"), "w", encoding="utf-8"), indent=1)
print("saved")
