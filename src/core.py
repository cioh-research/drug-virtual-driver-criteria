"""Core model built on the evidence synthesis (results/params.json produced by synth.py).
Features: P (eye closure %), BD (blink duration), SL (saccade latency), PV (saccade peak velocity),
G (smooth-pursuit gain), FD (fixation duration), D (pupil diameter), A (PLR relative amplitude), V (PLR constriction velocity).
Channels: eyelid E = {P, BD}; oculomotor O = {SL, PV, G, FD}; pupillary U = {D, A, V}.
"""
import json, os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results"); FIGD = os.path.join(BASE, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGD, exist_ok=True)
PRM = json.load(open(os.path.join(RESULTS_DIR, os.environ.get("PARAMS_NAME", "params.json")), encoding="utf-8"))

STATES = ["AL", "DR", "AC", "OP", "CB"]
FEATS = PRM["features"]
IDX = {f: i for i, f in enumerate(FEATS)}
CHANNEL = {"P": "E", "BD": "E", "SL": "O", "PV": "O", "G": "O", "FD": "O", "D": "U", "A": "U", "V": "U"}
LO = np.array([0, 50, 100, 100, 0.1, 100, 1.0, 0, 0.05])
HI = np.array([100, 1000, 500, 700, 1.2, 2000, 9.0, 80, 10])


C_RESP = float(os.environ.get("C_RESP", "0.25"))   # between-person heterogeneity of the response (CV of the effect)


RHO_B = 0.5   # share of between-subject variance that is a persistent personal offset (generator, Eq. gen)


def pop_var(s, rho_b=RHO_B):
    """population (between-subject) variance implied by the generator (Eq. gen, lambda = 1):
    x = [mu_AL + (1 + eta) * Delta] * (1 + beta) + u,  Var(beta) = b = rho_b * (SD0_AL / mu_AL)^2,
    Var(eta) = c_resp^2,  Var(u) = (1 - rho_b) * SD0_s^2,  Delta = mu_s - mu_AL
    =>  Var(x) = b * mu_s^2 + (1 - rho_b) * SD0_s^2 + (1 + b) * c_resp^2 * Delta^2."""
    b = rho_b * (SD0["AL"] / MU["AL"]) ** 2; dl = MU[s] - MU["AL"]
    return b * MU[s] ** 2 + (1 - rho_b) * SD0[s] ** 2 + (1 + b) * (C_RESP * dl) ** 2


def set_scenario(sc="E", c_resp=None):
    """population condition: single epoch of an unknown driver; SD from pop_var() (same stochastic model as Eq. gen)."""
    global MU, SD, SD0, THETA, C_RESP
    if c_resp is not None:
        C_RESP = c_resp
    MU = {s: np.array(v) for s, v in PRM[f"MU_{sc}"].items()}
    SD0 = {s: np.array(v) for s, v in PRM[f"SD_{sc}"].items()}
    SD = {s: np.sqrt(pop_var(s)) for s in MU}
    THETA = make_theta()


def corr_matrix(rho):
    R = np.eye(len(FEATS))
    for i, a in enumerate(FEATS):
        for j, b in enumerate(FEATS):
            if i != j and CHANNEL[a] == CHANNEL[b]:
                R[i, j] = rho
    return R


def sample_state(state, n, rng, rho=0.3, mu=None, sd=None):
    mu = MU[state] if mu is None else mu; sd = SD[state] if sd is None else sd
    L = np.linalg.cholesky(corr_matrix(rho))
    Z = rng.standard_normal((n, len(FEATS))) @ L.T
    return np.clip(mu + Z * sd, LO, HI)


# ---------------- camera measurement-noise model -------------------------------------------
TIERS = {"Reference": None, "Minimum": dict(f=30, r=0.3), "Recommended": dict(f=60, r=0.2), "Optimal": dict(f=120, r=0.1)}
W_PLR = 0.2


def noise_vec(f, r):
    """per-epoch measurement SD of each feature for frame rate f (fps) and pupil resolution r (mm), by state.
    Relative PLR amplitude A = 100 (1 - Dmin/Dpre): first-order propagation with independent, equal diameter errors,
    sigma_A = 100 sigma_D sqrt(1 + (Dmin/Dpre)^2) / Dpre, evaluated at each state's pupil diameter and amplitude."""
    q = 1000.0 / f * np.sqrt(2 / 12)                   # two quantised event times (ms)
    sD = r / 2.0                                        # pupil-diameter error (mm)
    sV = sD * np.sqrt(12.0 / (f * W_PLR ** 3))          # OLS slope s.e. over window W (mm/s), same operator as V_c
    sPV = PRM["alert"]["PV"]["mean"] * 2.0 / f          # peak-velocity under-sampling (assumed 2/f relative)
    out = {}
    for s in STATES:
        dpre = MU[s][IDX["D"]]; ratio = 1 - MU[s][IDX["A"]] / 100
        sA = 100 * sD * np.sqrt(1 + ratio ** 2) / dpre
        out[s] = np.array([0.5, q, q, sPV, 1.2 / f, q, sD, sA, sV])
    return out


def noise_sd(tier):
    t = TIERS[tier]
    return {s: np.zeros(len(FEATS)) for s in STATES} if t is None else noise_vec(t["f"], t["r"])


def add_noise(X, tier, rng, vec=None, y=None):
    s = noise_sd(tier) if vec is None else vec
    S = np.vstack([s[STATES[k]] for k in y]) if isinstance(s, dict) else s
    return np.clip(X + rng.standard_normal(X.shape) * S, LO, HI)


def make_dataset(n, rng, rho=0.3, tier="Recommended", vec=None):
    X = np.vstack([sample_state(s, n, rng, rho) for s in STATES]); y = np.repeat(np.arange(5), n)
    return add_noise(X, tier, rng, vec, y), y


# ---------------- Bayes boundaries and HMID ------------------------------------------------
def boundary_is_midpoint(a, b, k, prior_ratio=1.0):
    """True when no crossing of the prior-weighted densities lies between the two means (the boundary then falls back to the midpoint)."""
    m1, s1, m2, s2 = MU[a][k], SD[a][k], MU[b][k], SD[b][k]
    A = 1 / s2 ** 2 - 1 / s1 ** 2
    B = 2 * (m1 / s1 ** 2 - m2 / s2 ** 2)
    C = m2 ** 2 / s2 ** 2 - m1 ** 2 / s1 ** 2 + 2 * np.log(s2 / s1) - 2 * np.log(prior_ratio)
    roots = np.roots([A, B, C]) if abs(A) > 1e-12 else np.array([-C / B])
    roots = roots[np.isreal(roots)].real
    lo, hi = min(m1, m2), max(m1, m2)
    return not any(lo <= x <= hi for x in roots)


def bayes_boundary(a, b, k, prior_ratio=1.0):
    """point where prior_a*N(theta; a) = prior_b*N(theta; b), between the two means; prior_ratio = prior_b/prior_a."""
    m1, s1, m2, s2 = MU[a][k], SD[a][k], MU[b][k], SD[b][k]
    A = 1 / s2 ** 2 - 1 / s1 ** 2
    B = 2 * (m1 / s1 ** 2 - m2 / s2 ** 2)
    C = m2 ** 2 / s2 ** 2 - m1 ** 2 / s1 ** 2 + 2 * np.log(s2 / s1) - 2 * np.log(prior_ratio)
    roots = np.roots([A, B, C]) if abs(A) > 1e-12 else np.array([-C / B])
    roots = roots[np.isreal(roots)].real
    lo, hi = min(m1, m2), max(m1, m2)
    r = [x for x in roots if lo <= x <= hi]
    return float(r[0]) if r else float((m1 + m2) / 2)


RULE_PAIRS = {"D": ("D", "OP", "AL"), "V": ("V", "OP", "AL"), "A": ("A", "CB", "AL"), "V_CB": ("V", "CB", "AL"),
              "P": ("P", "DR", "AL"), "BD": ("BD", "DR", "AL"), "FD": ("FD", "AC", "AL"), "SL": ("SL", "AC", "AL")}


ALERT_PRIOR = 4.0   # alert:impaired prior ratio used for HMID screening thresholds (specificity-oriented)


def make_theta(alert_prior=None):
    w = ALERT_PRIOR if alert_prior is None else alert_prior
    global THETA_MID
    THETA_MID = {key: boundary_is_midpoint(a, b, IDX[f], prior_ratio=(w if b == "AL" else 1.0)) for key, (f, a, b) in RULE_PAIRS.items()}
    return {key: bayes_boundary(a, b, IDX[f], prior_ratio=(w if b == "AL" else 1.0)) for key, (f, a, b) in RULE_PAIRS.items()}


def hmid(X, th=None):
    """HMID reference discriminator (specificity-ordered rules):
    OP: static miosis AND slowed constriction; CB: weakened PLR with pupil not constricted;
    DR: eyelid anomaly without prolonged fixation; AC: prolonged latency or fixation; DR: remaining eyelid anomalies; else AL."""
    th = THETA if th is None else th
    c = {f: X[:, IDX[f]] for f in FEATS}
    eyelid = (c["P"] > th["P"]) | (c["BD"] > th["BD"])
    oculo = (c["SL"] > th["SL"]) | (c["FD"] > th["FD"])
    op = (c["D"] < th["D"]) & (c["V"] < th["V"])
    cb = ~op & (c["A"] < th["A"]) & (c["V"] < th["V_CB"]) & (c["D"] >= th["D"])
    dr = ~op & ~cb & eyelid & ~(c["FD"] > th["FD"])
    ac = ~op & ~cb & ~dr & oculo
    dr2 = ~op & ~cb & ~dr & ~ac & eyelid
    out = np.zeros(len(X), int)
    out[dr | dr2] = 1; out[ac] = 2; out[op] = 3; out[cb] = 4
    return out


class HMID:
    def fit(self, X, y): return self
    def predict(self, X): return hmid(X)


def models():
    return {"HMID": HMID,
            "Gaussian NB": lambda: make_pipeline(StandardScaler(), GaussianNB()),
            "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
            "SVM (RBF)": lambda: make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale")),
            "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0)}


MACRO = np.array([0, 1, 2, 3, 3])


def metrics(y, yp):
    m = dict(acc=accuracy_score(y, yp), f1=f1_score(y, yp, average="macro"))
    rec = recall_score(y, yp, average=None, labels=range(5))
    m.update({f"rec_{s}": rec[i] for i, s in enumerate(STATES)})
    m["alert_fpr"] = float(np.mean(yp[y == 0] != 0)); m["miss"] = float(np.mean(yp[y != 0] == 0))
    ym, ypm = MACRO[y], MACRO[yp]
    m["acc4"] = accuracy_score(ym, ypm); m["f1_4"] = f1_score(ym, ypm, average="macro")
    m["rec_DRUG"] = float(np.mean(ypm[ym == 3] == 3))
    drug = (ym == 3) & (ypm == 3)
    m["drug_subtype_acc"] = float(np.mean(y[drug] == yp[drug])) if drug.any() else float("nan")   # conditional on drug detected as drug
    m["drug_subtype_all"] = float(np.mean(y[ym == 3] == yp[ym == 3]))                              # correct subtype among all true drug cases
    return m


set_scenario("E")


# ---------------- operational condition: individual baseline + n-epoch aggregation --------
def effective_sd(n_ep=10, n_cal=10, rho_b=0.5, noise=None):
    """SD of the calibrated, epoch-averaged feature  x~ = mean_n(x) - mean_ncal(x_alert) + mu_AL  (Eq. eff), derived from
    the same generator as pop_var():  x~ = mu_AL + (1 + eta)(1 + beta) Delta + mean(u) - mean(u0) + noise, so that
    Var = (1 - rho_b)(SD0_s^2/n_ep + SD0_AL^2/n_cal) + (b + c^2 + b c^2) Delta^2 + nz_s^2/n_ep + nz_AL^2/n_cal."""
    z = np.zeros(len(FEATS))
    nz = {s: z for s in STATES} if noise is None else (noise if isinstance(noise, dict) else {s: noise for s in STATES})
    b = rho_b * (SD0["AL"] / MU["AL"]) ** 2
    out = {}
    for s in STATES:
        dl = MU[s] - MU["AL"]
        within = (1 - rho_b) * (SD0[s] ** 2 / n_ep + SD0["AL"] ** 2 / n_cal)
        persist = (b + C_RESP ** 2 + b * C_RESP ** 2) * dl ** 2
        out[s] = np.sqrt(within + persist + nz[s] ** 2 / n_ep + nz["AL"] ** 2 / n_cal)
    return out


def use_effective(n_ep=10, n_cal=10, rho_b=0.5, noise=None):
    """switch SD (and HMID thresholds) to the operational effective distributions."""
    global SD, THETA
    SD = effective_sd(n_ep, n_cal, rho_b, noise)
    THETA = make_theta()


def sample_eff(n, rng, rho=0.3):
    X = np.vstack([sample_state(s, n, rng, rho) for s in STATES]); y = np.repeat(np.arange(5), n)
    return X, y
