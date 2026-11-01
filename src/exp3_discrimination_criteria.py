"""Experiment 3 - pairwise ocular discrimination criteria from the evidence-derived distributions (scenario E).
Output: results/results3.json"""
import json, os, sys, itertools
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
from core import STATES, FEATS, bayes_boundary
rng = np.random.default_rng(3)
RES = {}
for cond in ("population", "operational"):
  core.set_scenario("E")
  if cond == "operational":
    core.use_effective(10, 10, 0.5, core.noise_sd("Recommended"))
  MU, SD = core.MU, core.SD
  out = {}
  for a, b in itertools.combinations(STATES, 2):
    J = (MU[a] - MU[b]) ** 2 / (SD[a] ** 2 + SD[b] ** 2); o = np.argsort(-J)[:2]
    rows = []
    for k in o:
        th = bayes_boundary(a, b, k); hi_a = MU[a][k] > MU[b][k]
        pa = 1 - norm.cdf(th, MU[a][k], SD[a][k]) if hi_a else norm.cdf(th, MU[a][k], SD[a][k])
        pb = norm.cdf(th, MU[b][k], SD[b][k]) if hi_a else 1 - norm.cdf(th, MU[b][k], SD[b][k])
        rows.append(dict(feat=FEATS[k], J=float(J[k]), theta=th, theta_rel=float(th / MU["AL"][k] - 1), acc=float((pa + pb) / 2), dir_a=">" if hi_a else "<"))
    n = 200000; k1, k2 = o
    def ll2(x, s): return norm.logpdf(x[:, 0], MU[s][k1], SD[s][k1]) + norm.logpdf(x[:, 1], MU[s][k2], SD[s][k2])
    xa = np.c_[rng.normal(MU[a][k1], SD[a][k1], n), rng.normal(MU[a][k2], SD[a][k2], n)]
    xb = np.c_[rng.normal(MU[b][k1], SD[b][k1], n), rng.normal(MU[b][k2], SD[b][k2], n)]
    acc2 = 0.5 * (np.mean(ll2(xa, a) > ll2(xa, b)) + np.mean(ll2(xb, b) > ll2(xb, a)))
    # deterministic value of the same two-feature diagonal-Gaussian rule (grid integration) and Monte Carlo s.e.
    g = [np.linspace(min(MU[a][k] - 8 * SD[a][k], MU[b][k] - 8 * SD[b][k]), max(MU[a][k] + 8 * SD[a][k], MU[b][k] + 8 * SD[b][k]), 1601) for k in (k1, k2)]
    G1, G2 = np.meshgrid(g[0], g[1], indexing="ij"); P = np.c_[G1.ravel(), G2.ravel()]
    la, lb = ll2(P, a), ll2(P, b); da = np.exp(la); db = np.exp(lb); cell = (g[0][1] - g[0][0]) * (g[1][1] - g[1][0])
    acc2_det = 0.5 * (np.sum(da[la > lb]) + np.sum(db[lb >= la])) * cell
    mc_se = float(np.sqrt(acc2 * (1 - acc2) / (2 * n)))
    def lla(x, s): return norm.logpdf(x, MU[s], SD[s]).sum(1)
    Xa = rng.normal(MU[a], SD[a], (n, len(FEATS))); Xb = rng.normal(MU[b], SD[b], (n, len(FEATS)))
    accall = 0.5 * (np.mean(lla(Xa, a) > lla(Xa, b)) + np.mean(lla(Xb, b) > lla(Xb, a)))
    out[f"{a}-{b}"] = dict(f1=rows[0], f2=rows[1], acc2=float(acc2), acc2_det=float(acc2_det), accall=float(accall), mc_se=mc_se)
    print(cond, a, b, rows[0]["feat"], round(rows[0]["acc"], 3), rows[1]["feat"], round(rows[1]["acc"], 3), round(acc2, 3), round(accall, 3))
  RES[cond] = dict(pairs=out, theta=core.THETA)
json.dump(RES, open(os.path.join(core.RESULTS_DIR, os.environ.get("RESULTS3_NAME", "results3.json")), "w", encoding="utf-8"), indent=1)
