"""Re-aggregation of published multi-state confusion matrices (rows = truth, columns = prediction; order A, C, D, S)."""
import numpy as np
MS = {"Tapia": np.array([[1644, 2, 18, 126], [597, 10102, 139, 1244], [60, 18, 1203, 7], [127, 108, 0, 650]]),
      "Zurita": np.array([[802, 3, 80, 0], [327, 5633, 354, 608], [0, 0, 110, 0], [55, 34, 0, 95]])}
I = {"A": 0, "C": 1, "D": 2, "S": 3}


def metrics(M):
    M = np.asarray(M, float); n = M.sum(1); tp = np.diag(M)
    rec = tp / n; prec = tp / M.sum(0); f1 = 2 * rec * prec / (rec + prec)
    d = I["D"]; TP = M[d, d]; se = TP / n[d]
    out = {"BA4": rec.mean(), "F1m": f1.mean(), "rec": dict(zip("ACDS", rec)), "n": dict(zip("ACDS", n.astype(int))), "TP": int(TP)}
    for name, comps in (("DvA", "A"), ("DvS", "S"), ("DvAS", "AS"), ("DvAll", "ACS")):
        fp = sum(M[I[c], d] for c in comps); nn = sum(n[I[c]] for c in comps)
        sp = 1 - fp / nn
        out[name] = {"Se": se, "Sp": sp, "PPV": TP / (TP + fp), "BA": (se + sp) / 2, "FP": int(fp), "N": int(nn)}
    return out


R = {k: metrics(v) for k, v in MS.items()}
if __name__ == "__main__":
    for k, r in R.items():
        print(k, round(100 * r["BA4"], 2), round(100 * r["F1m"], 2), {c: round(100 * v, 1) for c, v in r["rec"].items()})
        for t in ("DvA", "DvS", "DvAS", "DvAll"):
            print("  ", t, {a: round(100 * b, 2) if isinstance(b, float) else b for a, b in r[t].items()})
