"""Decision-threshold diagnostics for calibrated probabilities."""
from __future__ import annotations
import numpy as np
from modeling import decision_metrics


def cost_derived_threshold(fn_cost: float, fp_cost: float) -> float:
    if fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("Costs must be positive")
    return float(fp_cost / (fn_cost + fp_cost))


def sweep_threshold(y, p, fn_cost=5.0, fp_cost=1.0,
                    lo=0.02, hi=0.60, step=0.002):
    """Dense grid for plotting/sensitivity, vectorised over thresholds."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    ts = np.arange(lo, hi + step / 2, step)
    pred = p[:, None] >= ts[None, :]
    pos = y[:, None] == 1
    neg = ~pos
    fn = ((~pred) & pos).sum(axis=0)
    fp = (pred & neg).sum(axis=0)
    tp = (pred & pos).sum(axis=0)
    costs = (fn_cost * fn + fp_cost * fp) / len(y)
    recalls = tp / max(int((y == 1).sum()), 1)
    rows = [{
        "threshold": float(t), "cost_per_client": float(c),
        "recall": float(r)
    } for t, c, r in zip(ts, costs, recalls)]
    i = int(np.argmin(costs))
    best = decision_metrics(y, p, float(ts[i]), fn_cost, fp_cost)
    return rows, best


def empirical_best_threshold(y, p, fn_cost=5.0, fp_cost=1.0):
    """Exact empirical optimum over distinct score cut points.

    Sorting once avoids repeatedly constructing confusion matrices and makes
    bootstrap stability diagnostics fast enough for CI/reproducibility.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    total_pos = int(ys.sum())
    fn = total_pos - tp
    cost = (fn_cost * fn + fp_cost * fp) / len(y)

    # Evaluate only at the last observation of each tied probability value;
    # threshold == that score includes the whole tied block.
    ends = np.r_[np.where(ps[:-1] != ps[1:])[0], len(ps) - 1]
    costs = cost[ends]
    best_pos = int(np.argmin(costs))
    i = int(ends[best_pos])
    threshold = float(ps[i])
    return decision_metrics(y, p, threshold, fn_cost, fp_cost)


def bootstrap_empirical_threshold(y, p, fn_cost=5.0, fp_cost=1.0,
                                  n_boot=300, seed=123):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    best = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        best[b] = empirical_best_threshold(
            y[idx], p[idx], fn_cost, fp_cost)["threshold"]
    q = np.quantile(best, [0.025, 0.5, 0.975])
    return {
        "n_boot": int(n_boot),
        "p2_5": float(q[0]),
        "median": float(q[1]),
        "p97_5": float(q[2]),
        "values": best,
    }
