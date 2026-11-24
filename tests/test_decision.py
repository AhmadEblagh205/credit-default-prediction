import numpy as np
from decision import cost_derived_threshold, sweep_threshold, bootstrap_empirical_threshold


def test_cost_derived_threshold_is_analytic():
    assert np.isclose(cost_derived_threshold(5, 1), 1/6)


def test_threshold_sweep_returns_cost_minimum():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.2, 0.3, 0.9])
    rows, best = sweep_threshold(y, p, 5, 1, lo=0.05, hi=0.5, step=0.05)
    assert best["cost_per_client"] == min(r["cost_per_client"] for r in rows)


def test_bootstrap_threshold_interval_is_ordered():
    y = np.array([0, 0, 0, 1, 1, 1] * 20)
    p = np.array([.05, .1, .2, .3, .6, .9] * 20)
    b = bootstrap_empirical_threshold(y, p, n_boot=30, seed=0)
    assert b["p2_5"] <= b["median"] <= b["p97_5"]
