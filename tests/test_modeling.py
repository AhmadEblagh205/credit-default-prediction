import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import development_splits, decision_metrics
from explain import contributions


def test_development_splits_are_disjoint_and_complete():
    y = pd.Series([0, 1] * 100, index=np.arange(200))
    s = development_splits(y, seed=1)
    sets = [set(s.fit), set(s.calibration), set(s.policy)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(y.index)
    assert [len(s.fit), len(s.calibration), len(s.policy)] == [120, 40, 40]


def test_logit_contributions_are_exact():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = (X.a - X.b + rng.normal(0, .5, 300) > 0).astype(int)
    m = Pipeline([("scale", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    p, logit, ranked = contributions(m, X.iloc[0], list(X.columns))
    assert np.isclose(sum(v for _, v in ranked) + m.named_steps["clf"].intercept_[0], logit)
    assert np.isclose(1/(1+np.exp(-logit)), p)


def test_decision_metrics_account_for_costs():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .8, .2, .9])
    m = decision_metrics(y, p, .5, fn_cost=5, fp_cost=1)
    assert m["fp"] == 1 and m["fn"] == 1
    assert np.isclose(m["cost_per_client"], 1.5)
