"""Exact reason codes for the final behaviour-only logistic model.

Contributions are exact on the *raw logistic log-odds* scale:
    intercept + sum_j beta_j z_j
The isotonic calibrator is monotone but not additive, so the reason codes must
not be described as additive explanations of the calibrated probability.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from features import behavioural_features, clean, read_credit_csv, target
from modeling import calibrate_frozen, development_splits, tune_logistic

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def contributions(model, x_row, columns):
    scale = model.named_steps["scale"]
    clf = model.named_steps["clf"]
    z = scale.transform(x_row.to_frame().T)[0]
    vals = clf.coef_[0] * z
    logit = float(clf.intercept_[0] + vals.sum())
    raw_probability = float(1.0 / (1.0 + np.exp(-logit)))
    ranked = sorted(zip(columns, vals), key=lambda kv: abs(kv[1]), reverse=True)
    return raw_probability, logit, [(k, float(v)) for k, v in ranked]


def reason_codes(ranked, positive=True, top=3):
    if positive:
        vals = [(k, v) for k, v in ranked if v > 0][:top]
        label = "raised"
    else:
        vals = [(k, v) for k, v in ranked if v < 0][:top]
        label = "lowered"
    return [f"{k} {label} the raw default log-odds ({v:+.3f})" for k, v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = clean(read_credit_csv(RAW / "CreditCard_train.csv"))
    test = clean(read_credit_csv(RAW / "CreditCard_test.csv"))
    X, y = behavioural_features(dev), target(dev)
    Xt, yt = behavioural_features(test), target(test)
    s = development_splits(y, args.seed)
    search = tune_logistic(X, y, s.fit)
    model = search.best_estimator_
    cal = calibrate_frozen(model, X.loc[s.calibration], y.loc[s.calibration])

    row = Xt.iloc[args.row]
    raw_p, logit, ranked = contributions(model, row, list(X.columns))
    cal_p = float(cal.predict_proba(row.to_frame().T)[0, 1])
    out = {
        "test_row": int(args.row), "actual_default": int(yt.iloc[args.row]),
        "raw_logit": logit, "raw_probability": raw_p,
        "calibrated_probability": cal_p,
        "risk_factors": reason_codes(ranked, True),
        "protective_factors": reason_codes(ranked, False),
        "explanation_scope": "exact additive explanation of raw logistic log-odds; not additive on the isotonic-calibrated probability scale",
    }
    (ROOT / "reports" / "example_reasons.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
