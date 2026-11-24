"""Show why the original SMOTE step is not used by the v2 deployed model.

SMOTE is a useful coursework technique for class imbalance, but changing the
training class prior can badly distort raw probabilities.  This audit compares
plain and SMOTE logistic regression on identical splits, before and after a
disjoint isotonic calibration step.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import behavioural_features, clean, read_credit_csv, target
from modeling import calibrate_frozen, development_splits, probability_metrics

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def main(seed=42):
    dev = clean(read_credit_csv(RAW / "CreditCard_train.csv"))
    test = clean(read_credit_csv(RAW / "CreditCard_test.csv"))
    X, y = behavioural_features(dev), target(dev)
    Xt, yt = behavioural_features(test), target(test)
    s = development_splits(y, seed)

    plain = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=0.1, max_iter=3000,
                                                 random_state=11))])
    smote = ImbPipeline([("scale", StandardScaler()),
                         ("smote", SMOTE(random_state=11)),
                         ("clf", LogisticRegression(C=0.1, max_iter=3000,
                                                    random_state=11))])
    plain.fit(X.loc[s.fit], y.loc[s.fit])
    smote.fit(X.loc[s.fit], y.loc[s.fit])
    plain_cal = calibrate_frozen(plain, X.loc[s.calibration], y.loc[s.calibration])
    smote_cal = calibrate_frozen(smote, X.loc[s.calibration], y.loc[s.calibration])

    rows = {}
    for name, model, cal in [("plain_logistic", plain, plain_cal),
                             ("smote_logistic", smote, smote_cal)]:
        raw = model.predict_proba(Xt)[:, 1]
        calibrated = cal.predict_proba(Xt)[:, 1]
        rows[name] = {"raw": probability_metrics(yt, raw),
                      "calibrated": probability_metrics(yt, calibrated)}

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    names = ["Plain\nraw", "SMOTE\nraw", "Plain\ncalibrated", "SMOTE\ncalibrated"]
    vals = [rows["plain_logistic"]["raw"]["brier"],
            rows["smote_logistic"]["raw"]["brier"],
            rows["plain_logistic"]["calibrated"]["brier"],
            rows["smote_logistic"]["calibrated"]["brier"]]
    ax.bar(names, vals)
    ax.set(ylabel="Held-out Brier score (lower is better)",
           title="Resampling changes the probability scale")
    for i, v in enumerate(vals): ax.text(i, v + 0.002, f"{v:.3f}", ha="center")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(FIGURES / "resampling_calibration.png", dpi=160)
    plt.close(fig)

    (REPORTS / "resampling_audit.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
