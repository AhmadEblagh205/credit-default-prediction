"""Shared deterministic split, fitting and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DevSplits:
    fit: np.ndarray
    calibration: np.ndarray
    policy: np.ndarray


def development_splits(y: pd.Series, seed: int = 42) -> DevSplits:
    """60/20/20 fit/calibration/policy split of the 24k development file."""
    idx = np.asarray(y.index)
    fit, rest = train_test_split(
        idx, test_size=0.40, stratify=y.loc[idx], random_state=seed)
    calibration, policy = train_test_split(
        rest, test_size=0.50, stratify=y.loc[rest], random_state=seed + 1)
    return DevSplits(np.asarray(fit), np.asarray(calibration), np.asarray(policy))


def tune_logistic(X: pd.DataFrame, y: pd.Series, fit_idx: np.ndarray,
                  seed: int = 11):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=seed)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
        cv=cv, n_jobs=1, return_train_score=False)
    gs.fit(X.loc[fit_idx], y.loc[fit_idx])
    return gs


def calibrate_frozen(model, X_cal: pd.DataFrame, y_cal: pd.Series,
                     method: str = "isotonic"):
    cal = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    cal.fit(X_cal, y_cal)
    return cal


def probability_metrics(y_true, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "average_precision": float(average_precision_score(y_true, p)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def decision_metrics(y_true, p, threshold: float, fn_cost: float = 5.0,
                     fp_cost: float = 1.0) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "flag_rate": float(pred.mean()),
        "cost_per_client": float((fn_cost * fn + fp_cost * fp) / len(y)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
