import numpy as np
import pandas as pd
import pytest

from features import (behavioural_features, clean, demographic_enriched_features,
                      target)


def _tiny():
    return pd.DataFrame({
        "ID": [1, 2], "LIMIT_BAL": [50_000, 200_000], "SEX": [1, 2],
        "EDUCATION": [0, 2], "MARRIAGE": [0, 1], "AGE": [24, 40],
        "PAY_0": [-2, 2], "PAY_2": [-1, 0], "PAY_3": [0, 0],
        "PAY_4": [0, 0], "PAY_5": [0, 0], "PAY_6": [0, 0],
        **{f"BILL_AMT{i}": [1000 * i, 20_000 * i] for i in range(1, 7)},
        **{f"PAY_AMT{i}": [500, 4000] for i in range(1, 7)},
        "default payment next month": [1, 0],
    })


def test_clean_normalises_schema_without_deleting_rows():
    df = clean(_tiny())
    assert len(df) == 2
    assert df.loc[0, "EDUCATION"] == 4
    assert df.loc[0, "MARRIAGE"] == 3
    assert "PAY_1" in df and "DEF" in df


def test_behavioural_features_have_explicit_units_and_no_demographics():
    X = behavioural_features(_tiny())
    assert set(["SEX", "MARRIAGE", "AGE", "EDUCATION"]).isdisjoint(X.columns)
    assert {"RECENT_UTIL", "MEAN_UTIL", "RECENT_DELAY", "DELINQUENT_MONTHS"} <= set(X.columns)
    assert np.isfinite(X.to_numpy()).all()
    assert X.loc[0, "RECENT_DELAY"] == 0  # -2 means no positive delinquency
    assert X.loc[1, "RECENT_DELAY"] == 2


def test_demographic_enriched_model_is_audit_only_matrix():
    X = demographic_enriched_features(_tiny())
    assert any(c.startswith("SEX_") for c in X.columns)
    assert "AGE" in X.columns


def test_target_is_binary():
    assert target(_tiny()).tolist() == [1, 0]


def test_nonpositive_limit_fails_loudly():
    df = _tiny(); df.loc[0, "LIMIT_BAL"] = 0
    with pytest.raises(ValueError):
        behavioural_features(df)
