import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_committed_reports_are_current_v2_schema():
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert metrics["protocol"]["demographics_in_model"] is False
    assert metrics["protocol"]["smote_in_deployed_model"] is False
    assert metrics["protocol"]["test_rows"] == 6000
    assert "test_calibrated_brier" in metrics["calibration"]


def test_comparison_is_same_split_and_gap_is_not_hardcoded_old_claim():
    x = json.loads((ROOT / "reports" / "model_comparison.json").read_text())
    assert x["same_fit_calibration_test_splits"] is True
    assert 0 < x["held_out_auc_gap_hgb_minus_logistic"] < 0.1


def test_governance_report_explicitly_disclaims_fairness_conclusion():
    x = json.loads((ROOT / "reports" / "governance_audit.json").read_text())
    assert "do not establish fairness" in x["warning"]
