# Credit Default Decision Modelling

[![CI](https://github.com/AhmadEblagh205/credit-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/credit-default-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This is a credit-default modelling benchmark built on the public UCI Taiwan credit-card dataset (30,000 clients). It began as a five-person UCL BENG0095 group project, where my original contribution was exploratory analysis, feature engineering and the modelling pipeline. The group notebook and report are kept in `notebooks/`. The modular `src/` code is my later individual rebuild.

## What it does

- Downloads the UCI data and recreates the coursework split of 24,000 development rows and 6,000 test rows.
- Fits a regularised logistic regression on behaviour-only features, with isotonic calibration on a separate split.
- Applies a cost rule for false negatives and false positives, with the threshold checked on a third split.
- Compares logistic regression with histogram gradient boosting, and plain with SMOTE-resampled logistic regression.
- Audits demographic exclusion and subgroup metrics, and gives exact reason codes on the raw log-odds scale.

The development rows are split into 14,400 fit, 4,800 calibration and 4,800 policy rows. The test rows are not used for selection, calibration or thresholds. Raw data is git-ignored, and the script downloads it.

## Results

On the 6,000-client held-out test file, the logistic model has ROC-AUC 0.776, average precision 0.540 and calibrated Brier score 0.131.

With an illustrative 5:1 false-negative to false-positive cost, the threshold is 1/6 (0.167). At that threshold:

- recall is 0.714 and precision is 0.385;
- the flag rate is 0.392;
- expected cost is 0.543 per client.

The 5:1 ratio is an illustration, not an estimate of real lender economics.

Histogram gradient boosting reaches ROC-AUC 0.802, so the interpretable model gives up about 2.61 AUC points. SMOTE raises the raw Brier score to 0.179 without improving ranking, so the final pipeline keeps the natural class prior. Adding demographic variables back changes policy AUC by only 0.0032, and the deployed model excludes them.

Figures are in `reports/figures/` and the numbers are in `reports/*.json`.

## How to run

```bash
pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/model_comparison.py
python src/resampling_audit.py
python src/governance_audit.py
python src/explain.py --row 12
pytest -q
```

The test suite has 14 tests.

## Repository layout

```text
src/        features, modeling, decision, train, model_comparison,
            resampling_audit, governance_audit, explain, download_data
docs/       METHODOLOGY.md, MODEL_CARD.md
notebooks/  group_project_original.ipynb, group_project_report.pdf
reports/    metrics.json, audit summaries and figures
```

## Notes

The UCI sample is historical, so it says nothing about present-day performance, and this is not a lending product. The subgroup diagnostics do not establish fairness or legal compliance. Reason codes explain the fitted model and are not causal. Derivations and audit notes are in `docs/METHODOLOGY.md`, and the model card is `docs/MODEL_CARD.md`.

## Licence

MIT for this repository's code. Dataset: UCI Machine Learning Repository, Default of Credit Card Clients; refer to the source dataset page for its current data terms.
