# Model card — credit-default benchmark

## Intended use

Educational/research benchmark for probability estimation, cost-sensitive decision rules, model interpretability and governance diagnostics on a public historical dataset.

## Out of scope

- real lending decisions;
- automated approval/decline;
- regulatory compliance claims;
- present-day population inference;
- causal statements about why an individual defaults.

## Data

Public UCI Taiwan credit-card default sample, 30,000 clients. The repository does not intentionally commit raw data; `src/download_data.py` recreates the original 24k/6k coursework split.

## Deployed specification in this repository

Regularised logistic regression on eight behavioural features. `SEX`, `MARRIAGE`, `AGE` and `EDUCATION` are excluded from model inputs and retained only for diagnostic auditing.

## Validation design

Development file: 60% fit / 20% calibration / 20% policy diagnostics. Original 6,000-row test file remains untouched for final metrics.

## Headline held-out metrics

- raw logistic ROC-AUC: ~0.776
- calibrated Brier: ~0.131
- at illustrative 5:1 cost rule: recall ~0.714, precision ~0.385
- histogram gradient boosting ROC-AUC: ~0.802
- discrimination gap vs logistic: ~2.61 AUC points, paired-bootstrap 95% interval ~1.72–3.50 points

Exact machine-readable values live in `reports/*.json`.

## Known limitations

- historical, geographically specific dataset;
- no temporal validation;
- no reject inference / accepted-applicant selection model;
- illustrative constant misclassification costs, not profit/loss modelling;
- subgroup estimates can be noisy, especially small age bands;
- probability calibration is sample-dependent;
- reasons are model contributions, not causal explanations;
- no production controls such as authentication, audit logging, model registry or monitoring against observed future outcomes.
