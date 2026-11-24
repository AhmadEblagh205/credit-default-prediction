# Changelog

## 2.0.0 — defensibility and governance rebuild

- Replaced the original same-split calibration/threshold workflow with disjoint fit, calibration and policy subsets plus the untouched 6,000-row test file.
- Replaced the recursive `AV_EXP*` feature chain with compact, dimensionally interpretable behaviour ratios and delinquency summaries.
- Removed arbitrary extreme-customer deletion from the deployed pipeline.
- Retired SMOTE from the deployed probability model after a reproducible audit showed much worse raw Brier score without better discrimination; retained the comparison as an explicit experiment.
- Excluded direct demographic attributes from the deployed score and added a separate governance diagnostic rather than silently using them.
- Replaced the old XGBoost-vs-logistic script with a same-split, same-feature logistic vs histogram-gradient-boosting comparison.
- Added a paired bootstrap interval around the held-out AUC trade-off.
- Re-derived the transparent cost threshold `C_FP / (C_FN + C_FP)` and separated it from an empirical policy-split threshold diagnostic with bootstrap stability.
- Corrected calibration reporting: the old `0.197 -> 0.137` validation-set headline is retired because the isotonic calibrator was fitted on that same set.
- Clarified that exact reason codes sum to the **raw logistic logit**, not the isotonic-calibrated probability and not a causal explanation.
- Added `docs/METHODOLOGY.md` and `docs/MODEL_CARD.md`.
- Expanded tests to cover split isolation, decision-cost algebra, feature contracts, explanation additivity and committed-report schema.

## 1.x — original modular rebuild

The earlier public rebuild modularised the UCL group coursework pipeline, adding tests, calibration, cost-sensitive thresholding, reason-code demonstrations and tree/boosting comparisons. The preserved notebook/report remain the historical source artefacts.
