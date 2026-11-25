# Methodology

## 1. Why the protocol changed

The preserved group notebook is the historical coursework artefact. The public rebuild treats it as a starting point, not as a specification that must be defended forever.

The v2 audit identified four statistical issues worth correcting:

1. calibration quality was reported on the same observations used to fit the isotonic calibrator;
2. the cost threshold was tuned on that same set;
3. SMOTE deliberately changed the training class prior and therefore distorted the probability scale;
4. the old boosted/logistic comparison did not use an identical modelling protocol, making the quoted AUC gap harder to interpret.

The new protocol uses disjoint fit, calibration and policy subsets inside the development file, plus the original untouched 6,000-row test file.

## 2. Feature contract

The final model deliberately uses behavioural credit information only.

For credit limit `L`, bill statements `B_t`, payments `P_t` and repayment-status codes `S_t`, the core derived features include

```text
recent utilisation       = B_1 / L
mean utilisation         = mean(B_1,...,B_6) / L
bill change              = (B_1 - B_6) / L
mean payment / limit     = mean(P_1,...,P_6) / L
payment / exposure       = sum(P_t) / (sum(|B_t|) + L)
recent delay             = max(S_1, 0)
delinquent months        = sum( max(S_t,0) > 0 )
```

The positive-delay transformation intentionally asks **how many months overdue** the account is; source codes at or below zero are therefore non-positive delinquency for this summary.

`SEX`, `MARRIAGE`, `AGE` and `EDUCATION` are excluded from the deployed feature matrix. A separate audit adds them back to quantify the discrimination trade-off; it does not silently change the deployed model.

## 3. Model fitting

The 24,000-row development file is stratified into 60% fit, 20% calibration and 20% policy subsets.

Regularised logistic regression is selected by five-fold stratified ROC-AUC over

```text
C in {0.01, 0.1, 1, 10}
```

inside the fit split only.

The final score is

```text
p_raw(x) = sigmoid(beta_0 + beta^T z)
```

where `z` is the standardised feature vector.

## 4. Probability calibration

An isotonic map is fitted on the separate calibration split:

```text
p_cal = g(p_raw)
```

with `g` monotone. Probability quality is then assessed on the untouched test file using Brier score.

Because isotonic calibration can create ties, ROC-AUC can move slightly even though the map is monotone in the weak sense. Ranking claims therefore use the raw model scores; probability/decision claims use calibrated probabilities.

## 5. Cost-sensitive decision rule

Let

- `C_FN` = cost of missing a defaulter;
- `C_FP` = cost of flagging a non-defaulter;
- correct decisions have zero cost in the illustrative model.

Flag when the expected cost of *not* flagging exceeds the expected cost of flagging:

```text
p * C_FN > (1-p) * C_FP
```

which gives

```text
p > C_FP / (C_FN + C_FP)
```

For a 5:1 ratio, the transparent threshold is `1/6 ~= 0.167`.

The policy split is also swept empirically as a diagnostic. The empirical point optimum is bootstrapped because one exact threshold can be unstable in finite samples. The empirical sweep does **not** redefine the held-out test rule.

## 6. Model-comparison design

Histogram gradient boosting is tuned on the same fit rows as logistic regression and calibrated on the same calibration rows. Both are then scored on the same untouched test clients.

The reported AUC gap is paired-bootstrap resampled by client. This makes the interpretability/performance trade-off a measured quantity rather than a rhetorical claim.

## 7. Why SMOTE is an audit, not the deployed pipeline

SMOTE synthesises minority-class examples and changes the effective class prior seen by the learner. That can be useful for classification boundaries, but probabilities trained under the synthetic prior need not have the meaning required by a cost-sensitive decision rule.

The project therefore reproduces SMOTE in `resampling_audit.py`, then measures both ranking and Brier score before and after independent calibration. In this dataset it does not improve ranking and makes raw calibration much worse, so the final model keeps the natural class prior.

## 8. Explanation scope

For logistic regression on standardised features,

```text
logit(p_raw) = beta_0 + sum beta_j z_j
```

so `beta_j z_j` is exact on the raw-logit scale. The contributions sum to the raw logit by construction.

They do not sum to `p_cal`, because isotonic calibration is not additive. They also do not imply causality.

## 9. Governance audit

The governance module has two limited purposes:

1. quantify the development/test AUC change when direct demographic attributes are added to the behaviour-only specification;
2. report held-out recall, false-positive rate, flag rate and Brier score across source-reported sex categories and age bands under one global threshold.

It deliberately does **not** optimise group-specific thresholds or claim that any metric establishes fairness, discrimination or legal compliance.


## Notes moved from the README

- The earlier headline of 0.197 to 0.137 Brier score was measured on the same validation data used to fit the isotonic calibrator, so I retired it. On the held-out test file the raw logistic Brier score is 0.13114 and the isotonic calibrated score is 0.13101.
- SMOTE under the clean split protocol: held-out raw Brier 0.179 and ROC-AUC 0.773, against 0.131 and 0.776 for the plain logistic model. After disjoint isotonic calibration both Brier scores return to about 0.131.
- The empirical policy-split threshold optimum is 0.162, with the same measured policy cost at the theoretical 1/6. A bootstrap of that optimum spans about 0.147 to 0.320.
- Demographic audit (SEX, MARRIAGE, AGE, EDUCATION added back): policy AUC 0.7464 for behaviour only and 0.7496 with demographics, a gain of 0.0032. The test confirmation gain is 0.0042 AUC. The smallest age band holds 56 clients, so subgroup diagnostics are uncertain.
- The behaviour feature set replaces the original recursive AV_EXP chain, which mixed normalised quantities with raw currency amounts. Outlier deletion from the coursework pipeline was also dropped.
- Reason codes (beta_j * z_j) are exact on the raw log-odds scale. They do not sum to the calibrated probability and are not causal explanations.
- Logistic against histogram gradient boosting: ROC-AUC 0.776 against 0.802, a gap of 2.61 AUC points (paired bootstrap 95% interval 1.72 to 3.50), on the same fit, calibration and test splits.
