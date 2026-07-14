# Feature Selection Process Documentation

## Purpose

This document explains the feature selection workflow for the predictive maintenance unified snapshot dataset.

The goal is to reduce the number of features before model training while keeping the most useful and reliable predictors for the target:

```text
claim_next_45d
```

## High-level goals

- Remove features that are not useful because they are mostly missing.
- Remove features that do not vary in the training data.
- Identify highly redundant features using grouped correlation analysis.
- Rank features using several independent methods.
- Compare results across methods before making final keep or drop decisions.
- Preserve the validation and test sets for later model validation and final reporting.

## Key design principles

- Feature selection uses only the training portion of the data.
- The main validation set is not used for early feature selection.
- The test set is not used for feature selection.
- The process is split into separate numbered scripts so each step can be reviewed independently.
- The current process creates review reports only. It does not apply a final importance-based keep or drop threshold.

## Dataset split design

The unified snapshot dataframe is sorted by snapshot date and split chronologically.

| Dataset split | Purpose |
|---|---|
| Training set | Main data used for feature selection and temporary model training |
| Validation set | Reserved for later model validation after the feature set is chosen |
| Test set | Reserved for final unbiased performance reporting |

The training set is further split into two internal parts:

| Inner training split | Purpose |
|---|---|
| feature_train | Used to calculate missingness, constant features, statistical tests, and train temporary XGBoost models |
| feature_selection_holdout | Used for permutation importance and SHAP review |

This design allows feature selection to happen without using the main validation or test sets.


## Step 00 - Prepare data and apply first quality filters

### What this step does

- Loads the unified snapshot dataframe.
- Drops columns that should not be considered as model features.
- Applies the chronological train, validation, and test split.
- Splits the training set into `feature_train` and `feature_selection_holdout`.
- Creates raw missing-value reports before preprocessing.
- Drops features with more than 90 percent missing values in `feature_train`.
- Drops configured source columns that should be excluded before feature selection.
- Creates feature group assignments for later grouped correlation analysis.

### Important criteria

| Rule | Criteria | Reason |
|---|---|---|
| Remove ID and target columns | Exclude `model_id`, `snapshot_date`, and `claim_next_45d` | These are identifiers, time index, or target label, not normal model predictors |
| Remove configured source columns | Drop manually configured columns such as sparse fluid metadata columns | Some columns are known to be sparse or not intended for the feature selection review |
| Remove high-missingness features | Drop if missing rate in `feature_train` is greater than 90 percent | Features that are almost always missing are usually unreliable and can add noise |


## Step 01 - Basic unsupervised diagnostics and zero-variance filtering

### What this step does

- Reviews raw feature distributions after the high-missingness filter.
- Identifies features that are constant in `feature_train`.
- Removes raw source features with zero variance before correlation and modeling methods are run.
- Creates diagnostics after preprocessing.

### Important criteria

| Rule | Criteria | Reason |
|---|---|---|
| Remove zero-variance features | Drop features that have only one unique value in `feature_train` | A constant feature cannot help the model separate claim and non-claim snapshots |
| Review near-constant patterns | Report features with very low variation | These may not be removed automatically unless configured, but they are useful for review |


## Step 02 - Grouped correlation analysis

### What this step does

- Calculates pairwise correlations between features.
- Correlation is calculated within feature groups only, not globally across all features.
- Uses the smaller feature set after missingness and zero-variance filters.
- Helps identify redundant features that may carry very similar information.

### Why features are grouped before correlation

A full all-vs-all correlation matrix can become very large when the dataset has many features. Grouping reduces the number of pairs and makes the report easier to review.

Example groups include:

| Feature group | Example features |
|---|---|
| fault_codes | fault_count_90d, action_L03_count_90d, max_event_evidence_score_90d |
| maintenance | maintenance_events_180d, monitor_reset_count_90d, avg_remaining_hours |
| operation | working_hours_sum_90d, engine_running_hours_sum_90d, travel_hours_sum_90d |
| fluid_oil | Fe_Iron_PPM, Soot_Soot_PERCENT, Water_Water_PERCENT |
| smr_usage | smr_latest_hours, smr_delta_90d, days_since_last_smr |
| warranty_prior | prior_claim_count_365d, days_since_last_claim |
| machine_context | full_model and machine-level attributes if included |

### How to interpret results

Highly correlated pairs may indicate duplicate or redundant features. For example:

| Example pair | Possible interpretation |
|---|---|
| working_hours_sum_30d and engine_running_hours_sum_30d | Both measure recent machine usage |
| fault_count_30d and fault_count_90d | Short-term and longer-term fault history may overlap |
| travel_hours_sum_90d and travel_day_count_90d | Both measure recent travel activity |

Correlation does not automatically drop features in the current workflow. It provides evidence for review.


## Step 03 - Statistical tests

### What this step does

Runs supervised univariate statistical ranking methods on `feature_train` only.

The methods include:

| Method | High-level purpose |
|---|---|
| ANOVA F-test | Checks whether feature values differ between claim and non-claim rows |
| Mutual information | Measures dependency between a feature and the target, including nonlinear relationships |
| Chi-squared | Measures association between non-negative feature values and the target |

### How to interpret results

These methods rank features one at a time. A high-ranking feature may have a strong individual relationship with the target.

However, these methods do not fully capture feature interactions. A feature may look weak by itself but still be useful when combined with other features in XGBoost.


## Step 04 - XGBoost built-in feature importance

### What this step does

- Trains a temporary XGBoost model using `feature_train`.
- Extracts built-in XGBoost importance metrics.
- Evaluates the temporary model on `feature_selection_holdout` using threshold-free metrics for context.

### Main XGBoost importance types

| Importance type | Meaning |
|---|---|
| weight | How often a feature is used in tree splits |
| gain | Average improvement when the feature is used in a split |
| cover | Average number of rows affected by splits using the feature |
| total_gain | Total split improvement contributed by the feature |
| total_cover | Total row coverage affected by the feature |

### How to interpret results

XGBoost importance helps identify features the model uses during tree building. Total gain is often useful because it reflects the overall contribution of a feature to improving model splits.

A feature with high XGBoost importance should still be reviewed together with other methods because tree importance can be affected by correlated features.


## Step 05 - Permutation importance

### What this step does

- Trains a temporary XGBoost model on `feature_train`.
- Evaluates each feature on `feature_selection_holdout`.
- Randomly shuffles one feature at a time.
- Measures how much model performance drops after the shuffle.

### Scoring metric

Permutation importance uses F2 scoring.

F2 gives more weight to recall than precision, which is appropriate when missing a future claim is more costly than generating some false positives.

### How to interpret results

If shuffling a feature causes a large F2 score drop, the feature is likely important for model performance on the holdout data.

If shuffling a feature causes little or no performance change, the feature may be less important, redundant with another feature, or not useful for generalization.


## Step 06 - SHAP analysis

### What this step does

- Trains a temporary XGBoost model on `feature_train`.
- Computes SHAP values on `feature_selection_holdout`.
- Summarizes each feature by average absolute SHAP value.

### What SHAP means in this workflow

SHAP explains how much each feature contributes to model predictions.

For example:

| Feature behavior | Possible SHAP interpretation |
|---|---|
| Higher recent fault count increases predicted risk | Positive contribution to future claim probability |
| Longer time since last fault decreases predicted risk | Negative contribution to future claim probability |
| High fluid contamination indicator increases predicted risk | Positive contribution to future claim probability |

The current report uses SHAP mainly for feature ranking and explainability review.

## Step 07 - Consensus report

### What this step does

- Reads output reports from earlier steps.
- Combines feature rankings into a single review table.
- Creates a final Excel workbook and Markdown summary.
- Does not automatically select a final feature list.

### How to use the consensus report

The consensus report helps the team compare evidence across methods.

A feature is stronger when it appears important across multiple methods, for example:

- High statistical-test rank.
- High XGBoost importance.
- High permutation importance.
- High SHAP importance.
- Not highly redundant with another simpler feature.
- Not removed due to missingness or zero variance.
- Makes business and engineering sense.

## What gets dropped automatically today

The current workflow applies only early quality filters automatically.

| Drop category | When it happens | Criteria | Why |
|---|---|---|---|
| ID, date, and target columns | Step 00 | Excluded by configuration | These should not be normal model features |
| Manually configured source columns | Step 00 | Listed in config.py | Known sparse or out-of-scope columns |
| High-missingness features | Step 00 | Missing rate in feature_train is greater than 90 percent | Too sparse for reliable modeling |
| Zero-variance features | Step 01 | Only one unique value in feature_train | No predictive separation is possible |

The workflow does not yet automatically drop features based on:

- Correlation threshold.
- ANOVA rank.
- Mutual information rank.
- Chi-squared rank.
- XGBoost importance rank.
- Permutation importance rank.
- SHAP importance rank.

These method outputs are currently for review and discussion.

## Recommended review process

After running all steps, review the outputs in this order:

1. Review correlation pairs to identify redundant feature clusters.
2. Review statistical-test rankings for simple target relationships.
3. Review XGBoost importance for model-based signal.
4. Review permutation importance for holdout performance impact.
5. Review SHAP importance for explainability and directionality.
6. Review the consensus report to identify candidate keep or drop.

## Example stakeholder interpretation

A feature may be considered a strong keep candidate if:

- It has acceptable missingness.
- It has meaningful variation.
- It ranks well in multiple methods.
- It has a clear business interpretation.
- It is not just a duplicate of another simpler feature.

A feature may be considered a drop candidate if:

- It is missing in more than 10 percent of feature_train.
- It is constant or zero-variance in feature_train.
- It is highly correlated with another stronger and simpler feature.
- It ranks low across all supervised and model-based methods.
- It has unclear business meaning or questionable source reliability.

## limitations

- Rare positive claims can make rankings unstable.
- Correlated features can split importance across similar variables.
- SHAP and XGBoost importance explain model behavior, not causality.
- Final production feature decisions should be reviewed on the full dataset with enough machines and positive examples.

## Final expected deliverables from the workflow

| Deliverable | Purpose |
|---|---|
| Early filter reports | Show which features were removed due to missingness or zero variance |
| Correlation report | Show redundant feature pairs within each feature group |
| Statistical-test reports | Show univariate target relationships |
| XGBoost importance report | Show features used by the temporary model |
| Permutation importance report | Show holdout performance impact using F2 score |
| SHAP report | Show explainability-based feature contribution |
| Consensus report | Bring all evidence together for team review |

## Final takeaway

The feature selection process is designed to be reviewable:

Early steps remove clearly unusable features, such as very sparse or constant columns. Later steps rank the remaining features using multiple independent methods. The final feature list should be selected after reviewing the combined evidence, business meaning, source reliability, and validation performance.
