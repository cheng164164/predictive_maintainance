# Case-Control Modeling Experiment Workflow

This package is organized as a staged workflow for warranty-claim risk modeling with XGBoost.

## Standard validation workflow

Run the normal workflow with:

```bash
python main.py
```

This executes:

```text
00_profile_sources.py
01_build_claim_episodes.py
02_build_case_control_dataset.py
03_cross_validation.py
04_fit_validate_model_report.py
```

`04_fit_validate_model_report.py` fits on the chronological training split and evaluates validation views only. It does not evaluate the test set.

## Phase 1 design sweep

Run:

```bash
python 05_run_design_sweep.py
```

The design sweep is intentionally compact:

- It does not run cross validation.
- It directly fits on the training split and evaluates validation views.
- It does not save XGBoost learning curves.
- It does not save SHAP values.
- It does not save feature-importance artifacts.
- It does not save detailed prediction files.
- It does not evaluate the test set.

Configure the sweep in `config.py` using list-style grids:

```python
DESIGN_SWEEP_GRID = {
    "CONTROLS_PER_POSITIVE_CASE": [3, 5, 10],
    "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE": [10],
    "scale_pos_weight": ["none", 3.0, "auto"],
}
```

Experiment IDs are created automatically, for example:

```text
ctrl3__valneg10__spwnone
ctrl5__valneg10__spw3p0
ctrl10__valneg10__spwauto
```

Main summary outputs:

```text
output/05_design_sweep/design_sweep_run_summary.csv
output/05_design_sweep/design_sweep_validation_summary_for_review.csv
output/05_design_sweep/design_sweep_validation_metrics.csv
output/05_design_sweep/design_sweep_validation_top_k.csv
```

Use `design_sweep_validation_summary_for_review.csv` for fast comparison. Prioritize population-like validation top-k precision, lift, and average precision.

## Phase 2 validation diagnostics for a chosen design

After the Phase 1 sweep identifies a promising design, update the regular config values and run:

```bash
python 02_build_case_control_dataset.py
python 03_cross_validation.py
python 04_fit_validate_model_report.py
```

This validation report can save detailed outputs, feature importance, SHAP values, and learning curves depending on the config flags:

```python
VALIDATION_SAVE_DETAILED_OUTPUTS = True
SAVE_FEATURE_IMPORTANCE = True
SAVE_SHAP_VALUES = True
XGBOOST_ENABLE_LEARNING_CURVE = True
```

## Phase 3 coarse XGBoost hyperparameter tuning

Run:

```bash
python 06_tune_xgboost_hyperparameters.py
```

Configure the fixed data design and coarse grid in `config.py`:

```python
HYPERPARAMETER_TUNING_DATA_DESIGN = {
    "CONTROLS_PER_POSITIVE_CASE": 5,
    "VALIDATION_RANDOM_NEGATIVES_PER_POSITIVE": 10,
    "scale_pos_weight": 3.0,
}

HYPERPARAMETER_TUNING_GRID = {
    "max_depth": [2, 3],
    "min_child_weight": [5, 10],
    "subsample": [0.85],
    "colsample_bytree": [0.85],
    "gamma": [0, 1],
    "reg_lambda": [1, 10],
    "reg_alpha": [0, 0.1],
    "learning_rate": [0.03],
    "n_estimators": [800],
    "early_stopping_rounds": [0, 50],
}
```

Main summary outputs:

```text
output/06_xgboost_hyperparameter_tuning/hyperparameter_tuning_run_summary.csv
output/06_xgboost_hyperparameter_tuning/hyperparameter_tuning_summary_for_review.csv
output/06_xgboost_hyperparameter_tuning/hyperparameter_tuning_validation_metrics.csv
output/06_xgboost_hyperparameter_tuning/hyperparameter_tuning_validation_top_k.csv
```

This step also prints highlighted progress for each grid-search run.

## Final test evaluation

Only after design and hyperparameters are locked, edit the final parameters in `config.py`:

```python
FINAL_CONTROLS_PER_POSITIVE_CASE = 5
FINAL_TEST_RANDOM_NEGATIVES_PER_POSITIVE = 20
FINAL_XGBOOST_CLASS_IMPORTANCE_MODE = "fixed"
FINAL_XGBOOST_FIXED_SCALE_POS_WEIGHT = 3.0
FINAL_XGBOOST_PARAMS = {...}
FINAL_SAVE_MODEL_ARTIFACT = True
```

Then run:

```bash
python 07_final_test_evaluation.py
```

This is the only stage intended to evaluate the test set and save the final model artifact.


## Evaluation-only future-claim horizon

The training dataset still uses the original case-control `target`. For model
validation, cross-validation, and final test reporting, the scripts can instead
score against a relaxed future-claim label. Configure this in `config.py`:

```python
EVALUATION_TARGET_MODE = "claim_within_horizon"
EVALUATION_CLAIM_HORIZON_DAYS = 120
EVALUATION_INCLUDE_CLAIM_ON_WINDOW_END = True
```

This means a validation/test window is counted as positive when the same machine
has a claim on or after `window_end` and within the next 120 days. This does not
change how the case-control training rows are generated or how the model is fit.
Prediction files include `next_claim_date_on_or_after_window_end`,
`days_to_next_claim_on_or_after_window_end`, `future_claim_lead_time_bucket`, and
`eval_target_claim_within_next_<N>d` columns so you can review how much lead time
the model is giving.


## Evaluation horizon sweep update

`EVALUATION_CLAIM_HORIZON_DAYS` can now be a list, for example `[30, 60, 90, 120, 180, 365]`. Step 04 fits the model once and evaluates validation metrics for every configured horizon. Training still uses the original case-control `target`. The main review file is `validation_horizon_trend_summary_for_review.csv`. The older `EVALUATION_ADDITIONAL_CLAIM_HORIZON_DAYS` setting was removed; use only `EVALUATION_CLAIM_HORIZON_DAYS`.
