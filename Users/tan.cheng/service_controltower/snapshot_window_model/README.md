# Machine-learning training and validation workflow

Run from this folder:

```bash
python main.py
```

Outputs are saved under:

```text
machine_learning/output
```

## What the workflow does

1. Loads the snapshot dataframe from `data_preparation/output/snapshot_dataframe`.
2. Optionally cleans `9999` sentinel values in configured `days_since_*` columns.
3. Creates the same outer chronological split used by feature selection: `training_main`, `validation_holdout`, and `test_holdout`.
4. Runs expanding-window date-based CV inside `training_main` only.
5. By default, compares XGBoost, LightGBM, and Random Forest using configured default settings with no tuning.
6. If XGBoost hyperparameter tuning is enabled, expands the XGBoost grid configured in `config.py` and reports progress as `1/total`, `2/total`, etc.
7. Records average precision, ROC AUC, F2, recall, precision, flagged rate, snapshot-level top-K, and optional machine-level top-K metrics.
8. Retrains final model(s) on full `training_main`.
9. Chooses the probability threshold on `validation_holdout` using F2 under a max flagged-rate constraint.
10. Evaluates the selected final model once on `test_holdout` by default.

## Main settings

Edit `config.py` only. Important parameters include:

```python
INPUT_DATA_PATH
MODEL_VARIANTS_TO_RUN
MODEL_ALGORITHMS_TO_RUN
HYPERPARAMETER_TUNING_ENABLED
SAVE_REDUCED_SNAPSHOT_DATAFRAME
REDUCED_SNAPSHOT_MODEL_VARIANT
SENTINEL_CLEANING_ENABLED
SENTINEL_COLUMNS_TO_CLEAN
AUTO_SELECT_FINAL_VARIANT_BY_VALIDATION_F2
FINAL_MODEL_ALGORITHM
FINAL_MODEL_VARIANT
MAX_FLAGGED_RATE
CV_GAP_DAYS
CV_VALIDATION_WINDOW_DAYS
ENABLE_MACHINE_LEVEL_TOP_K
MACHINE_ID_COL
```

## Feature variants

- Model A: all prepared features from feature-selection review.
- Model B: Model A minus high-confidence drop candidates.
- Model C: lean feature set without the `full_model` categorical context.
- Model D: Model C plus protected machine-context features such as `full_model` one-hot features.

The default is:

```python
MODEL_VARIANTS_TO_RUN = ["C"]
REDUCED_SNAPSHOT_MODEL_VARIANT = "C"
```

Step `00_split_data.py` writes feature-set documentation files:

```text
machine_learning/output/00_data_split/03_feature_set_summary.csv
machine_learning/output/00_data_split/04_feature_sets_prepared_features.csv
machine_learning/output/00_data_split/05_model_C_prepared_features.csv
machine_learning/output/00_data_split/05_model_D_prepared_features.csv
```

## 9999 sentinel cleaning

The recommended default is enabled:

```python
SENTINEL_CLEANING_ENABLED = True
SENTINEL_VALUE = 9999
SENTINEL_COLUMNS_TO_CLEAN = {
    "days_since_last_fault": "has_prior_fault",
    "days_since_last_severe_fault": "has_prior_severe_fault",
    "days_since_last_claim": "has_prior_claim",
    "days_since_last_reset": "has_prior_reset",
    "days_since_last_smr": "has_prior_smr",
}
```

For each configured column, the workflow creates a `has_prior_*` indicator and replaces the `9999` value with missing before preprocessing. The numeric imputer then handles the cleaned `days_since_*` value, while the indicator preserves the information that no prior event was observed.

Sentinel cleaning reports are saved as:

```text
01b_sentinel_cleaning_report.csv
00b_sentinel_cleaning_report.csv
```

## Reduced snapshot CSV

`00_split_data.py` can save a source-level dataframe reduced to the columns needed by one selected model variant. The export includes traceability/context columns (`model_id`, `snapshot_date`, and the target column), then only the source-level feature columns required by the selected model variant. It intentionally excludes the train/validation/test split column.

For categorical features, the export keeps the original source column such as `full_model`; it does not save one-hot encoded columns. With the default Model C setting, `full_model` is not included because Model C drops the full-model categorical features.

Default settings:

```python
SAVE_REDUCED_SNAPSHOT_DATAFRAME = True
REDUCED_SNAPSHOT_MODEL_VARIANT = "C"
REDUCED_SNAPSHOT_OUTPUT_FILENAME = "06_snapshot_dataframe_model_C_reduced_snapshot.csv"
```

Output:

```text
machine_learning/output/00_data_split/06_snapshot_dataframe_model_C_reduced_snapshot.csv
machine_learning/output/00_data_split/06_snapshot_dataframe_model_reduced_metadata.json
```

## Default algorithm comparison

With tuning disabled:

```python
HYPERPARAMETER_TUNING_ENABLED = False
MODEL_ALGORITHMS_TO_RUN = ["xgboost", "lightgbm", "random_forest"]
```

`01_cross_validation.py` runs each algorithm once using configured default settings and saves results under:

```text
machine_learning/output/01_cross_validation/
```

Useful files:

```text
02_cv_metrics_by_fold.csv
03_cv_param_summary.csv
06_cv_top_k_metrics_by_fold.csv
07_cv_top_k_summary.csv
08_cv_machine_top_k_metrics_by_fold.csv
09_cv_machine_top_k_summary.csv
```

`02_train_validate_test.py` then trains final model(s), selects the validation threshold, and evaluates the selected final model on test.

## XGBoost hyperparameter tuning

To tune XGBoost, set:

```python
HYPERPARAMETER_TUNING_ENABLED = True
HYPERPARAMETER_TUNING_ALGORITHM = "xgboost"
```

The script then ignores the multi-algorithm default comparison and tunes XGBoost only using:

```python
HYPERPARAMETER_SEARCH_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [3, 4],
    "learning_rate": [0.03],
    "min_child_weight": [5, 20],
    "subsample": [0.85],
    "colsample_bytree": [0.85],
    "scale_pos_weight": [1, 3, "auto"],
}
```

Progress is printed during CV, for example:

```text
CV run plan
  mode: xgboost_hyperparameter_tuning
  parameter configurations per algorithm: 24
  folds: 4
  total model fits: 96
[CV 1/96] algorithm=xgboost variant=C param=grid_001... fold=1
[CV 2/96] algorithm=xgboost variant=C param=grid_001... fold=2
```

Tuning outputs:

```text
10_hyperparameter_tuning_results.csv
11_hyperparameter_tuning_best_by_variant.csv
12_hyperparameter_tuning_config.json
```

## Machine-level top-K

Set this in `config.py`:

```python
ENABLE_MACHINE_LEVEL_TOP_K = True
MACHINE_ID_COL = "model_id"
MACHINE_PROBABILITY_AGGREGATION = "max"
MACHINE_TARGET_AGGREGATION = "max"
```

When enabled, CV writes:

```text
08_cv_machine_top_k_metrics_by_fold.csv
09_cv_machine_top_k_summary.csv
```

Machine-level top-K collapses repeated snapshot rows into one machine-level risk score before ranking. This is often more actionable than snapshot-level top-K for inspection or worklist planning.

The scripts do not import or depend on the feature-selection project.
