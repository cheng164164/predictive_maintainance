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

1. Creates the same outer chronological split used by feature selection:
   `training_main`, `validation_holdout`, and `test_holdout`.
2. Runs expanding-window date-based CV inside `training_main` only.
3. Uses mean validation PR-AUC / average precision as the CV hyperparameter metric.
4. Records F2, recall, precision, flagged rate, and snapshot-level top-K metrics per CV fold.
5. Optionally records machine-level top-K metrics by collapsing repeated snapshots per machine.
6. Uses a configurable 45-day gap between each fold training end and CV start.
7. Retrains final XGBoost models on full `training_main`.
8. Chooses the probability threshold on `validation_holdout` using F2 under a max flagged-rate constraint.
9. Evaluates the final selected model once on `test_holdout` by default.

## Main settings

Edit `config.py` only. Important parameters include:

```python
INPUT_DATA_PATH
MODEL_VARIANTS_TO_RUN
AUTO_SELECT_FINAL_VARIANT_BY_VALIDATION_F2
FINAL_MODEL_VARIANT
MAX_FLAGGED_RATE
CV_GAP_DAYS
CV_VALIDATION_WINDOW_DAYS
XGB_DEFAULT_PARAMS
HYPERPARAMETER_GRID
ENABLE_MACHINE_LEVEL_TOP_K
MACHINE_ID_COL
MACHINE_PROBABILITY_AGGREGATION
MACHINE_TARGET_AGGREGATION
```

## Feature variants

- Model A: all 93 prepared features from feature-selection review.
- Model B: Model A minus high-confidence drop candidates.
- Model C: Model B minus additional review candidates.
- Model D: Model C plus protected machine-context features such as `full_model` one-hot features.

Step `00_split_data.py` writes feature-set documentation files:

```text
machine_learning/output/00_data_split/03_feature_set_summary.csv
machine_learning/output/00_data_split/04_feature_sets_prepared_features.csv
machine_learning/output/00_data_split/05_model_D_prepared_features.csv
```

`05_model_D_prepared_features.csv` is the direct list of prepared features used by Model D.

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

The final validation/test step also writes per-variant machine-level top-K files.

The scripts do not import or depend on the feature-selection project.

## Optional reduced Model D snapshot CSV

`00_split_data.py` can save a source-level snapshot dataframe reduced to the
columns needed by one selected model variant. By default this is enabled for
Model D and writes:

```text
machine_learning/output/00_data_split/06_snapshot_dataframe_model_D_reduced.csv
```

The file includes ID/date/target columns, an optional `split` column, and the
original source columns required to generate Model D prepared features. For
example, one-hot prepared features like `cat__full_model_*` are represented by
their source column `full_model`.

Control it in `config.py`:

```python
SAVE_REDUCED_SNAPSHOT_DATAFRAME = True   # set False to disable
REDUCED_SNAPSHOT_MODEL_VARIANT = "D"
REDUCED_SNAPSHOT_OUTPUT_FILENAME = "06_snapshot_dataframe_model_D_reduced.csv"
REDUCED_SNAPSHOT_INCLUDE_SPLIT_COLUMN = True
```
