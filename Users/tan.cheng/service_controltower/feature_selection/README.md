# Feature Selection Analysis

This folder contains a config-driven, step-by-step feature-selection analysis workflow for the unified snapshot dataframe.

The workflow is intentionally split into numbered scripts so you can run one method at a time, inspect its output folder, then continue to the next step. `main.py` is still available when you want to run every step in order.

## Default input

By default, `config.py` expects the production unified snapshot dataframe here:

```text
../data_preparation/output/snapshot_dataframe.csv
```

This matches the project layout where `feature_selection/` sits beside `data_preparation/` under the same project root.

`config.py` also drops these sparse fluid-sample metadata columns immediately after loading the snapshot dataframe, so older snapshot CSVs can be used without rebuilding the snapshot: `days_since_last_fluid_sample`, `fluid_sample_severity_max_365d`, and `fluid_sample_latest_smr`.

Supported input types:

- CSV
- Parquet
- Excel

## How to run step by step

Run from inside the `feature_selection` folder:

```bash
cd feature_selection
python 00_prepare_data.py
python 01_unsupervised_selection.py
python 02_correlation_analysis.py
python 03_statistical_tests.py
python 04_xgboost_importance.py
python 05_permutation_importance.py
python 06_shap_analysis.py
python 07_consensus_report.py
```

Each script runs without command-line arguments. Edit `config.py` to change input paths, split ratios, model settings, feature-group rules, and reporting options.

## Run everything at once

```bash
cd feature_selection
python main.py
```

`main.py` runs the same numbered steps in order and writes a combined summary to:

```text
output/00_all_steps_run_summary.json
```

## Step order and output folders

All outputs are saved under `feature_selection/output`, with one subfolder per step:

```text
output/
    00_prepare_data/
    01_unsupervised_selection/
    02_correlation_analysis/
    03_statistical_tests/
    04_xgboost_importance/
    05_permutation_importance/
    06_shap_analysis/
    07_consensus_report/
```

### 00_prepare_data.py

Creates the reusable data audit outputs and applies the first early feature-quality filter.

This step now removes raw source columns whose **feature_train missing rate is greater than 90%** before preprocessing and downstream methods. The threshold is controlled in `config.py`:

```python
DROP_FEATURES_WITH_HIGH_MISSINGNESS = True
HIGH_MISSINGNESS_THRESHOLD = 0.90
```

Main outputs include:

- chronological training / validation / test split summary
- inner `feature_train` / `feature_selection_holdout` split summary
- initial candidate feature list before early filters
- final candidate feature list after early filters
- raw missing-value counts and percentages before preprocessing
- dropped high-missingness feature report
- dropped zero-variance feature report
- preselection filter summary
- prepared feature mapping after imputation / one-hot encoding
- feature group assignment review

Output folder:

```text
output/00_prepare_data
```

### 01_unsupervised_selection.py

Creates source-level unsupervised diagnostics and applies the second early feature-quality filter.

This step removes raw source columns that are **constant / zero-variance in feature_train** before correlation, statistical tests, XGBoost, permutation, and SHAP are run. The behavior is controlled in `config.py`:

```python
DROP_ZERO_VARIANCE_FEATURES = True
```

Main outputs include:

- raw feature inventory after the high-missingness filter
- dropped zero-variance / constant raw feature report
- raw feature inventory after both early filters
- prepared feature diagnostics after both early filters
- constant prepared feature check after both early filters

Output folder:

```text
output/01_unsupervised_selection
```

### 02_correlation_analysis.py

Runs grouped correlation analysis within configured feature groups only.

This step starts from the smaller prepared feature set after the high-missingness and zero-variance filters have already been applied.

Output folder:

```text
output/02_correlation_analysis
```

Main outputs:

- `01_grouped_correlation_pairs_feature_train.csv`
- `02_grouped_correlation_summary_feature_train.csv`

### 03_statistical_tests.py

Runs supervised univariate statistical filters on `feature_train` only:

- ANOVA F-test
- mutual information
- chi-squared after min-max scaling

This step also uses the smaller feature set after the early quality filters.

Output folder:

```text
output/03_statistical_tests
```

### 04_xgboost_importance.py

Trains a temporary XGBoost model on `feature_train` and saves built-in feature importance:

- weight
- gain
- cover
- total gain
- total cover

It also saves threshold-free metrics on `feature_selection_holdout` for context.

Output folder:

```text
output/04_xgboost_importance
```

### 05_permutation_importance.py

Trains the temporary XGBoost model on `feature_train` and computes permutation importance on `feature_selection_holdout`.

Permutation importance uses F2 scoring through:

```python
PERMUTATION_SCORING = "f2"
```

Output folder:

```text
output/05_permutation_importance
```

### 06_shap_analysis.py

Trains the temporary XGBoost model on `feature_train` and computes SHAP importance on `feature_selection_holdout`.

Output folder:

```text
output/06_shap_analysis
```

### 07_consensus_report.py

Reads the outputs from earlier method folders and creates combined review artifacts:

- consensus ranking table
- combined Excel workbook
- markdown summary

Output folder:

```text
output/07_consensus_report
```

The consensus step does not apply a final importance-based keep/drop threshold. It combines available ranks for review only.

## Split design

The workflow uses the non-cross-validation design discussed:

```text
full dataset
    |-- training_main
    |     |-- feature_train
    |     |-- feature_selection_holdout
    |-- validation_holdout
    |-- test_holdout
```

`feature_train` is used for:

- preprocessing fit
- high-missingness feature screening
- zero-variance / constant feature screening
- grouped correlation
- ANOVA
- mutual information
- chi-squared
- temporary XGBoost model training
- XGBoost built-in feature importance

`feature_selection_holdout` is used for:

- permutation importance
- SHAP importance
- threshold-free temporary model metrics

`validation_holdout` and `test_holdout` are preserved for later model validation and final reporting. They are not used for feature-selection ranking in this project.

## Feature groups for correlation

Correlation analysis is performed within feature groups only to reduce pair volume and make the result easier to review. The grouping rules are defined in:

```python
config.FEATURE_GROUP_RULES
```

The current rules explicitly cover:

- machine context
- fault-code features
- maintenance features
- warranty/prior-claim features
- operation/utilization features
- SMR/usage-meter features
- fluid/oil lab-result features

Fluid/oil patterns are intentionally specific, for example `Fe_Iron_PPM`, `Fuel_Fuel_PERCENT`, `Soot_Soot_PERCENT`, and `Water_Water_PERCENT`. Broad patterns such as plain `fuel` or `oil` are avoided so operation and maintenance features are not accidentally mis-grouped.

## Important behavior

The workflow now applies two early source-level quality filters before downstream methods:

1. Drop features with more than 90% missing values in `feature_train`.
2. Drop features that are constant / zero-variance in `feature_train`.

The workflow still does **not** produce a final importance-based feature freeze. It saves ranked reports and diagnostics so you can manually review correlation, statistical tests, XGBoost, permutation, and SHAP evidence before choosing final thresholds or feature subsets.

Each step writes a `step_run_summary.json` file in its output folder.

## Main scripts

- `config.py`: editable run configuration.
- `workflow.py`: shared workflow implementation used by all step scripts.
- `00_prepare_data.py`: split, missingness, high-missingness filter, candidate feature, and feature-map reports.
- `01_unsupervised_selection.py`: raw inventory, zero-variance filter, prepared diagnostics, constant feature reports.
- `02_correlation_analysis.py`: grouped within-source correlation analysis.
- `03_statistical_tests.py`: ANOVA, mutual information, chi-squared.
- `04_xgboost_importance.py`: XGBoost built-in feature importance.
- `05_permutation_importance.py`: F2 permutation importance.
- `06_shap_analysis.py`: SHAP importance.
- `07_consensus_report.py`: combined consensus, Excel, and markdown reports.
- `main.py`: run all numbered steps in order.
- `fs_utils.py`: reusable helper functions with detailed docstrings and comments.
- `requirements.txt`: Python package requirements.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
