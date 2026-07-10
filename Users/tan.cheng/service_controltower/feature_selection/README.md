# Feature Selection Analysis

This folder contains a config-driven, step-by-step feature-selection analysis workflow for the unified snapshot dataframe.

The workflow is intentionally split into numbered scripts so you can run one method at a time, inspect its output folder, then continue to the next step. `main.py` is still available when you want to run every step in order.

## Default input

By default, `config.py` expects the production unified snapshot dataframe here:

```text
../data_preparation/output/snapshot_dataframe.csv
```

This matches the project layout where `feature_selection/` sits beside `data_preparation/` under the same project root.

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

Creates the reusable data audit outputs:

- chronological training / validation / test split summary
- inner `feature_train` / `feature_selection_holdout` split summary
- candidate feature list
- raw missing-value counts and percentages before preprocessing
- prepared feature mapping after imputation / one-hot encoding
- feature group assignment review

Output folder:

```text
output/00_prepare_data
```

### 01_unsupervised_selection.py

Creates unsupervised diagnostics without applying any keep/drop threshold:

- raw feature inventory on `feature_train`
- prepared feature diagnostics after preprocessing
- exact constant prepared feature report

Output folder:

```text
output/01_unsupervised_selection
```

### 02_correlation_analysis.py

Runs grouped correlation analysis within configured feature groups only.

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

The consensus step does not apply a final keep/drop threshold. It combines available ranks for review only.

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
- unsupervised diagnostics
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

The workflow does **not** produce a final keep/drop feature freeze. It saves ranked reports and diagnostics so you can manually review the evidence before choosing thresholds or final feature subsets.

Each step writes a `step_run_summary.json` file in its output folder.

## Main scripts

- `config.py`: editable run configuration.
- `workflow.py`: shared workflow implementation used by all step scripts.
- `00_prepare_data.py`: split, missingness, candidate feature, and feature-map reports.
- `01_unsupervised_selection.py`: raw inventory, prepared diagnostics, constant feature report.
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
