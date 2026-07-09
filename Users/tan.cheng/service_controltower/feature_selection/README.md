# Feature Selection Analysis

This folder contains a config-driven feature-selection analysis workflow for the unified snapshot dataframe.

The scripts now include detailed function-level docstrings and comments so the workflow is easier to review, maintain, and modify.

## How to run

```bash
cd feature_selection
python main.py
```

No command-line arguments are required. Edit `config.py` to change the input dataset path, target/date columns, split ratios, model parameters, feature-group rules, and reporting options.

## Default input

By default, `config.py` expects the input file here:

```text
../snapshot_dataframe_mini.csv
```

For the production unified snapshot dataframe, update this value in `config.py`:

```python
INPUT_DATA_PATH = PROJECT_DIR.parent / "your_unified_snapshot.csv"
```

CSV, Parquet, and Excel inputs are supported.

## Split design

The script uses the non-cross-validation workflow discussed. The configured full split uses the requested `0.75/0.15/0.15` values as ratio weights. Because those values sum to `1.05`, the script normalizes them internally and records the effective shares in `00_run_summary.json`.

```text
full dataset
    |-- training_main: ratio weight 0.75
    |     |-- feature_train: 75% of training_main
    |     |-- feature_selection_holdout: 25% of training_main
    |-- validation_holdout: ratio weight 0.15
    |-- test_holdout: ratio weight 0.15
```

`feature_train` is used for training-only diagnostics and rankings:

- raw feature inventory
- raw missing-value counts and percentages before preprocessing
- prepared feature diagnostics after preprocessing
- exact constant feature report
- grouped correlation pairs within configured feature groups
- ANOVA F-test
- mutual information
- chi-squared on min-max scaled features
- XGBoost feature importance

`feature_selection_holdout` is used for model-based review methods that benefit from holdout data:

- permutation importance using F2 scoring
- SHAP importance
- threshold-free XGBoost holdout metrics

`validation_holdout` and `test_holdout` are not used for feature-ranking reports.

## Important behavior

The script does **not** produce a final keep/drop feature freeze. It saves ranked reports and diagnostics so you can review the evidence manually before choosing thresholds or final feature subsets.

Correlation analysis is performed **within feature groups only** to reduce pair volume and keep the report easier to review. The grouping rules are defined in `config.FEATURE_GROUP_RULES`. There is no `MAX_CORRELATION_PAIRS_TO_SAVE` setting anymore; all within-group correlation pairs are saved to the CSV report.

Permutation importance uses F2 scoring through `config.PERMUTATION_SCORING = "f2"`. F2 emphasizes recall more than precision. In this script, sklearn permutation importance uses the model's default `predict()` behavior for the F2 scorer.

## Main outputs

All outputs are saved under:

```text
feature_selection/output
```

Important files include:

- `feature_selection_report.xlsx`: combined review workbook
- `feature_selection_summary.md`: human-readable run summary
- `00_run_summary.json`: machine-readable run summary
- `01_split_summary.csv`: row/date/target summary by split
- `01_split_assignments.csv`: outer chronological split audit table
- `01_inner_training_split_assignments.csv`: inner training split audit table
- `02_candidate_features.csv`: raw candidate feature list
- `03_raw_missing_values_before_preprocessing.csv`: missing counts and percentages before imputation/preprocessing
- `04_raw_feature_inventory_feature_train.csv`: raw feature diagnostics on `feature_train`
- `05_prepared_feature_mapping.csv`: prepared-to-raw feature mapping with assigned feature group
- `05_prepared_feature_groups_for_correlation.csv`: feature groups used for grouped correlation analysis
- `06_prepared_feature_diagnostics_feature_train.csv`: post-encoding/imputation diagnostics
- `06_constant_features_exact_feature_train.csv`: exact constant feature report
- `07_grouped_correlation_pairs_feature_train.csv`: sorted within-group correlation pairs
- `08_anova_f_classif_feature_train.csv`: ANOVA ranking
- `09_mutual_info_classif_feature_train.csv`: mutual information ranking
- `10_chi2_minmax_scaled_feature_train.csv`: chi-squared ranking
- `11_xgboost_importance_feature_train.csv`: XGBoost importance ranking
- `11_xgboost_threshold_free_metrics_feature_selection_holdout.json`: holdout average precision / ROC-AUC / log-loss where valid
- `12_permutation_importance_f2_feature_selection_holdout.csv`: permutation importance ranking using F2 scoring
- `13_shap_importance_feature_selection_holdout.csv`: SHAP ranking
- `14_consensus_rank_review_no_threshold.csv`: combined review table across methods
- `plots/`: top-N bar charts for quick review

## Main scripts

- `config.py`: all editable run configuration.
- `main.py`: end-to-end workflow orchestration.
- `fs_utils.py`: reusable helper functions with detailed docstrings and comments.
- `requirements.txt`: Python package requirements.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
