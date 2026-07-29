# Case-Control Predictive Maintenance Modeling

This project builds machine-level case-control datasets for predictive maintenance. It supports:

- Fast `base` features and the historical `frozen` feature set.
- Fixed machine-level train, validation, and test partitions.
- Controlled, random, or mixed training negatives.
- Controlled, random, or deployment-like shared-date validation/test cohorts.
- Nested validation/test negative-to-positive ratios such as 1:1 through 5:1.
- Bootstrap confidence intervals for ranking metrics.
- Percentage-based Top-K and fixed-count Top-N operational metrics.
- Fair comparison of multiple feature-window definitions on common holdout machines.
- Actual next-claim date and days-to-claim in validation/test prediction outputs.
- Separate ranking reports for every configured claim horizon.

There is no component-level feature mode and no 4,000-hour SMR exclusion rule.

## Main workflow

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build source profiles, claim episodes, and datasets:

```bash
python 00_profile_sources.py
python 01_build_claim_episodes.py
python 02_build_case_control_dataset.py
```

Evaluate sensitivity to validation/test prevalence:

```bash
python 08_holdout_ratio_sensitivity.py
```

When `WINDOW_CONFIGS` contains more than one feature window, compare the models on the same labeled machines:

```bash
python 09_compare_window_designs.py
```

The standard development workflow remains:

```bash
python main.py
```

`main.py` runs Steps 00 through 04 and does not score the test set. Steps 08 and 09 are intentionally separate because they include test-set sensitivity analysis.

## Configuration switches

### Feature set

```python
FEATURE_SET = "base"  # "base" or "frozen"
```

- `base`: compact and explainable features calculated with the fast vectorized window logic.
- `frozen`: the historical frozen snapshot feature definition.

### Feature windows

One window:

```python
WINDOW_CONFIGS = [
    {"lead_max_days": 90, "lead_min_days": 30},
]
```

Multiple windows for a comparison experiment:

```python
WINDOW_CONFIGS = [
    {"lead_max_days": 90, "lead_min_days": 30},
    {"lead_max_days": 90, "lead_min_days": 7},
    {"lead_max_days": 90, "lead_min_days": 1},
]
```

For a claim on June 30:

- `90 to 30` uses events from April 1 through May 31 and preserves a 30-day lead period.
- `90 to 7` uses events from April 1 through June 22 and preserves a 7-day lead period.
- `90 to 1` uses events from April 1 through June 28 and preserves a 1-day lead period.

Feature calculations use records in the half-open interval:

```text
window_start <= event_date < window_end
```

Events on `window_end` are excluded.

When several window configurations are run together, Step 02 uses one shared deterministic holdout-machine ranking. This aligns positive claim selection and negative-machine ordering across windows as closely as eligibility permits. A normal single-window run retains the historical window-specific hashing logic, so the default 90-to-30 holdout remains backward compatible.

Step 09 performs the strict comparison by intersecting machine and label identities across all window designs before recomputing metrics.

### Training negative design

```python
NEGATIVE_SAMPLING_MODE = "mixed"  # "controlled", "random", or "mixed"
NEGATIVES_PER_POSITIVE_CASE = 3
```

These settings affect training only.

- `controlled`: another machine of the same full model uses the exact positive calendar window.
- `random`: another machine of the same full model uses an independently selected eligible no-claim window.
- `mixed`: combines controlled and random negatives. When the requested number is odd, controlled receives one additional slot before fallback filling.

### Validation/test negative design

```python
HOLDOUT_NEGATIVE_SAMPLING_MODE = "random"
# Choices: "random", "controlled", or "as_of_anchor"
```

This is independent of the training negative switch.

#### Random holdout

- Each negative machine receives one independently sampled eligible no-claim window.
- Negatives are not matched to positives by calendar date or full model.
- The design approximates scoring the broader machine population.

#### Controlled holdout

- Each positive claim window is matched to negative machines from the same full model.
- Controls use the exact same calendar window as the positive.
- Every retained positive must have enough unique controls for the largest configured ratio.
- This design reduces calendar-time and model-mix confounding, but it represents a matched comparison rather than natural field prevalence.

#### As-of-anchor holdout

```python
HOLDOUT_NEGATIVE_SAMPLING_MODE = "as_of_anchor"
HOLDOUT_AS_OF_MIN_POSITIVE_MACHINES = 10
```

- Step 02 searches eligible dates in a deterministic random order and chooses one shared anchor date with positive and negative machines in both validation and test.
- Every machine uses the same `window_end` and feature-window length. For a 90-to-30 design, all features use the 60-day interval ending at the anchor.
- A design positive has a claim within `lead_min_days` after the anchor. A design negative has no claim in that same interval.
- Negatives are not restricted to the same `full_model`; they represent the eligible fleet at that calendar snapshot.
- The nested ratio cohorts are controlled by `HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS`.
- This is the most deployment-like option for asking whether the top-ranked machines experience claims soon after one scoring date.

Every holdout machine contributes at most one row. Validation and test use different physical machines, while the as-of design uses the same anchor date for both splits.

### Validation/test prevalence ratios

```python
HOLDOUT_NEGATIVE_TO_POSITIVE_RATIOS = [1, 2, 3, 4, 5]
HOLDOUT_RANDOM_STATE = 42
```

Step 02 creates one master pool for the largest ratio and then materializes nested subsets.

For 60 positive machines:

- 1:1 contains 60 positives and 60 negatives.
- 2:1 contains the same 60 positives and the first 120 negatives.
- 5:1 contains the same 60 positives and the first 300 negatives.

The model is fitted once per dataset design in Step 08. The same fitted model scores every ratio cohort, so changes across ratios come from the evaluation population rather than refitting.

### Bootstrap confidence intervals

```python
HOLDOUT_BOOTSTRAP_ENABLED = True
HOLDOUT_BOOTSTRAP_N_RESAMPLES = 1000
HOLDOUT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
HOLDOUT_BOOTSTRAP_RANDOM_STATE = 20260727
```

Step 08 and Step 09 use a stratified machine-level nonparametric bootstrap:

- Positive and negative machines are resampled separately with replacement.
- Each replicate preserves the evaluated negative-to-positive ratio.
- Percentile confidence intervals are reported.
- The bootstrap unit is the physical machine because Step 02 creates one holdout row per machine.

Confidence intervals are calculated for:

- ROC AUC.
- Average precision.
- Precision, recall, and lift at each percentage Top-K cutoff.
- Precision, recall, and lift at each fixed Top-N count.

### Percentage Top-K and fixed Top-N

```python
VALIDATION_TOP_K_RATES = [0.01, 0.05, 0.10, 0.20]
HOLDOUT_TOP_N_COUNTS = [10, 20, 50]
```

Percentage Top-K answers questions such as:

> What happens when the program flags the highest-scored 5 percent of machines?

Fixed Top-N answers questions such as:

> What happens when the maintenance team can inspect exactly 20 machines?

Fixed Top-N is the preferred diagnostic for extreme-top degradation across prevalence ratios because the operational workload remains constant. In contrast, Top 5 percent selects more machines when the cohort becomes larger.

## Fixed machine split

```python
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
FIXED_SPLIT_RANDOM_STATE = 42
```

Machine assignments are deterministic and stratified by full model and eligible claim-history status. A physical machine appears in exactly one split.

The assignment asset is stored under:

```text
output/fixed_split_assets/fixed_machine_split_assignments.csv
```

Validation/test row locks are stored by window and negative design, for example:

```text
output/fixed_split_assets/lead_90_to_30/
    fixed_random_validation_test_master_base_rows.csv
    fixed_random_validation_test_metadata.json
    fixed_random_validation_test_sampling_audit.csv
    fixed_random_validation_test_pool_summary.csv
```

or:

```text
output/fixed_split_assets/lead_90_to_30/
    fixed_controlled_validation_test_master_base_rows.csv
    fixed_controlled_validation_test_metadata.json
    fixed_controlled_validation_test_sampling_audit.csv
    fixed_controlled_validation_test_pool_summary.csv
```

or, for the shared-date deployment design:

```text
output/fixed_split_assets/lead_90_to_30/
    fixed_as_of_anchor_validation_test_master_base_rows.csv
    fixed_as_of_anchor_validation_test_metadata.json
    fixed_as_of_anchor_validation_test_sampling_audit.csv
    fixed_as_of_anchor_validation_test_pool_summary.csv
```

Changing model parameters, feature mode, or training-negative settings does not change a compatible locked holdout. Changing a holdout-defining setting or source population causes a fingerprint mismatch rather than silently replacing the holdout.

## Evaluation target

```python
EVALUATION_TARGET_MODE = "training_target"
```

Use `training_target` for the case-control prevalence-ratio experiment.

The separate future-claim option remains available:

```python
EVALUATION_TARGET_MODE = "claim_within_horizon"
EVALUATION_CLAIM_HORIZON_DAYS = [30, 60, 90, 120, 180, 365]
```

In horizon mode, labels are derived dynamically from the number of days to the next claim rather than trusted from stale saved columns. Step 04, Step 07, and Step 08 report every configured horizon separately.

- With `random` or `controlled`, the standard validation/final-test workflow uses the separate outcome-independent fixed horizon cohort.
- With `as_of_anchor`, the same anchored fleet snapshot is reused for every horizon. Machines, feature windows, and scores remain fixed while the true label changes with the horizon.

For example, a next claim 75 days after `window_end` is negative at 30 and 60 days, but positive at 90, 120, 180, and 365 days.

Prediction files retain these review fields when available:

```text
next_claim_date_on_or_after_window_end
days_to_next_claim_on_or_after_window_end
has_future_claim_on_or_after_window_end
future_claim_lead_time_bucket
as_of_anchor_date
as_of_actual_next_claim_date
as_of_days_to_next_claim
```

## Step 02 outputs

For each dataset design, Step 02 writes mode-specific and generic files:

```text
holdout_master_dataset.csv
holdout_ratio_index.csv
random_holdout_master_dataset.csv                 # random mode alias
random_holdout_ratio_index.csv                    # random mode alias
controlled_holdout_master_dataset.csv             # controlled mode
controlled_holdout_ratio_index.csv                # controlled mode
as_of_anchor_holdout_master_dataset.csv              # shared-date mode
as_of_anchor_holdout_ratio_index.csv                  # shared-date mode
as_of_anchor_horizon_label_summary.csv               # positives by horizon
case_control_validation_ratio_1_to_1.csv
...
case_control_validation_ratio_5_to_1.csv
case_control_test_ratio_1_to_1.csv
...
case_control_test_ratio_5_to_1.csv
```

`holdout_ratio_index.csv` is the authoritative map from split and requested ratio to dataset path and row counts. For backward compatibility, `validation_dataset_path` and `test_dataset_path` in `dataset_index.csv` point to the largest configured ratio.

## Step 08 outputs

```text
output/08_holdout_ratio_sensitivity/
    holdout_ratio_metrics_all_datasets.csv
    holdout_ratio_top_k_all_datasets.csv
    holdout_ratio_fixed_top_n_all_datasets.csv
    holdout_ratio_sensitivity_for_review.csv
    validation_percentage_top_k_with_confidence_intervals.csv
    test_percentage_top_k_with_confidence_intervals.csv
    validation_fixed_top_n_with_confidence_intervals.csv
    test_fixed_top_n_with_confidence_intervals.csv
    *__horizon_30d__machine_predictions.csv
    *__horizon_60d__machine_predictions.csv
    ... one prediction file per configured horizon
    *__ap_roc_ratio.png
    *__fixed_top_n_precision_at_n.png
    *__fixed_top_n_recall_at_n.png
    run_summary.json
```

When `EVALUATION_TARGET_MODE = "claim_within_horizon"`, Step 08 fits and scores once per dataset/ratio, then reuses those scores for every horizon. This isolates label-horizon effects from model-refitting effects.

Important interpretation notes:

- The as-of ratio is exact for the design horizon (`lead_min_days`). At longer horizons, some design negatives may become positives, so Step 08 reports both the requested design ratio and the actual horizon-specific positive rate.
- Average precision depends on class prevalence.
- ROC AUC is less directly prevalence-dependent but can move when harder negatives are added.
- Raw percentage Top-K precision can decline because both prevalence and the absolute flagged count change.
- Fixed Top-N precision and recall directly test whether new negatives displace positives at the extreme top.
- Lift should be interpreted alongside raw precision because it normalizes enrichment against the cohort positive rate.

## Step 09 outputs

When multiple windows are present, run Step 09 after Step 08. It writes:

```text
output/09_window_design_comparison/
    common_machine_overlap_audit.csv
    common_machine_window_metrics.csv
    common_machine_percentage_top_k.csv
    common_machine_fixed_top_n.csv
    common_machine_window_comparison_for_review.csv
    run_summary.json
```

Step 09 groups predictions by algorithm, split, requested ratio, and evaluation target. It then:

1. Intersects machine IDs across all window designs.
2. Removes any machine whose evaluation label differs across windows.
3. Recomputes AP, ROC AUC, percentage Top-K, fixed Top-N, and bootstrap intervals on the identical labeled cohort.

This is the preferred table for selecting among 90-to-30, 90-to-7, and 90-to-1 designs.

## Recommended model-selection practice

- Use validation results to select a feature window and model design.
- Use the test set only as a limited final confirmation.
- Do not choose a window by repeatedly optimizing test metrics.
- When operational capacity is fixed, prioritize Precision@Top-N and Recall@Top-N.
- When broad ranking quality matters, evaluate AP and ROC AUC together.
- Treat overlapping bootstrap intervals as evidence that apparent differences may be sampling noise.

## Source layout

Preferred layout:

```text
project_root/
    enriched_data/
        warranty.csv
        fault_codes.csv
        fluid_samples.csv
        maintenance.csv
        operation.csv
    case_control_modeling/
        config.py
        cc_utils.py
        feature_definitions.py
        feature_engineering.py
        00_profile_sources.py
        01_build_claim_episodes.py
        02_build_case_control_dataset.py
        03_cross_validation.py
        04_fit_validate_model_report.py
        05_run_design_sweep.py
        06_tune_xgboost_hyperparameters.py
        07_final_test_evaluation.py
        08_holdout_ratio_sensitivity.py
        09_compare_window_designs.py
        main.py
```

The smoke-test environment also recognizes uploaded filenames such as `warranty(5).csv` and `operation_partial(3).csv` under `/mnt/data`.

## Tests

Run:

```bash
python -m unittest discover -s tests -v
```

The regression suite covers:

- Deterministic machine-level split assignments.
- No train/validation/test machine leakage.
- Controlled, random, and mixed training negatives.
- Random and controlled validation/test holdouts.
- Exact nested 1:1 through 5:1 ratios.
- One row per holdout machine.
- Fixed holdout reuse.
- Backward compatibility of the single-window random design.
- Shared machine ranking for multi-window experiments.
- Percentage Top-K and fixed Top-N calculations.
- Deterministic bootstrap confidence intervals.
- Base and frozen feature behavior.
- Dynamic future-claim horizon labels.
- Removal of the 4,000-hour SMR filter.
