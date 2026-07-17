# Window-Based Case-Control Modeling

This folder contains a concept-validation workflow for predictive-maintenance modeling using warranty claims as positive events and matched non-claim machines as controls.

## How to run

From this folder:

```bash
python main.py
```

No command-line arguments are required. All settings are controlled in `config.py`.

All outputs are written to:

```text
case_control_modeling/output/
```

## Workflow steps

```text
00_profile_sources.py
01_build_claim_episodes.py
02_build_case_control_dataset.py
03_smoke_run.py
04_cross_validation.py
main.py
```

## Main modeling idea

Each positive row is one claim episode with an observation window before the claim.

Example configuration:

```python
{"name": "lead_120_to_30", "lead_max_days": 120, "lead_min_days": 30}
```

This means:

```text
window_start = claim_date - 120 days
window_end   = claim_date - 30 days
```

The model only uses source events inside the observation window. Events after `window_end` are excluded so the business keeps at least 30 days of action lead time.

Controls use the same calendar observation window as the positive case, but they must not have a warranty claim in the configured future exclusion horizon.

## Easy tuning

Important parameters in `config.py`:

```python
WINDOW_CONFIGS = [
    {"name": "lead_120_to_30", "lead_max_days": 120, "lead_min_days": 30},
    {"name": "lead_180_to_45", "lead_max_days": 180, "lead_min_days": 45},
]

CONTROLS_PER_POSITIVE_CASE = 3
CONTROL_NO_CLAIM_DAYS_AFTER_WINDOW_END = 180
MAX_POSITIVE_CASES_PER_WINDOW = 1000
```

Change these values and rerun `python main.py`. The dataset builder will create a separate output dataset ID for each window/control setting.


## Train / validation / test split

Step `02_build_case_control_dataset.py` now creates a chronological train / validation / test split at the `case_control_group_id` level. This keeps each positive claim case and its sampled controls together in the same split.

Default split settings in `config.py`:

```python
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_DATE_COL = "window_end"
```

Generated files for each dataset include:

```text
case_control_dataset_with_split.csv
case_control_training_dataset.csv
case_control_validation_dataset.csv
case_control_test_dataset.csv
split_summary.csv
case_control_group_split_assignments.csv
```

Step `03_smoke_run.py` trains on the train split and evaluates on the validation split. Step `04_cross_validation.py` runs GroupKFold only within the train split. The test split is intentionally untouched and should be used only after model/window/feature choices are locked.

Dataset IDs were also simplified. If the window name is already `lead_120_to_30`, the file name will use it only once, for example:

```text
lead_120_to_30__controls_3__neg_180__components_on
```

## Base features

The base feature set is intentionally compact and explainable. It includes:

- prior warranty claim history before the window
- source coverage flags
- fault count, unique fault code count, L03/L04 counts, max action level, max evidence score
- fluid sample count, severity, and selected lab measurements
- maintenance event count, reset count, due/overdue counts, remaining hours
- operation usage, working hours, idle hours, SMR delta, high-throttle days

## Optional component-level features

This version adds optional granular component features from fault-code and maintenance sources.

Control this in `config.py`:

```python
ENABLE_COMPONENT_FEATURES = True
```

When enabled, the dataset builder adds features such as:

```text
fault_component_engine_count_window
fault_component_engine_l03plus_count_window
fault_component_engine_max_action_level_window
fault_component_engine_max_evidence_score_window
maintenance_component_engine_count_window
maintenance_component_engine_overdue_count_window
maintenance_component_engine_due_now_count_window
maintenance_component_engine_monitor_reset_count_window
maintenance_component_engine_min_remaining_hours_window
```

The same structure is generated for configured groups such as:

- engine
- hydraulic
- power_train
- work_equipment
- urea_scr
- cooling
- control
- transmission
- final_drive

The keyword mapping is controlled by:

```python
COMPONENT_FEATURE_GROUPS = {...}
```

Set `ENABLE_COMPONENT_FEATURES = False` to run only the original compact base feature set.

## XGBoost class importance

This version also adds configurable XGBoost class importance using `scale_pos_weight`.

Control this in `config.py`:

```python
XGBOOST_CLASS_IMPORTANCE_MODE = "auto"  # allowed: "auto", "fixed", "none"
XGBOOST_FIXED_SCALE_POS_WEIGHT = 1.0
```

- `auto`: compute `negative rows / positive rows` separately in each smoke/CV training split.
- `fixed`: use `XGBOOST_FIXED_SCALE_POS_WEIGHT`.
- `none`: do not set `scale_pos_weight`.

The resolved value is written into smoke/CV metric outputs as:

```text
fit_xgboost_class_importance_mode
fit_xgboost_scale_pos_weight
```

## Model comparison

Configured models:

```python
MODELS_TO_RUN = [
    "linear_regression",
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "xgboost",
]
```

`linear_regression` is only a simple ranking baseline. For binary classification, use logistic regression, random forest, or XGBoost as the main candidates.

## Important interpretation note

Case-control validation is for concept validation. Because controls are sampled, the positive rate is artificial. Do not interpret scores as real fleet probabilities.

Use the results to answer:

- Can the model separate pre-claim evidence windows from matched non-claim windows?
- Which lead window performs better?
- Which source/component signals are useful?
- Which model ranks positive cases higher?

Before production deployment, run a full-fleet historical backtest where every active machine is scored on historical scoring dates and evaluated against future claims.
