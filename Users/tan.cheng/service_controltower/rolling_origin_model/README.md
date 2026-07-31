# Rolling-Origin Machine Risk Model

This package supports two interchangeable outcome sources while preserving the
same feature engineering, rolling-origin split, model training, calibration, and
multi-anchor fleet-ranking logic.

Supported target sources:

- `physical_failure`: `Physical Failure Events.csv`
- `warranty`: `warranty.csv`

## Required local folder layout

Place the scripts and source data as sibling directories:

```text
project_root/
  enriched_data/
    fault_codes.csv
    fluid_samples.csv
    maintenance.csv
    operation.csv
    warranty.csv
    Physical Failure Events.csv
  rolling_origin_model/
    config.py
    run_all.py
    ...
```

The scripts resolve all source paths relative to their own location. There are
no hard-coded `/mnt/data` paths.

The configuration also recognizes the uploaded smoke-run names such as
`fault_codes(5).csv` and `warranty(9).csv`, but the preferred local names are the
simple names shown above.

## Select one target source

The default in `config.py` is:

```python
TARGET_SOURCE = "physical_failure"
```

The environment variable `RISK_TARGET_SOURCE` overrides the default. The easiest
commands are:

```bash
cd rolling_origin_model
python run_all.py --target-source physical_failure
python run_all.py --target-source warranty
```

To run both targets and automatically generate aligned comparison tables:

```bash
python run_all.py --target-source both
```

## What changes when the target is switched

Only the outcome event table and target-event history features change.

The following remain identical:

- Fleet roster construction
- Monthly snapshot dates
- 90-day feature lookback
- 90-day outcome horizon
- Condition features from faults, fluid samples, and maintenance
- Rolling-origin folds
- 90-day purge
- Three-month calibration blocks
- XGBoost parameters
- Probability calibration
- Top-K and Top-N definitions
- Multi-anchor fleet dates

The generic target columns are:

```text
target_event_0_90
future_target_event_count_90d
days_since_prior_target_event
prior_target_event_count_90d
prior_target_event_count_365d
prior_target_event_count_730d
```

For `physical_failure`, the history columns summarize prior physical-failure
events. For `warranty`, they summarize prior warranty claims.

`base27` is the cleanest target-only comparison because it excludes prior target
event history. `base27_plus_history` tests the operational model where the prior
event history source changes together with the label source.

## Same multi-anchor dates

Both target runs use these fixed dates from the previous experiment:

```text
2024H1: 2024-02-03, 2024-04-17, 2024-06-01
2024H2: 2024-08-03, 2024-10-24, 2024-12-10
2025H1: 2025-02-14, 2025-04-03, 2025-05-22
2025H2: 2025-07-19, 2025-10-27, 2025-11-28
2026Q1: 2026-01-14, 2026-02-22, 2026-04-28
```

They are configured in `MULTI_ANCHOR_FLEET_FIXED_DATES`.

The supplied original warranty extract has a maximum observed event date of
June 26, 2026. Therefore, the April 28, 2026 anchor may be right-censored for a
full 90-day warranty outcome. The script preserves this date for exact comparison
and writes `outcome_window_complete` in the anchor manifest and anchor metrics.
Set `MULTI_ANCHOR_ALLOW_INCOMPLETE_FIXED_DATES = False` for strict validation,
or replace the source with a warranty extract known to be complete through the
end of July 2026.

## Main scripts

```text
00_validate_configuration.py
00_prepare_operation_cache.py
01_build_snapshot_dataset.py
02_run_smoke_validation.py
03_train_final_xgboost.py
04_score_latest.py
10_multi_anchor_fleet_evaluation.py
11_compare_target_sources.py
run_all.py
```

## Output locations

Target-specific artifacts are isolated automatically:

```text
rolling_origin_model/outputs/physical_failure/
rolling_origin_model/outputs/warranty/
rolling_origin_model/models/physical_failure/
rolling_origin_model/models/warranty/
```

After running both targets, comparison tables are written to:

```text
rolling_origin_model/outputs/target_comparison/
```

Key comparison files include:

```text
target_variant_summary_wide.csv
target_fold_metrics_wide.csv
target_anchor_metrics_wide.csv
target_top_k_top_n_wide.csv
target_anchor_alignment_check.csv
```

## Run only the multi-anchor evaluation

After the monthly snapshot table has been built for the selected target:

```bash
RISK_TARGET_SOURCE=physical_failure python 10_multi_anchor_fleet_evaluation.py
RISK_TARGET_SOURCE=warranty python 10_multi_anchor_fleet_evaluation.py
```

On Windows PowerShell:

```powershell
$env:RISK_TARGET_SOURCE = "warranty"
python 10_multi_anchor_fleet_evaluation.py
```

## Validate paths before a full run

```bash
RISK_TARGET_SOURCE=physical_failure python 00_validate_configuration.py
RISK_TARGET_SOURCE=warranty python 00_validate_configuration.py
```
