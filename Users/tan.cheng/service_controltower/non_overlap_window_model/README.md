# Exact 90-Day Future-Event Risk Model

This package builds one model at a time. The target source is controlled only in `config.py`:

```python
TARGET_SOURCE = "physical_failure"
```

or:

```python
TARGET_SOURCE = "warranty"
```

No comparison script or command-line argument is required.

## Local folder layout

```text
project_root/
  enriched_data/
    fault_codes.csv
    fluid_samples.csv
    maintenance.csv
    operation.csv
    warranty.csv
    Physical Failure Events.csv

  future_claim_predictive_model_90d/
    config.py
    run_all_steps.py
    01_validate_inputs.py
    ...
    14_anchor_fleet_validation.py
```

The source directory and script directory must be siblings. Suffixed exports such as `fault_codes(5).csv` are accepted when the match is unambiguous.

## Run the native 90-day model

Set all controls in `config.py`, then run:

```bash
python run_all_steps.py
```

The default executes Steps 01 through 13. Outputs are target-specific:

```text
outputs/physical_failure/
outputs/warranty/
artifacts/physical_failure/
artifacts/warranty/
```

Changing `TARGET_SOURCE` therefore does not overwrite the other target run.

## Run anchor fleet validation

Run this separately after Step 03, or after the full native pipeline:

```bash
python 14_anchor_fleet_validation.py
```

For every configured anchor date it scores the complete fleet and writes:

- ROC AUC, average precision, threshold precision/recall/F1/F2
- Top 1%, 5%, and 10% precision, recall, and lift
- fixed Top 5, 10, and 20 precision, recall, and lift
- one full fleet file sorted by risk score for each anchor
- true label, prediction label, TP/FP/FN/TN outcome
- raw score, calibrated probability, risk index, rank
- future event count, first event date, next event date, and days to event
- all model feature values for engineering audit

Anchor controls, dates, Top-K rates, and Top-N counts are all in `config.py`.

## Reproduce the prior experiment

The defaults retain:

- 90-day feature lookback
- 90-day future target horizon
- 90-day non-overlapping training segments
- XGBoost primary model
- reviewed 140-feature history-enhanced variant
- the same native split cutoff (`2026-06-26`)
- the same 15 fleet anchor dates
- the same random seed and model parameters

Set `RUN_CANDIDATE_SCREEN = True` to fit all configured candidate algorithms. The default `False` is the faster XGBoost smoke-run mode.
