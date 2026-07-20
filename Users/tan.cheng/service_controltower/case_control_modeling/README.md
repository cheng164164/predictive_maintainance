# Case-Control Claim Selection Modes

This update adds a simple switch for positive claim events in the window-based case-control workflow.

## Main config switch

Edit `config.py`:

```python
POSITIVE_CLAIM_SELECTION_MODE = "first"
```

Use only the first claim event for each machine.

```python
POSITIVE_CLAIM_SELECTION_MODE = "multiple"
```

Use the first claim event for each machine, plus later claim events when the later claim is at least `lead_max_days` after the immediately previous claim event for the same machine.

For example, if the window config is:

```python
{"name": "lead_120_to_30", "lead_max_days": 120, "lead_min_days": 30}
```

a later claim for the same machine is selected only if:

```text
current_claim_date - previous_claim_date >= 120 days
```

This keeps the logic simple and ensures every selected claim has a consistent monitoring window before the claim. The selection does not compare failure cause, failure component, critical part, or whether the old issue was fixed.

## Files changed

- `config.py`
  - Added `POSITIVE_CLAIM_SELECTION_MODE`.
- `cc_utils.py`
  - Added `select_positive_claims_for_window_config`.
  - Updated case-control row building so selected claims define positives, while the full claim history is still used for prior-claim features and control exclusion checks.
- `02_build_case_control_dataset.py`
  - Applies the claim-selection mode separately for each window config.
  - Saves claim-selection audit files.
- `main.py`
  - Keeps the validation prediction report step as step 05.

## New output files

For each dataset under `output/02_case_control_datasets/<dataset_id>/`:

- `positive_claim_selection_audit.csv`
  - Shows every claim event and whether it was selected as a positive case.
- `selected_positive_claim_events.csv`
  - Shows only claim events selected as positive cases for that window config.

Useful audit columns include:

- `selected_as_positive_claim`
- `claim_selection_reason`
- `claim_sequence_number`
- `machine_claim_event_count`
- `previous_claim_date_same_machine`
- `days_since_previous_claim_same_machine`
- `lead_max_days_threshold_for_repeat_claim`

## Recommended usage

Start with:

```python
POSITIVE_CLAIM_SELECTION_MODE = "first"
```

Then compare with:

```python
POSITIVE_CLAIM_SELECTION_MODE = "multiple"
```

Compare validation average precision, top-k precision, and case-control group ranking results from step 05.
