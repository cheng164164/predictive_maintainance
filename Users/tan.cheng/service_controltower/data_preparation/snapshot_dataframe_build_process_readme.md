# Snapshot Dataframe Build Process

## Purpose

This document explains, at a high level, how the unified snapshot dataframe is built from multiple source files.

The goal is to create a machine-time dataset that can be used for predictive maintenance modeling.

Each row in the final dataset represents:

- One machine or model_id
- At one snapshot date
- With historical information from different source systems summarized as features
- With a future warranty claim indicator as the target label

Example final row meaning:

```text
model_id = D71EX-24 70155
snapshot_date = 2023-06-01
```

This row represents what was known about machine `D71EX-24 70155` as of `2023-06-01`.

---

## High-Level Process

The build process follows this workflow:

1. Start with the machine snapshot backbone from `machine.csv`
2. Standardize each source file into a consistent format
3. Build one source snapshot dataframe for each source file
4. Validate that every source snapshot aligns to the same machine snapshot backbone
5. Join all source snapshots together using `model_id` and `snapshot_date`
6. Finalize the unified dataset using the feature freeze file
7. Save the final `snapshot_dataframe.csv` and supporting source snapshot files

---

## Input Files

The current workflow uses the following source files.

| Source file | Main purpose |
|---|---|
| `machine.csv` | Defines the snapshot backbone and the machines included in the dataset |
| `fault_codes.csv` | Provides fault and diagnostic event history |
| `maintenance.csv` | Provides maintenance and reset history |
| `operation.csv` | Provides machine usage and operation history |
| `fluid_samples.csv` | Provides lab sample and fluid analysis history |
| `warranty.csv` | Provides historical and future warranty claim information |
| `xgb_feature_freeze(all).csv` | Defines the expected final feature set for modeling |

---

## Key Concept: Snapshot Backbone

The snapshot backbone is the foundation of the dataset.

It defines the rows that should exist in the final dataset.

The backbone grain is:

```text
model_id + snapshot_date
```

Example backbone:

| model_id | snapshot_date |
|---|---|
| D71EX-24 70155 | 2023-05-01 |
| D71EX-24 70155 | 2023-05-15 |
| D71EX-24 70155 | 2023-06-01 |
| D61PX-24 12345 | 2023-05-01 |
| D61PX-24 12345 | 2023-05-15 |

Each source file is summarized onto this same backbone.

This means every source snapshot should have the same key columns:

```text
model_id, snapshot_date
```

---

## Key Concept: Leakage-Safe Feature Building

For each snapshot date, the features only use information that happened before the snapshot date.

Example:

```text
model_id = D71EX-24 70155
snapshot_date = 2023-06-01
```

When building features for this row:

- Fault features only use fault events before 2023-06-01
- Maintenance features only use maintenance events before 2023-06-01
- Operation features only use operation records before 2023-06-01
- Fluid sample features only use samples before 2023-06-01
- Prior warranty features only use claims before 2023-06-01

This prevents the model from using future information when making a prediction.

---

## Step 1: Load the Machine Snapshot Backbone

The process starts from `machine.csv`.

This file defines the machines and snapshot dates that will be included in the final dataset.

The machine backbone is saved as a separate source snapshot file:

```text
source_snapshots/machine_backbone.csv
```

This file acts as the row template for all source snapshots.

---

## Step 2: Standardize Each Source File

Each source file may use different column names, date fields, or formats.

Before feature building, each source is standardized so the script can process it consistently.

Examples of standardization:

- Convert source machine identifier to `model_id`
- Convert event dates to a standard date format
- Keep only records that can be matched to the snapshot backbone
- Convert numeric fields such as hours, counts, claim amounts, and lab values
- Clean categorical fields such as fault code, component, maintenance type, or claim type

The goal of this step is to make each source ready for snapshot aggregation.

---

## Step 3: Build Source Snapshot Dataframes

Each source file is summarized into its own snapshot dataframe.

All source snapshot files use the same grain:

```text
model_id + snapshot_date
```

### Fault Snapshot

Input:

```text
fault_codes.csv
```

Output:

```text
source_snapshots/fault_snapshot.csv
```

Example features:

- `fault_count_90d`
- `fault_count_30d`
- `days_since_last_fault`
- `unique_fault_code_count_90d`
- `mechanical_fault_count_90d`
- `electrical_fault_count_90d`

Example interpretation:

For a snapshot date of `2023-06-01`, `fault_count_90d` counts fault events for the same `model_id` during the prior 90 days only.

### Maintenance Snapshot

Input:

```text
maintenance.csv
```

Output:

```text
source_snapshots/maintenance_snapshot.csv
```

Example features:

- `maintenance_events_180d`
- `monitor_reset_count_180d`
- `days_since_last_reset`
- `overdue_item_count`
- `due_now_item_count`

Example interpretation:

For a snapshot date of `2023-06-01`, `maintenance_events_180d` counts maintenance records for the same `model_id` during the prior 180 days.

### Operation Snapshot

Input:

```text
operation.csv
```

Output:

```text
source_snapshots/operation_snapshot.csv
```

Example features:

- Recent machine usage metrics
- Recent operation hour summaries
- Days since last operation record
- Usage activity indicators

Example interpretation:

For each snapshot date, operation features summarize how recently and how heavily the machine was used before that date.

### Fluid Sample Snapshot

Input:

```text
fluid_samples.csv
```

Output:

```text
source_snapshots/fluid_sample_snapshot.csv
```

Example features:

- `Fe_Iron_PPM`
- `Al_Aluminum_PPM`
- `Fuel_Fuel_PERCENT`
- `Soot_Soot_PERCENT`
- `Water_Water_PERCENT`
- `days_since_last_fluid_sample`

Example interpretation:

For a snapshot date of `2023-06-01`, the fluid sample snapshot uses the latest available sample before that date, within the configured lookback period.

### Warranty Snapshot

Input:

```text
warranty.csv
```

Output:

```text
source_snapshots/warranty_target_snapshot.csv
```

Example features and target:

- `claim_next_45d`
- `prior_claim_count_365d`
- `prior_claim_count_180d`
- `days_since_last_claim`
- `prior_claim_amount_sum_365d`

Example interpretation:

For a snapshot date of `2023-06-01`:

- Prior warranty features use claims before `2023-06-01`
- The target `claim_next_45d` checks whether a claim happens after `2023-06-01` and within the next 45 days

---

## Step 4: Validate Source Snapshot Alignment

Before joining the source snapshots, the process checks that each source snapshot aligns with the machine backbone.

The key validation checks are:

- Each source snapshot has `model_id` and `snapshot_date`
- Each source snapshot has one row per `model_id + snapshot_date`
- Each source snapshot matches the row count of the machine backbone
- Joins will not accidentally duplicate rows

Example expected alignment:

| File | Expected grain |
|---|---|
| `machine_backbone.csv` | one row per `model_id + snapshot_date` |
| `fault_snapshot.csv` | one row per `model_id + snapshot_date` |
| `maintenance_snapshot.csv` | one row per `model_id + snapshot_date` |
| `operation_snapshot.csv` | one row per `model_id + snapshot_date` |
| `fluid_sample_snapshot.csv` | one row per `model_id + snapshot_date` |
| `warranty_target_snapshot.csv` | one row per `model_id + snapshot_date` |

This validation step helps protect the final dataset from unexpected row duplication or row loss.

---

## Step 5: Join Source Snapshots into the Unified Dataset

The final unified dataset starts with the machine backbone.

Then each source snapshot is joined using the same keys:

```text
model_id, snapshot_date
```

Conceptual join process:

```text
machine_backbone
join fault_snapshot
join maintenance_snapshot
join operation_snapshot
join fluid_sample_snapshot
join warranty_target_snapshot
```

The final output is:

```text
snapshot_dataframe.csv
```

The final dataset includes:

- Machine and snapshot identifiers
- Fault features
- Maintenance features
- Operation features
- Fluid sample features
- Prior warranty features
- Future warranty target label

---

## Step 6: Finalize the Dataset

After the source snapshots are joined, the script finalizes the dataset.

This includes:

- Checking that expected features from `xgb_feature_freeze(all).csv` are present
- Filling missing count features with 0
- Filling missing ratio or rate features with 0
- Filling missing recency features with a large value such as 9999
- Keeping the target column `claim_next_45d`
- Saving the final CSV output

The feature freeze file helps keep the modeling dataset consistent across runs.

---

## Final Output Files

The process produces the final unified dataset and supporting source snapshot files.

Main output:

```text
snapshot_dataframe.csv
```

Supporting outputs:

```text
source_snapshots/machine_backbone.csv
source_snapshots/fault_snapshot.csv
source_snapshots/maintenance_snapshot.csv
source_snapshots/operation_snapshot.csv
source_snapshots/fluid_sample_snapshot.csv
source_snapshots/warranty_target_snapshot.csv
```
---

## How to Read the Final Dataset

Each row should be interpreted as:

```text
What did we know about this machine as of this snapshot date, and did a warranty claim occur in the future prediction window?
```

Example:

```text
model_id = D71EX-24 70155
snapshot_date = 2023-06-01
claim_next_45d = 1
```

This means the machine had a warranty claim within 45 days after `2023-06-01`.

The feature columns describe what was known before `2023-06-01`.

---

