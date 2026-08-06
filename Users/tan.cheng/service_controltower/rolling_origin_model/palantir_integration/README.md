# Palantir Foundry integration

This folder contains the Foundry-specific layer for the existing local machine-risk project. The local development scripts remain the source of truth for model training, rolling-origin validation, probability calibration, threshold selection, tier-policy construction, and final artifact creation.

## Recommended Code Repository layout

Keep this folder as a Python package inside the repository, including `__init__.py`. The feature-preparation transform also imports the existing local snapshot modules, so include these files in the same Python source package or expose them through an internal library:

```text
config.py
snapshot_builder.py
target_events.py
palantir_integration/
  __init__.py
  foundry_paths.py
  data_preparation_core.py
  scoring_core.py
  xgboost_classifier_serializer.py
  machine_risk_model_adapter.py
  01_prepare_features_transform.py
  02_publish_model_transform.py
  03_run_inference_transform.py
  04_postprocess_predictions_transform.py
```

The Foundry transforms use package-relative imports. Preserve this package structure when copying the files into the Code Repository.

## Configure paths once

Edit `foundry_paths.py` and replace the placeholders for:

- Fault history dataset
- Fluid history dataset
- Maintenance history dataset
- Operation history dataset
- Historical target-event dataset
- Complete machine-roster dataset
- Prepared feature dataset
- Unstructured model-files dataset
- Published model asset
- Full prediction dataset
- UI/customer output dataset

`MODEL_VERSION = None` consumes the latest model version. Set it to a specific version identifier when inference must be pinned to an approved release.

## Build and deployment order

### 1. Prepare model files

Run the local development workflow through script `09_migrate_production_settings.py`. Upload the following artifacts from `models/<target_source>/` to the unstructured dataset configured by `MODEL_FILES_DATASET`:

- `xgboost_base27_plus_history.json`
- `model_metadata_base27_plus_history.json`
- `tier_policy_base27_plus_history.json`

Update the filenames in `foundry_paths.py` when a different variant is promoted.

### 2. Publish the model

Build `02_publish_model_transform.py`.

The transform:

1. Reads the XGBoost JSON from the unstructured dataset.
2. Loads it explicitly as an `XGBClassifier`.
3. Reads the immutable calibration and tier settings from metadata and policy JSON.
4. Wraps the classifier and settings in `MachineRiskXGBoostAdapter`.
5. Publishes the adapter through `ModelOutput`.

The classifier-specific serializer uses XGBoost's JSON `save_model` and `load_model` interface and restores an `XGBClassifier`, so `predict_proba` remains available after deserialization.

### 3. Prepare current features

Build `01_prepare_features_transform.py`.

The transform reads source datasets as pandas dataframes, determines the scoring cutoff, materializes a temporary source bundle, and calls the same `snapshot_builder.py` functions used locally. This minimizes differences between local development and Foundry feature computation.

### 4. Run model inference

Build `03_run_inference_transform.py`.

It passes the prepared feature dataset to the published model through:

```python
ModelInput(MODEL_ASSET, model_version=MODEL_VERSION, use_sidecar=True)
```

The adapter returns raw score, calibrated probability, risk index, operating-threshold flag, provisional tier, confirmed tier, rank, decision reason, and optional positive XGBoost contributions.

### 5. Create the UI output

Build `04_postprocess_predictions_transform.py`.

This transform validates a stable output schema, rounds customer-facing display values, orders tiers from Critical through Low, and writes the final application dataset.

## Source-history expectations

The prepared input datasets must contain enough history for the promoted feature design:

- Fault: at least the prior `LOOKBACK_DAYS`.
- Operation: at least the prior `LOOKBACK_DAYS`.
- Maintenance: enough retained history for the configured recency and prior reset features.
- Historical target events: prior records only; no future labels are needed for inference.
- Fluid: sufficient retained history for last-observation-carried-forward and worsening-trend features.
- Machine roster: one row per machine, preferably with `machine_key`.

When `PALANTIR_SCORE_DATE` is `None`, the feature transform uses the latest operation date as the cutoff. The snapshot window remains:

```text
[score_date - LOOKBACK_DAYS, score_date)
```

## Python dependencies

Add the packages in `requirements.txt` to the relevant Foundry environment. The model-publishing repository and the model sidecar should use an XGBoost version compatible with the locally saved model. The included requirement pins the version used to produce and verify the packaged artifact.

## Official Palantir references

- https://www.palantir.com/docs/foundry/integrate-models/serialization
- https://www.palantir.com/docs/foundry/integrate-models/model-asset-files
- https://www.palantir.com/docs/foundry/integrate-models/model-adapter-creation
- https://www.palantir.com/docs/foundry/integrate-models/transform-model-input
