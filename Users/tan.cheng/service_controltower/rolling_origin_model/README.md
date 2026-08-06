# Rolling-Origin Machine Risk Model

This project contains the complete development workflow, production scoring workflow, and Palantir Foundry integration components for the calibrated 90-day machine-risk model.

The numbered local workflow remains unchanged in structure:

- Development and model promotion: scripts `00` through `09`
- Deployment inference: script `10`
- Foundry-specific transforms: `palantir_integration/`

## Local folder layout

```text
project_root/
  enriched_data/
    fault_codes.csv
    fluid_samples.csv
    maintenance.csv
    operation.csv
    Physical Failure Events.csv
    warranty.csv                  # required only for TARGET_SOURCE="warranty"
  rolling_origin_model/
    config.py
    run_development.py
    00_prepare_operation_data.py
    ...
    10_score_new_data.py
    incoming-dir/                 # actual or generated three-month refresh
    models/
    outputs/
    palantir_integration/
```

## Development and retraining workflow

Run all enabled stages in dependency order:

```bash
python run_development.py
```

The stage flags are in the last section of `config.py`.

1. `00_prepare_operation_data.py`
   - Cleans the operation extract and creates the retained machine roster.
2. `01_build_snapshot_dataset.py`
   - Builds monthly leakage-safe historical machine snapshots.
3. `02_tune_xgboost.py`
   - Tunes XGBoost L1, L2, and early-stopping settings across rolling-origin folds.
4. `03_rolling_origin_validation.py`
   - Evaluates configured algorithms and model variants on forward validation folds.
5. `04_calibrate_probabilities.py`
   - Fits and evaluates probability calibration without using future validation outcomes.
6. `05_select_operating_threshold.py`
   - Selects the configured F1 or F2 operating threshold.
7. `06_multi_anchor_validation.py`
   - Scores complete anchor fleets and exports fixed Top-N performance and score statistics.
8. `07_build_tier_policy.py`
   - Selects Critical, High, and Medium Top-N boundaries and confidence-aware score gates.
9. `08_train_final_model.py`
   - Fits the final model and exports calibration, tier thresholds, feature importance, SHAP results, and learning curves.
10. `09_migrate_production_settings.py`
    - Exports reviewable production settings or atomically promotes them into `config.py`, depending on `PRODUCTION_SETTINGS_MODE`.

### Production-settings export or migration

The safe default is:

```python
PRODUCTION_SETTINGS_MODE = "export_only"
```

Run stage 09 directly with the configured mode:

```bash
python 09_migrate_production_settings.py
```

In `export_only` mode, the script writes `production_settings_candidate.json` and a complete `config_candidate.py` without modifying the active `config.py`. To perform the reviewed promotion, set `PRODUCTION_SETTINGS_MODE = "update_config"` or run:

```bash
python 09_migrate_production_settings.py --mode update_config
```

The update mode replaces only `XGB_PARAMS`, `PRODUCTION_MODEL_SETTINGS`, `PRODUCTION_SELECTED_FAULT_CODES`, and `PRODUCTION_TIER_POLICY`. It syntax-checks the candidate, creates a timestamped backup when enabled, imports the new configuration in a separate process, verifies exact equality, and restores the original file if verification fails. Both modes write an audit record with source-artifact hashes.

The tuned parameter JSON is preferred. If tuning was intentionally skipped and `xgboost_tuned_params.json` is absent, stage 09 reuses the approved `XGB_PARAMS` block and writes a separate hashed fallback artifact. Set `PRODUCTION_SETTINGS_REQUIRE_TUNED_PARAMS_FILE = True` to require the JSON and fail instead.

Set `RUN_PRODUCTION_SETTINGS_MIGRATION_STEP = False` to skip stage 09 entirely.

## Deployment workflow

Script `10_score_new_data.py` supports two input modes through `SCORING_INPUT_MODE` in `config.py` or `--data-mode` on the command line.

### Actual incoming data

Set:

```python
SCORING_INPUT_MODE = "new_data"
```

Place the latest three months of source files in `rolling_origin_model/incoming-dir/`, then run:

```bash
python 10_score_new_data.py \
    --data-mode new_data \
    --score-date YYYY-MM-DD
```

### Mocked incoming data

Set:

```python
SCORING_INPUT_MODE = "mocked_data"
```

Then run:

```bash
python 10_score_new_data.py --data-mode mocked_data
```

Set `MOCKED_INCOMING_SCORE_DATE` to any historical cutoff you want to examine, for example `"2026-06-01"`. Leave it as `None` to use the latest retained operation date. A one-run `--score-date YYYY-MM-DD` argument overrides the config value. The script writes canonical source files for the preceding configured history window to `rolling_origin_model/incoming-dir/` and records the selected cutoff, exact date range, and source hashes in `mocked_incoming_manifest.json`.

The feature window is left-closed and right-open:

```text
[score_date - LOOKBACK_DAYS, score_date)
```

Therefore, mocked inference excludes records on or after the score date. Historical target records may be used only for prior-event features; no future target is read and no validation metric is calculated.

Both modes then use the same production path to:

- validate incoming source schemas and history coverage;
- combine longer-memory source history where required;
- preserve the complete retained machine roster;
- build one current snapshot per machine;
- generate the raw XGBoost probability;
- apply the promoted Platt calibrator;
- calculate `risk_index = calibrated probability x 100`;
- create nested Top-N tier candidates;
- apply the promoted score gates with demotion-only logic; and
- save machine scores, final tiers, source manifests, and tier-count summaries.

## Palantir Foundry integration

The `palantir_integration/` folder keeps Foundry-specific code separate from the local workflow.

- `01_prepare_features_transform.py` - converts Foundry source datasets into one model-ready machine snapshot.
- `02_publish_model_transform.py` - loads the locally trained XGBoost JSON, metadata, and tier policy from an unstructured dataset and publishes a model version.
- `03_run_inference_transform.py` - invokes the published model through `ModelInput`.
- `04_postprocess_predictions_transform.py` - produces a stable UI/customer output dataset.
- `machine_risk_model_adapter.py` - defines the Palantir model API and serialized scoring behavior.
- `xgboost_classifier_serializer.py` - preserves the XGBoost classifier type while using XGBoost JSON serialization.
- `data_preparation_core.py` and `scoring_core.py` - reusable feature and inference logic.
- `foundry_paths.py` - central location for dataset, model, version, and filename placeholders.

See `palantir_integration/README.md` for repository placement, required model files, data-retention expectations, and build order.
