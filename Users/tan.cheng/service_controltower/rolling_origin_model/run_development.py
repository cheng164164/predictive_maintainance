"""Run the complete model-development and release-preparation workflow.

The workflow is controlled by the stage flags in ``config.py``. Each stage is
executed in a separate Python process so that newly tuned parameters written by
an earlier stage are reloaded by every later stage.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import config

PROJECT_DIR = Path(__file__).resolve().parent


def configured_stages() -> tuple[str, ...]:
    """Return enabled development stages in dependency order."""
    stages: list[str] = []
    if config.RUN_OPERATION_PREPARATION_STEP:
        stages.append("00_prepare_operation_data.py")
    stages.append("01_build_snapshot_dataset.py")
    if config.RUN_XGBOOST_TUNING_STEP:
        stages.append("02_tune_xgboost.py")
    if config.RUN_ROLLING_ORIGIN_VALIDATION_STEP:
        stages.append("03_rolling_origin_validation.py")
    if config.RUN_PROBABILITY_CALIBRATION_STEP:
        stages.append("04_calibrate_probabilities.py")
    if config.RUN_THRESHOLD_SELECTION_STEP:
        stages.append("05_select_operating_threshold.py")
    if config.RUN_MULTI_ANCHOR_VALIDATION_STEP:
        stages.append("06_multi_anchor_validation.py")
    if config.RUN_TIER_POLICY_STEP:
        stages.append("07_build_tier_policy.py")
    if config.RUN_FINAL_MODEL_TRAINING_STEP:
        stages.append("08_train_final_model.py")
    if config.RUN_PRODUCTION_SETTINGS_EXPORT_STEP:
        stages.append("09_export_production_settings.py")
    return tuple(stages)


def main() -> None:
    """Execute each enabled development stage and stop on the first failure."""
    print("Model development workflow", flush=True)
    print(f"  target source: {config.TARGET_SOURCE}", flush=True)
    print(f"  model variants: {config.MODEL_VARIANTS}", flush=True)
    print(f"  algorithms: {config.ALGORITHMS}", flush=True)
    print(f"  calibration: {config.PROBABILITY_CALIBRATION_METHOD}", flush=True)
    print(f"  threshold metric: {config.THRESHOLD_SELECTION_METRIC}", flush=True)
    print(f"  tier precision targets: {config.TIER_PRECISION_TARGETS}", flush=True)

    for stage in configured_stages():
        print(f"\n===== running {stage} =====", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / stage)],
            cwd=PROJECT_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
