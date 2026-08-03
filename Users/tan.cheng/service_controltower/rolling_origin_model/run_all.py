"""Run the rolling-origin pipeline using settings from config.py only.

No command-line arguments are required. Select the target source, feature
variants, algorithms, and optional steps in config.py, then run::

    python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import config

PROJECT_DIR = Path(__file__).resolve().parent


def configured_pipeline_scripts() -> tuple[str, ...]:
    scripts: list[str] = ["00_validate_configuration.py"]
    if config.RUN_OPERATION_CACHE_STEP:
        scripts.append("00_prepare_operation_cache.py")
    scripts.extend(
        [
            "01_build_snapshot_dataset.py",
            "02_run_smoke_validation.py",
            "03_train_final_xgboost.py",
        ]
    )
    if config.RUN_LATEST_SCORING_STEP:
        scripts.append("04_score_latest.py")
    if config.MULTI_ANCHOR_FLEET_ENABLED and config.RUN_MULTI_ANCHOR_FLEET_STEP:
        scripts.append("10_multi_anchor_fleet_evaluation.py")
    return tuple(scripts)


def main() -> None:
    print("Rolling-origin pipeline configuration", flush=True)
    print(f"  target source: {config.TARGET_SOURCE}", flush=True)
    print(f"  model variants: {config.MODEL_VARIANTS}", flush=True)
    print(
        f"  multi-anchor variants: {config.MULTI_ANCHOR_FLEET_MODEL_VARIANTS}",
        flush=True,
    )
    print(f"  algorithms: {config.ALGORITHMS}", flush=True)

    for script in configured_pipeline_scripts():
        print(f"\n===== running {script} =====", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / script)],
            cwd=PROJECT_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
