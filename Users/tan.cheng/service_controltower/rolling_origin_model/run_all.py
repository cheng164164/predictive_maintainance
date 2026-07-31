"""Run the rolling-origin pipeline for one or both target sources.

Examples::

    python run_all.py --target-source physical_failure
    python run_all.py --target-source warranty
    python run_all.py --target-source both

The source switch is passed through the ``RISK_TARGET_SOURCE`` environment
variable. Each target writes to its own outputs/ and models/ subdirectory.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPTS = (
    "00_validate_configuration.py",
    "00_prepare_operation_cache.py",
    "01_build_snapshot_dataset.py",
    "02_run_smoke_validation.py",
    "03_train_final_xgboost.py",
    "04_score_latest.py",
    "10_multi_anchor_fleet_evaluation.py",
)


def _run(script: str, target_source: str) -> None:
    env = os.environ.copy()
    env["RISK_TARGET_SOURCE"] = target_source
    command = [sys.executable, str(PROJECT_DIR / script)]
    print(
        f"\n===== target={target_source} | running {script} =====",
        flush=True,
    )
    subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-source",
        choices=("physical_failure", "warranty", "both"),
        default=os.getenv("RISK_TARGET_SOURCE", "physical_failure"),
        help="Outcome table used to build labels and prior-event history features.",
    )
    parser.add_argument(
        "--skip-operation-cache",
        action="store_true",
        help="Skip Step 00 when the operation_clean.csv cache already exists.",
    )
    parser.add_argument(
        "--skip-latest-scoring",
        action="store_true",
        help="Run validation and multi-anchor evaluation without Step 04.",
    )
    args = parser.parse_args()

    targets = (
        ("physical_failure", "warranty")
        if args.target_source == "both"
        else (args.target_source,)
    )

    for target_source in targets:
        for script in PIPELINE_SCRIPTS:
            if args.skip_operation_cache and script == "00_prepare_operation_cache.py":
                continue
            if args.skip_latest_scoring and script == "04_score_latest.py":
                continue
            _run(script, target_source)

    if len(targets) == 2:
        print("\n===== comparing target sources =====", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "11_compare_target_sources.py")],
            cwd=PROJECT_DIR,
            check=True,
        )


if __name__ == "__main__":
    main()
