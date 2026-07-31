"""Run the configured numbered pipeline without command-line arguments."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent
STEPS = {
    1: "01_validate_inputs.py",
    2: "02_define_data_splits.py",
    3: "03_build_features_targets.py",
    4: "04_select_features.py",
    5: "05_train_candidate_models.py",
    6: "06_validate_candidate_models.py",
    7: "07_calibrate_tune_selected_model.py",
    8: "08_finalize_manual_selection.py",
    9: "09_evaluate_locked_test.py",
    10: "10_consolidate_results.py",
    11: "11_score_latest.py",
    12: "12_generate_report.py",
    13: "13_validate_outputs.py",
    14: "14_anchor_fleet_validation.py",
}


def main() -> None:
    start = int(config.PIPELINE_START_STEP)
    end = int(config.PIPELINE_END_STEP)
    if start not in STEPS or end not in STEPS or start > end:
        raise ValueError("Invalid PIPELINE_START_STEP / PIPELINE_END_STEP in config.py")
    config.LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_ROOT / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")

    selected_steps = list(range(start, end + 1))
    if config.RUN_ANCHOR_VALIDATION_IN_PIPELINE and 14 not in selected_steps:
        selected_steps.append(14)

    print(f"Target source: {config.TARGET_SOURCE}")
    print(f"90-day lookback / {config.HORIZON_DAYS}-day horizon")
    print(f"Log: {log_path}")
    with log_path.open("a", encoding="utf-8") as log:
        for step in selected_steps:
            script = STEPS[step]
            banner = f"\n{'='*78}\nSTEP {step:02d}: {script}\n{'='*78}\n"
            print(banner, end="")
            log.write(banner)
            process = subprocess.Popen(
                [sys.executable, str(ROOT / script)],
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, [sys.executable, script])
    print(f"\nCompleted. Output root: {config.OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
