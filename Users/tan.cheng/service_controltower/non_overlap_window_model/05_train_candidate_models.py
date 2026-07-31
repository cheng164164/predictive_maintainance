"""Step 05: fit configured candidate model(s) on the native 90-day split."""
from __future__ import annotations

import json
import shutil

import pandas as pd

import config
from modeling_90d import native_experiment
from pipeline_90d_io import load_split_definition


def main() -> None:
    config.STEP_05_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(config.FULL_DATASET_PATH)
    split_dates = load_split_definition()
    metrics, selections, predictions = native_experiment(
        data,
        target_source=config.TARGET_SOURCE,
        split_dates=split_dates,
        output_dir=config.STEP_05_OUTPUT_DIR,
        skip_candidate_screen=not config.RUN_CANDIDATE_SCREEN,
    )
    source_artifact = config.STEP_05_OUTPUT_DIR / f"{config.MODEL_VARIANT}__{config.SELECTED_ALGORITHM}.joblib"
    if not source_artifact.exists():
        raise FileNotFoundError(f"Selected artifact was not created: {source_artifact}")
    shutil.copy2(source_artifact, config.CANDIDATE_ARTIFACT_PATH)
    summary = {
        "target_source": config.TARGET_SOURCE,
        "model_variant": config.MODEL_VARIANT,
        "selected_algorithm": config.SELECTED_ALGORITHM,
        "candidate_screen_enabled": config.RUN_CANDIDATE_SCREEN,
        "metric_rows": len(metrics),
        "selection_rows": len(selections),
        "prediction_rows": len(predictions),
    }
    (config.STEP_05_OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
