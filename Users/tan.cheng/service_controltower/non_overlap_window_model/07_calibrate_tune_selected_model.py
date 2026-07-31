"""Step 07: freeze the selected model, Platt calibrator, and validation threshold."""
from __future__ import annotations

import json
import shutil

import joblib
import pandas as pd

import config


def main() -> None:
    config.STEP_07_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(config.CANDIDATE_ARTIFACT_PATH)
    if artifact["algorithm"] != config.SELECTED_ALGORITHM or artifact["variant"] != config.MODEL_VARIANT:
        raise ValueError("Candidate artifact does not match config.py selection.")
    shutil.copy2(config.CANDIDATE_ARTIFACT_PATH, config.SELECTED_MODEL_ARTIFACT_PATH)
    metrics = pd.read_csv(config.STEP_06_OUTPUT_DIR / "validation_model_metrics.csv")
    selected = metrics[
        metrics["algorithm"].eq(config.SELECTED_ALGORITHM)
        & metrics["variant"].eq(config.MODEL_VARIANT)
    ].copy()
    selected.to_csv(config.STEP_07_OUTPUT_DIR / "selected_model_validation_metrics.csv", index=False)
    pd.DataFrame([{"threshold": artifact["threshold"], "metric": config.THRESHOLD_TUNING_METRIC}]).to_csv(
        config.STEP_07_OUTPUT_DIR / "selected_model_threshold.csv", index=False
    )
    (config.STEP_07_OUTPUT_DIR / "selected_model_metadata.json").write_text(
        json.dumps(
            {
                "target_source": config.TARGET_SOURCE,
                "algorithm": artifact["algorithm"],
                "variant": artifact["variant"],
                "feature_count": len(artifact["features"]),
                "threshold": artifact["threshold"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selected.to_string(index=False))
    print(f"Frozen threshold: {artifact['threshold']:.8f}")


if __name__ == "__main__":
    main()
