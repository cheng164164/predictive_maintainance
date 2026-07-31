"""Step 13: validate required native outputs and ranking invariants."""
from __future__ import annotations

import json

import pandas as pd

import config


def main() -> None:
    config.STEP_13_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        config.FULL_DATASET_PATH,
        config.MODEL_FEATURE_LIST_PATH,
        config.FINAL_MODEL_ARTIFACT_PATH,
        config.STEP_09_OUTPUT_DIR / "locked_test_metrics.csv",
        config.STEP_09_OUTPUT_DIR / "locked_test_ranked_predictions.csv.gz",
        config.STEP_10_OUTPUT_DIR / "native_top_k_top_n_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    predictions = pd.read_csv(config.STEP_09_OUTPUT_DIR / "locked_test_ranked_predictions.csv.gz", low_memory=False)
    checks = {
        "required_files_present": not missing,
        "prediction_labels_binary": bool(set(predictions["predicted_label"].dropna().astype(int).unique()).issubset({0, 1})),
        "true_labels_binary": bool(set(predictions["true_label"].dropna().astype(int).unique()).issubset({0, 1})),
        "probability_range_valid": bool(predictions["calibrated_probability"].between(0, 1).all()),
        "no_duplicate_machine_period_algorithm": bool(not predictions.duplicated(
            ["machine_key", "period_start", "variant", "algorithm", "evaluation_scope"]
        ).any()),
    }
    passed = not missing and all(checks.values())
    payload = {"status": "passed" if passed else "failed", "missing": missing, "checks": checks}
    (config.STEP_13_OUTPUT_DIR / "quality_check_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
