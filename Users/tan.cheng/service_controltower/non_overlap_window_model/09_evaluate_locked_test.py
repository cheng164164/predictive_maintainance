"""Step 09: publish locked-test metrics, Top-K/Top-N, and ranked predictions."""
from __future__ import annotations

import pandas as pd

import config
from modeling_90d import pooled_selection_summary


def main() -> None:
    config.STEP_09_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_metrics.csv")
    selections = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_top_k_top_n_by_period.csv")
    predictions = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_predictions.csv.gz", low_memory=False)
    locked_metrics = metrics[metrics["evaluation_scope"].eq("native_locked_test")].copy()
    locked_selections = selections[selections["evaluation_scope"].eq("native_locked_test")].copy()
    locked_predictions = predictions[predictions["evaluation_scope"].eq("native_locked_test")].copy()
    locked_metrics.to_csv(config.STEP_09_OUTPUT_DIR / "locked_test_metrics.csv", index=False)
    locked_selections.to_csv(config.STEP_09_OUTPUT_DIR / "locked_test_top_k_top_n_by_period.csv", index=False)
    pooled_selection_summary(locked_selections).to_csv(
        config.STEP_09_OUTPUT_DIR / "locked_test_top_k_top_n_summary.csv", index=False
    )
    locked_predictions.to_csv(
        config.STEP_09_OUTPUT_DIR / "locked_test_ranked_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print(locked_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
