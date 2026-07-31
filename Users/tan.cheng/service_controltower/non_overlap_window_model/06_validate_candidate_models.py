"""Step 06: publish validation metrics, Top-K/Top-N, and ranked predictions."""
from __future__ import annotations

import pandas as pd

import config
from modeling_90d import pooled_selection_summary


def main() -> None:
    config.STEP_06_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_metrics.csv")
    selections = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_top_k_top_n_by_period.csv")
    predictions = pd.read_csv(config.STEP_05_OUTPUT_DIR / "native_predictions.csv.gz", low_memory=False)
    validation_metrics = metrics[metrics["evaluation_scope"].eq("native_validation")].copy()
    validation_selections = selections[selections["evaluation_scope"].eq("native_validation")].copy()
    validation_predictions = predictions[predictions["evaluation_scope"].eq("native_validation")].copy()
    validation_metrics.to_csv(config.STEP_06_OUTPUT_DIR / "validation_model_metrics.csv", index=False)
    validation_selections.to_csv(config.STEP_06_OUTPUT_DIR / "validation_top_k_top_n_by_period.csv", index=False)
    pooled_selection_summary(validation_selections).to_csv(
        config.STEP_06_OUTPUT_DIR / "validation_top_k_top_n_summary.csv", index=False
    )
    validation_predictions.to_csv(
        config.STEP_06_OUTPUT_DIR / "validation_ranked_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    print(validation_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
