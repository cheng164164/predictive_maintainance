"""Step 10: consolidate native validation and locked-test results."""
from __future__ import annotations

import pandas as pd

import config


def main() -> None:
    config.STEP_10_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(config.STEP_06_OUTPUT_DIR / "validation_model_metrics.csv")
    locked = pd.read_csv(config.STEP_09_OUTPUT_DIR / "locked_test_metrics.csv")
    metrics = pd.concat([validation, locked], ignore_index=True, sort=False)
    metrics.to_csv(config.STEP_10_OUTPUT_DIR / "native_metrics_summary.csv", index=False)
    topk_val = pd.read_csv(config.STEP_06_OUTPUT_DIR / "validation_top_k_top_n_summary.csv")
    topk_test = pd.read_csv(config.STEP_09_OUTPUT_DIR / "locked_test_top_k_top_n_summary.csv")
    ranking = pd.concat([topk_val, topk_test], ignore_index=True, sort=False)
    ranking.to_csv(config.STEP_10_OUTPUT_DIR / "native_top_k_top_n_summary.csv", index=False)
    print(metrics.to_string(index=False))
    print("\nRanking summary")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
