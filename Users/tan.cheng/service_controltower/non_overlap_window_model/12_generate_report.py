"""Step 12: generate a concise native model report from saved outputs."""
from __future__ import annotations

import pandas as pd

import config


def main() -> None:
    config.STEP_12_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(config.STEP_10_OUTPUT_DIR / "native_metrics_summary.csv")
    ranking = pd.read_csv(config.STEP_10_OUTPUT_DIR / "native_top_k_top_n_summary.csv")
    report = f"""# 90-Day Future-Event Risk Model Report

## Configuration

- Target source: `{config.TARGET_SOURCE}`
- Feature lookback: {config.LOOKBACK_DAYS} days
- Prediction horizon: {config.HORIZON_DAYS} days
- Segment stride: {config.SEGMENT_STRIDE_DAYS} days
- Feature variant: `{config.MODEL_VARIANT}`
- Selected algorithm: `{config.SELECTED_ALGORITHM}`
- Warranty filter mode: `{config.WARRANTY_FILTER_MODE}`

## Native chronological metrics

{metrics.to_markdown(index=False, floatfmt='.4f')}

## Native Top-K and Top-N metrics

{ranking.to_markdown(index=False, floatfmt='.4f')}

## Anchor evaluation

Run `python 14_anchor_fleet_validation.py` separately. It retrains a fold-specific model and writes one fully ranked fleet file per configured anchor date.
"""
    path = config.STEP_12_OUTPUT_DIR / "MODEL_REPORT.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
