"""Rebuild the precision-governed tier policy from saved anchor predictions.

This standalone step is useful when tier precision targets, confidence settings,
Top-N grid limits, or score-gate quantiles change. It does not retrain models or
rebuild anchor features.
"""
from __future__ import annotations

import pandas as pd

import config
from risk_tier_utils import build_validation_tier_policy


def main() -> None:
    """Rebuild tier boundaries and score-gate statistics from anchor outputs."""
    anchor_dir = config.OUTPUT_DIR / config.MULTI_ANCHOR_FLEET_OUTPUT_SUBDIR
    prediction_path = anchor_dir / "multi_anchor_all_fleet_scores_sorted.csv.gz"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing {prediction_path}. Run 06_multi_anchor_validation.py first."
        )
    try:
        predictions = pd.read_csv(prediction_path, low_memory=False)
    except (EOFError, OSError, ValueError) as exc:
        print(
            f"Combined prediction archive is incomplete ({type(exc).__name__}); "
            "rebuilding it from per-anchor checkpoints.",
            flush=True,
        )
        prediction_files = sorted(
            (anchor_dir / "predictions_by_anchor").glob("*/*/*.csv.gz")
        )
        if not prediction_files:
            raise FileNotFoundError(
                "No per-anchor prediction checkpoints were found."
            ) from exc
        predictions = pd.concat(
            [pd.read_csv(path, low_memory=False) for path in prediction_files],
            ignore_index=True,
        )
        predictions = predictions.sort_values(
            [
                "algorithm",
                "variant",
                "fold",
                "snapshot_date",
                "risk_rank_within_anchor",
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        predictions.to_csv(
            prediction_path, index=False, compression="gzip"
        )
    for column in (
        "snapshot_date",
        "feature_window_start",
        "feature_window_end_exclusive",
        "outcome_window_start",
        "outcome_window_end_exclusive",
        "next_target_event_date",
    ):
        if column in predictions.columns:
            predictions[column] = pd.to_datetime(
                predictions[column], errors="coerce"
            )
    output_dir = anchor_dir / config.TIER_POLICY_OUTPUT_SUBDIR
    summary = build_validation_tier_policy(predictions, output_dir)
    selected = pd.DataFrame(summary["selected_boundaries"])
    print("Selected cumulative Top-N boundaries")
    if selected.empty:
        print("  No tiers were selected.")
    else:
        print(
            selected[
                [
                    "algorithm",
                    "variant",
                    "tier",
                    "selected_top_n",
                    "required_precision",
                    "cumulative_micro_precision",
                    "cumulative_precision_anchor_bootstrap_ci95_low",
                    "band_precision",
                    "band_precision_anchor_bootstrap_ci95_low",
                    "selection_status",
                ]
            ].to_string(index=False)
        )
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
