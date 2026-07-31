"""Compare completed physical-failure and warranty target runs.

This script does not train models. It reads the two target-specific output
folders and writes aligned rolling-fold, anchor, Top-K, and Top-N comparison
CSVs under outputs/target_comparison/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "outputs"
COMPARISON_DIR = OUTPUT_ROOT / "target_comparison"
TARGETS = ("physical_failure", "warranty")


def _read(target: str, relative: str) -> pd.DataFrame:
    path = OUTPUT_ROOT / target / relative
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_all.py --target-source both first."
        )
    frame = pd.read_csv(path, low_memory=False)
    frame.insert(0, "target_source", target)
    return frame


def _wide(
    frame: pd.DataFrame,
    index: list[str],
    values: list[str],
) -> pd.DataFrame:
    kept = [column for column in values if column in frame.columns]
    result = frame.pivot_table(
        index=index,
        columns="target_source",
        values=kept,
        aggfunc="first",
    )
    result.columns = [f"{metric}__{target}" for metric, target in result.columns]
    result = result.reset_index()
    for metric in kept:
        physical = f"{metric}__physical_failure"
        warranty = f"{metric}__warranty"
        if physical in result.columns and warranty in result.columns:
            result[f"{metric}__physical_minus_warranty"] = (
                result[physical] - result[warranty]
            )
    return result


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    variant_summary = pd.concat(
        [_read(target, "xgboost_variant_summary.csv") for target in TARGETS],
        ignore_index=True,
    )
    variant_summary.to_csv(
        COMPARISON_DIR / "target_variant_summary_long.csv", index=False
    )
    _wide(
        variant_summary,
        ["variant"],
        [
            "mean_roc_auc",
            "latest_fold_auc",
            "pooled_roc_auc",
            "pooled_average_precision",
            "precision_top10",
            "recall_top10",
            "lift_top10",
        ],
    ).to_csv(COMPARISON_DIR / "target_variant_summary_wide.csv", index=False)

    fold_metrics = pd.concat(
        [_read(target, "xgboost_variant_fold_metrics.csv") for target in TARGETS],
        ignore_index=True,
    )
    fold_metrics.to_csv(COMPARISON_DIR / "target_fold_metrics_long.csv", index=False)
    _wide(
        fold_metrics,
        ["variant", "fold"],
        [
            "val_base_rate",
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f2",
            "precision_top10",
            "recall_top10",
            "lift_top10",
        ],
    ).to_csv(COMPARISON_DIR / "target_fold_metrics_wide.csv", index=False)

    anchor_relative = (
        "10_multi_anchor_fleet_evaluation/multi_anchor_metrics_by_anchor.csv"
    )
    anchor_metrics = pd.concat(
        [_read(target, anchor_relative) for target in TARGETS], ignore_index=True
    )
    anchor_metrics.to_csv(
        COMPARISON_DIR / "target_anchor_metrics_long.csv", index=False
    )
    _wide(
        anchor_metrics,
        ["variant", "fold", "anchor_id", "anchor_date"],
        [
            "positive_rate",
            "roc_auc",
            "average_precision",
            "threshold_precision",
            "threshold_recall",
        ],
    ).to_csv(COMPARISON_DIR / "target_anchor_metrics_wide.csv", index=False)

    selection_relative = (
        "10_multi_anchor_fleet_evaluation/multi_anchor_top_k_top_n_overall.csv"
    )
    selections = pd.concat(
        [_read(target, selection_relative) for target in TARGETS], ignore_index=True
    )
    selections.to_csv(
        COMPARISON_DIR / "target_top_k_top_n_long.csv", index=False
    )
    _wide(
        selections,
        ["variant", "selection_type", "selection_value"],
        [
            "natural_positive_rate",
            "micro_precision",
            "micro_recall",
            "micro_lift_vs_fleet",
            "minimum_anchor_precision",
            "maximum_anchor_precision",
        ],
    ).to_csv(COMPARISON_DIR / "target_top_k_top_n_wide.csv", index=False)

    # Verify that both runs used exactly the same anchors.
    manifests = []
    for target in TARGETS:
        frame = _read(
            target,
            "10_multi_anchor_fleet_evaluation/multi_anchor_anchor_manifest.csv",
        )
        manifests.append(frame[["target_source", "fold", "anchor_id", "anchor_date"]])
    manifest = pd.concat(manifests, ignore_index=True)
    manifest.to_csv(COMPARISON_DIR / "target_anchor_manifests.csv", index=False)
    pivot = manifest.pivot_table(
        index=["fold", "anchor_id"],
        columns="target_source",
        values="anchor_date",
        aggfunc="first",
    ).reset_index()
    pivot["same_anchor_date"] = (
        pivot["physical_failure"].astype(str) == pivot["warranty"].astype(str)
    )
    pivot.to_csv(COMPARISON_DIR / "target_anchor_alignment_check.csv", index=False)
    if not bool(pivot["same_anchor_date"].all()):
        raise AssertionError("The two target runs did not use identical anchor dates.")

    print(f"Target comparison outputs written to {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
