"""Preprocess an incoming three-month source refresh and score all machines.

Example::

    python 10_score_new_data.py \
        --incoming-dir ../incoming_data \
        --score-date 2026-07-01

The script validates source schemas and coverage, merges longer-memory historical
state where required, builds one leakage-safe snapshot, predicts raw XGBoost
scores, applies Platt calibration, calculates ``risk_index``, and assigns final
risk tiers with the approved Top-N plus score-gate logic from ``config.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from production_scoring import run_incoming_scoring


def parse_arguments() -> argparse.Namespace:
    """Parse production scoring paths, date, and explanation options."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the latest three months of source data and produce calibrated "
            "machine-risk scores and risk tiers."
        )
    )
    parser.add_argument(
        "--incoming-dir",
        type=Path,
        default=Path(config.INCOMING_DATA_DIR),
        help="Directory containing the newest fault, fluid, maintenance, operation, and optional target files.",
    )
    parser.add_argument(
        "--score-date",
        default=config.INCOMING_SCORE_DATE,
        help="Snapshot date in YYYY-MM-DD format. Required unless INCOMING_SCORE_DATE is set in config.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.PRODUCTION_SCORING_OUTPUT_DIR),
        help="Directory for the prepared snapshot, machine scores, summary, and source manifest.",
    )
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Skip per-machine top positive contribution text to reduce runtime.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete incoming-data preprocessing and scoring workflow."""
    args = parse_arguments()
    if not args.score_date:
        raise ValueError(
            "A production score date is required. Pass --score-date YYYY-MM-DD "
            "or set INCOMING_SCORE_DATE in config.py."
        )
    score_date = pd.Timestamp(args.score_date).normalize()
    scores, summary, bundle = run_incoming_scoring(
        incoming_dir=args.incoming_dir,
        score_date=score_date,
        output_dir=args.output_dir,
        include_explanations=not args.skip_explanations,
    )
    print("\nProduction scoring complete")
    print(f"  score date: {score_date.date()}")
    print(f"  machines: {len(scores):,}")
    print(f"  source manifest: {bundle.manifest_path}")
    print(
        "  final tiers: "
        f"Critical={summary['critical_confirmed']:,}, "
        f"High={summary['high_confirmed']:,}, "
        f"Medium={summary['medium_confirmed']:,}, "
        f"Low={summary['low_confirmed']:,}"
    )
    print(f"  output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
