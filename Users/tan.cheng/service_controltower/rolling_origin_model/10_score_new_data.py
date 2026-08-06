"""Prepare and score either actual or mocked incoming source data.

The default mode is controlled by ``SCORING_INPUT_MODE`` in ``config.py``:

``new_data``
    Read source extracts already placed in ``INCOMING_DATA_DIR``. A score date
    must be supplied through ``--score-date`` or ``INCOMING_SCORE_DATE``.

``mocked_data``
    Recreate canonical incoming files in ``rolling_origin_model/incoming-dir``
    from the latest retained source history, then execute the exact same
    preprocessing and inference path. No records on or after the score date are
    included and no future outcomes are evaluated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from production_scoring import generate_mocked_incoming_data, run_incoming_scoring


def parse_arguments() -> argparse.Namespace:
    """Parse scoring mode, input path, cutoff date, and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a current fleet snapshot, predict raw XGBoost risk, apply "
            "probability calibration, calculate risk_index, and assign tiers."
        )
    )
    parser.add_argument(
        "--data-mode",
        choices=tuple(config.SUPPORTED_SCORING_INPUT_MODES),
        default=str(config.SCORING_INPUT_MODE),
        help=(
            "Use actual incoming files (new_data) or generate a deterministic "
            "three-month refresh from retained sources (mocked_data)."
        ),
    )
    parser.add_argument(
        "--incoming-dir",
        type=Path,
        default=Path(config.INCOMING_DATA_DIR),
        help=(
            "Project-local directory containing actual incoming files or receiving "
            "the generated mocked files."
        ),
    )
    parser.add_argument(
        "--score-date",
        default=None,
        help=(
            "Snapshot date in YYYY-MM-DD format. In new_data mode this overrides "
            "INCOMING_SCORE_DATE. In mocked_data mode it overrides "
            "MOCKED_INCOMING_SCORE_DATE; when neither is set, the latest retained "
            "operation date is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.PRODUCTION_SCORING_OUTPUT_DIR),
        help="Directory for the prepared snapshot, scores, summary, and manifest.",
    )
    parser.add_argument(
        "--skip-explanations",
        action="store_true",
        help="Skip per-machine positive contribution text to reduce runtime.",
    )
    return parser.parse_args()


def resolve_scoring_inputs(args: argparse.Namespace) -> tuple[Path, pd.Timestamp]:
    """Resolve or generate the incoming files and return their scoring cutoff."""
    incoming_dir = Path(args.incoming_dir)
    if args.data_mode == "mocked_data":
        generated = generate_mocked_incoming_data(
            incoming_dir=incoming_dir,
            score_date=args.score_date,
        )
        print("Mocked incoming refresh generated", flush=True)
        print(f"  score date: {generated.score_date.date()}", flush=True)
        print(f"  incoming directory: {generated.incoming_dir}", flush=True)
        print(f"  generation manifest: {generated.manifest_path}", flush=True)
        return generated.incoming_dir, generated.score_date

    configured_date = args.score_date or config.INCOMING_SCORE_DATE
    if not configured_date:
        raise ValueError(
            "new_data mode requires --score-date YYYY-MM-DD or "
            "INCOMING_SCORE_DATE in config.py."
        )
    score_date = pd.Timestamp(configured_date).normalize()
    if pd.isna(score_date):
        raise ValueError(f"Invalid score date: {configured_date!r}")
    return incoming_dir, score_date


def main() -> None:
    """Run data preparation, model inference, calibration, and tier assignment."""
    args = parse_arguments()
    incoming_dir, score_date = resolve_scoring_inputs(args)
    scores, summary, bundle = run_incoming_scoring(
        incoming_dir=incoming_dir,
        score_date=score_date,
        output_dir=args.output_dir,
        include_explanations=not args.skip_explanations,
        input_mode=args.data_mode,
    )

    print("\nProduction scoring complete")
    print(f"  input mode: {args.data_mode}")
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
