"""Build the dense monthly snapshot dataset used for model development."""
from __future__ import annotations

import json
import time

import pandas as pd

import config
from modeling_utils import ensure_directories
from snapshot_builder import build_snapshot_dataframe, load_sources


def main() -> None:
    """Load source data, build monthly snapshots, and save source profiles."""
    ensure_directories()
    started = time.time()
    print("Loading and normalizing source files...", flush=True)
    sources = load_sources()

    snapshot_dates = pd.date_range(
        config.TRAIN_SNAPSHOT_START,
        config.TRAIN_SNAPSHOT_END,
        freq=config.SNAPSHOT_FREQUENCY,
    )
    snapshot_pickle = config.OUTPUT_DIR / "snapshot_features.pkl"
    if bool(config.SNAPSHOT_BUILD_RESUME) and snapshot_pickle.exists():
        print(f"Loading existing {snapshot_pickle.name}...", flush=True)
        dataframe = pd.read_pickle(snapshot_pickle)
        dataframe["snapshot_date"] = pd.to_datetime(dataframe["snapshot_date"])
    else:
        print(
            f"Building {len(snapshot_dates)} monthly snapshots for "
            f"{len(sources.machine_index):,} machines...",
            flush=True,
        )
        dataframe = build_snapshot_dataframe(
            sources=sources,
            snapshot_dates=snapshot_dates,
            include_targets=True,
            verbose=True,
        )
        if config.WRITE_PICKLE:
            dataframe.to_pickle(snapshot_pickle)
        if config.WRITE_COMPRESSED_CSV:
            dataframe.to_csv(
                config.OUTPUT_DIR / "snapshot_features.csv.gz",
                index=False,
                compression="gzip",
            )

    elapsed_seconds = time.time() - started
    profile = dict(sources.profile)
    profile.update(
        {
            "snapshot_rows": int(len(dataframe)),
            "snapshot_count": int(dataframe["snapshot_date"].nunique()),
            "snapshot_date_min": str(dataframe["snapshot_date"].min().date()),
            "snapshot_date_max": str(dataframe["snapshot_date"].max().date()),
            "snapshot_positive_rows": int(dataframe[config.TARGET_COLUMN].sum()),
            "snapshot_positive_rate": float(dataframe[config.TARGET_COLUMN].mean()),
            "lookback_days": int(config.LOOKBACK_DAYS),
            "horizon_days": int(config.HORIZON_DAYS),
            "target_source": config.TARGET_SOURCE,
            "target_display_name": config.TARGET_DISPLAY_NAME,
            "elapsed_seconds": elapsed_seconds,
        }
    )
    (config.OUTPUT_DIR / "source_and_snapshot_profile.json").write_text(
        json.dumps(profile, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "snapshot_date": date,
                "rows": len(group),
                "positive_rows": int(group[config.TARGET_COLUMN].sum()),
                "positive_rate": float(group[config.TARGET_COLUMN].mean()),
            }
            for date, group in dataframe.groupby("snapshot_date")
        ]
    ).to_csv(config.OUTPUT_DIR / "snapshot_month_profile.csv", index=False)

    print("\nSnapshot build complete")
    print(f"  rows: {len(dataframe):,}")
    print(f"  machines: {dataframe['machine_key'].nunique():,}")
    print(f"  positive rate: {dataframe[config.TARGET_COLUMN].mean():.2%}")
    print(f"  target: {config.TARGET_DISPLAY_NAME}")
    print(f"  elapsed: {elapsed_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
