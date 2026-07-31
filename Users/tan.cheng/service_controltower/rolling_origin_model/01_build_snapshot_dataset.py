"""Build the dense training snapshots and latest scoring snapshot."""
from __future__ import annotations

import json
import time

import pandas as pd

import config
from modeling_utils import ensure_directories
from snapshot_builder import build_snapshot_dataframe, load_sources


def main() -> None:
    ensure_directories()
    started = time.time()
    print('Loading and normalizing source files...', flush=True)
    sources = load_sources()

    snapshot_dates = pd.date_range(
        config.TRAIN_SNAPSHOT_START,
        config.TRAIN_SNAPSHOT_END,
        freq=config.SNAPSHOT_FREQUENCY,
    )
    print(
        f'Building {len(snapshot_dates)} monthly training snapshots for '
        f'{len(sources.machine_index):,} machines...',
        flush=True,
    )
    dataframe = build_snapshot_dataframe(
        sources,
        snapshot_dates,
        include_targets=True,
        verbose=True,
    )

    if config.WRITE_PICKLE:
        dataframe.to_pickle(config.OUTPUT_DIR / 'snapshot_features.pkl')
    if config.WRITE_COMPRESSED_CSV:
        dataframe.to_csv(
            config.OUTPUT_DIR / 'snapshot_features.csv.gz',
            index=False,
            compression='gzip',
        )

    print(f'Building latest score date {config.LATEST_SCORE_DATE}...', flush=True)
    latest = build_snapshot_dataframe(
        sources,
        [config.LATEST_SCORE_DATE],
        include_targets=False,
        verbose=True,
    )
    if config.WRITE_PICKLE:
        latest.to_pickle(config.OUTPUT_DIR / 'latest_snapshot_features.pkl')
    if config.WRITE_COMPRESSED_CSV:
        latest.to_csv(
            config.OUTPUT_DIR / 'latest_snapshot_features.csv.gz',
            index=False,
            compression='gzip',
        )

    profile = dict(sources.profile)
    profile.update(
        {
            'snapshot_rows': int(len(dataframe)),
            'snapshot_count': int(dataframe['snapshot_date'].nunique()),
            'snapshot_date_min': str(dataframe['snapshot_date'].min().date()),
            'snapshot_date_max': str(dataframe['snapshot_date'].max().date()),
            'snapshot_positive_rows': int(dataframe[config.TARGET_COLUMN].sum()),
            'snapshot_positive_rate': float(dataframe[config.TARGET_COLUMN].mean()),
            'latest_score_date': config.LATEST_SCORE_DATE,
            'latest_score_rows': int(len(latest)),
            'lookback_days': config.LOOKBACK_DAYS,
            'horizon_days': config.HORIZON_DAYS,
            'target_source': config.TARGET_SOURCE,
            'target_display_name': config.TARGET_DISPLAY_NAME,
            'raw_warranty_csv_used': config.TARGET_SOURCE == 'warranty',
            'elapsed_seconds': time.time() - started,
        }
    )
    (config.OUTPUT_DIR / 'source_and_snapshot_profile.json').write_text(
        json.dumps(profile, indent=2), encoding='utf-8'
    )
    pd.DataFrame(
        [
            {
                'snapshot_date': date,
                'rows': len(group),
                'positive_rows': int(group[config.TARGET_COLUMN].sum()),
                'positive_rate': float(group[config.TARGET_COLUMN].mean()),
            }
            for date, group in dataframe.groupby('snapshot_date')
        ]
    ).to_csv(config.OUTPUT_DIR / 'snapshot_month_profile.csv', index=False)

    print('\nBuild complete')
    print(f"  rows: {len(dataframe):,}")
    print(f"  machines: {dataframe['machine_key'].nunique():,}")
    print(f"  positive rate: {dataframe[config.TARGET_COLUMN].mean():.2%}")
    print(f"  target source: {profile['target_source']} ({profile['target_display_name']})")
    print(f"  elapsed: {profile['elapsed_seconds']:.1f} seconds")


if __name__ == '__main__':
    main()
