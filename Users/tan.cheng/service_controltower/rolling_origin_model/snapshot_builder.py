"""Build dense, leakage-safe machine snapshot features.

Feature window: [snapshot_date - LOOKBACK_DAYS, snapshot_date)
Target window:  [snapshot_date, snapshot_date + HORIZON_DAYS)

The target source is selected in config.py. The same feature, split, training,
and multi-anchor evaluation logic is used for either physical-failure events or
the original warranty claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd

import config
from target_events import load_target_events


@dataclass
class SnapshotSources:
    target_events: pd.DataFrame
    target_event_day: pd.DataFrame
    fault: pd.DataFrame
    fluid: pd.DataFrame
    fluid_known: pd.DataFrame
    maintenance: pd.DataFrame
    operation: pd.DataFrame
    operation_first_date: pd.Series
    machine_index: pd.Index
    candidate_serious_codes: list[str]
    candidate_code_names: dict[str, str]
    profile: dict


def machine_key_from_model_serial(model: pd.Series, serial: pd.Series) -> pd.Series:
    model_text = model.astype('string').str.strip().str.upper()
    serial_text = serial.astype('string').str.extract(r'(\d+)', expand=False)
    return model_text + '-' + serial_text


def normalize_machine_key(series: pd.Series) -> pd.Series:
    return (
        series.astype('string')
        .str.strip()
        .str.upper()
        .str.replace(r'\s+(\d+)$', r'-\1', regex=True)
    )


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype('string')
        .str.strip()
        .str.lower()
        .map({'true': True, 'false': False, '1': True, '0': False})
        .fillna(False)
        .astype(bool)
    )


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', str(value)).strip('_')


def _require_files() -> None:
    required = [
        config.FAULT_FILE,
        config.FLUID_FILE,
        config.MAINTENANCE_FILE,
        config.OPERATION_FILE,
        config.TARGET_FILE,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('Missing required source files:\n' + '\n'.join(missing))


def load_sources(include_operation_features: bool = True) -> SnapshotSources:
    """Load and normalize all sources once."""
    _require_files()
    load_started = time.time()
    print('[sources] starting source normalization', flush=True)

    target_tables = load_target_events()
    target_events = target_tables.raw
    target_event_day = target_tables.by_machine_day
    print(
        f"[sources] target complete ({config.TARGET_SOURCE}): "
        f"{time.time() - load_started:.1f}s",
        flush=True,
    )

    fault_columns = [
        'serial_number', 'full_model', 'event_date', 'fault_code',
        'event_action_level', 'logical_name', 'occurrence_count',
        'log_occurrence_count', 'smr_hours', 'applicable_component',
        'failure_code_evidence_score', 'failure_code_evidence_group',
        'action_level_num',
    ]
    fault = pd.read_csv(config.FAULT_FILE, usecols=fault_columns, low_memory=False)
    fault['machine_key'] = machine_key_from_model_serial(
        fault['full_model'], fault['serial_number']
    )
    fault['event_date'] = pd.to_datetime(fault['event_date'], errors='coerce')
    fault['severity'] = pd.to_numeric(fault['action_level_num'], errors='coerce')
    severity_from_text = pd.to_numeric(
        fault['event_action_level'].astype('string').str.extract(r'(\d+)', expand=False),
        errors='coerce',
    )
    fault['severity'] = fault['severity'].fillna(severity_from_text).fillna(0).clip(0, 4)
    fault['log_occ'] = pd.to_numeric(fault['log_occurrence_count'], errors='coerce')
    fallback_log_occ = np.log1p(
        pd.to_numeric(fault['occurrence_count'], errors='coerce')
        .fillna(1)
        .clip(lower=0)
    )
    fault['log_occ'] = fault['log_occ'].fillna(fallback_log_occ).clip(0, 10)
    fault['evidence_score'] = (
        pd.to_numeric(fault['failure_code_evidence_score'], errors='coerce')
        .fillna(0)
        .clip(0, 2)
    )
    fault['is_serious'] = (
        fault['severity'].ge(config.SERIOUS_ACTION_LEVEL_MIN)
        & fault['failure_code_evidence_group']
        .astype('string')
        .str.upper()
        .eq(config.SERIOUS_EVIDENCE_GROUP)
    ).astype('int8')
    fault['is_l03'] = fault['severity'].eq(3).astype('int8')
    fault['is_l04'] = fault['severity'].ge(4).astype('int8')
    fault['serious_log_occ'] = fault['is_serious'] * fault['log_occ']
    fault['severity_weight'] = fault['is_serious'] * fault['severity'] * fault['log_occ']
    fault['evidence_weight'] = (
        fault['is_serious'] * fault['evidence_score'] * fault['log_occ']
    )
    fault['system'] = (
        fault['applicable_component']
        .astype('string')
        .fillna(fault['logical_name'].astype('string'))
        .str.upper()
        .fillna('UNKNOWN')
    )
    fault['is_hst'] = (
        fault['system'].str.contains('HST', na=False).fillna(False)
        | fault['logical_name']
        .astype('string')
        .str.upper()
        .eq('HST')
        .fillna(False)
    ).astype('int8')
    fault = (
        fault.dropna(subset=['machine_key', 'event_date'])
        .sort_values(['event_date', 'machine_key'])
        .reset_index(drop=True)
    )
    candidate_serious_codes = (
        fault.loc[fault['is_serious'].eq(1), 'fault_code']
        .astype('string')
        .value_counts()
        .head(config.TOP_CODE_CANDIDATE_COUNT)
        .index.tolist()
    )
    candidate_code_names = {
        code: f'fault_code_{safe_name(code)}_90d'
        for code in candidate_serious_codes
    }
    print(f'[sources] fault complete: {time.time() - load_started:.1f}s', flush=True)

    fluid_columns = [
        'FULL_MODEL', 'SERIAL', 'sample_drawn_date',
        'sample_result_severity_order', 'Water_Water_PERCENT',
        'Gly_Glycol_PERCENT', 'EthyleneGlycol_Ethylene_Glycol_PERCENT',
        'Fuel_Fuel_PERCENT', 'Si_Silicon_PPM', 'Fe_Iron_PPM',
        'Cu_Copper_PPM', 'TELEMETRY_SMR_NUMERIC',
    ]
    fluid = pd.read_csv(config.FLUID_FILE, usecols=fluid_columns, low_memory=False)
    fluid['machine_key'] = machine_key_from_model_serial(fluid['FULL_MODEL'], fluid['SERIAL'])
    fluid['event_date'] = pd.to_datetime(fluid['sample_drawn_date'], errors='coerce')
    fluid = fluid[
        fluid['event_date'].between('2015-01-01', '2026-12-31', inclusive='both')
    ].copy()
    raw_severity = pd.to_numeric(fluid['sample_result_severity_order'], errors='coerce')
    known_mask = raw_severity.between(0, 4)
    for unknown_value in config.FLUID_UNKNOWN_SEVERITY_VALUES:
        known_mask &= raw_severity.ne(unknown_value)
    fluid['severity_known'] = raw_severity.where(known_mask)
    fluid['is_abnormal'] = fluid['severity_known'].ge(2).astype('int8')
    analytes = [
        'Water_Water_PERCENT', 'Gly_Glycol_PERCENT',
        'EthyleneGlycol_Ethylene_Glycol_PERCENT', 'Fuel_Fuel_PERCENT',
        'Si_Silicon_PPM', 'Fe_Iron_PPM', 'Cu_Copper_PPM',
    ]
    for column in analytes:
        fluid[column] = pd.to_numeric(fluid[column], errors='coerce')
    # Conservative, sample-level contaminant flag. The calibrated abnormal
    # severity gate prevents normal coolant concentration from being treated as contamination.
    fluid['contaminant_flag'] = (
        fluid['is_abnormal'].eq(1)
        & (
            fluid['Water_Water_PERCENT'].gt(0.10)
            | fluid['EthyleneGlycol_Ethylene_Glycol_PERCENT'].gt(0.05)
            | fluid['Fuel_Fuel_PERCENT'].gt(2.0)
            | fluid['Si_Silicon_PPM'].gt(100.0)
            | (
                fluid['Gly_Glycol_PERCENT'].notna()
                & (
                    fluid['Gly_Glycol_PERCENT'].lt(40)
                    | fluid['Gly_Glycol_PERCENT'].gt(60)
                )
            )
        )
    ).astype('int8')
    fluid = (
        fluid.dropna(subset=['machine_key', 'event_date'])
        .sort_values(['machine_key', 'event_date'])
        .reset_index(drop=True)
    )
    fluid_known = fluid[fluid['severity_known'].notna()].copy()
    fluid_known['prev_severity'] = fluid_known.groupby('machine_key')['severity_known'].shift(1)
    fluid_known['severity_trend'] = (
        fluid_known['severity_known'] - fluid_known['prev_severity']
    ).fillna(0).clip(-4, 4)
    print(f'[sources] fluid complete: {time.time() - load_started:.1f}s', flush=True)

    maintenance_columns = [
        'full_model', 'SERIAL', 'EVENT_NAME_ID', 'event_date',
        'is_monitor_reset', 'is_overdue', 'is_due_now',
        'remaining_hours', 'INTERVAL_HOURS',
    ]
    maintenance = pd.read_csv(
        config.MAINTENANCE_FILE,
        usecols=maintenance_columns,
        low_memory=False,
    )
    maintenance['machine_key'] = machine_key_from_model_serial(
        maintenance['full_model'], maintenance['SERIAL']
    )
    maintenance['event_date'] = pd.to_datetime(maintenance['event_date'], errors='coerce')
    for column in ['is_monitor_reset', 'is_overdue', 'is_due_now']:
        maintenance[column] = as_bool(maintenance[column])
    maintenance = (
        maintenance.dropna(subset=['machine_key', 'event_date'])
        .sort_values(['event_date', 'machine_key'])
        .reset_index(drop=True)
    )
    print(f'[sources] maintenance complete: {time.time() - load_started:.1f}s', flush=True)

    operation_columns = [
        'full_model', 'SERIAL', 'LOCAL_DATE', 'smr_hours',
        'smr_delta_clean_since_prev_obs_hours', 'engine_running_hours_clean',
        'actual_working_hours_clean', 'engine_idle_share_daily',
        'throttle_full_share_clean', 'traveling_hours_clean',
        'engine_running_day_flag', 'actual_work_day_flag',
    ]
    if include_operation_features:
        operation = pd.read_csv(
            config.OPERATION_FILE,
            usecols=operation_columns,
            low_memory=False,
        )
        # The partial export may contain a large blank tail; remove it before parsing.
        operation = operation.dropna(subset=['LOCAL_DATE']).copy()
        operation['machine_key'] = machine_key_from_model_serial(
            operation['full_model'], operation['SERIAL']
        )
        operation['event_date'] = pd.to_datetime(
            operation['LOCAL_DATE'], format='mixed', errors='coerce'
        )
        for column in operation_columns[3:]:
            operation[column] = pd.to_numeric(operation[column], errors='coerce')
        operation = (
            operation.dropna(subset=['machine_key', 'event_date'])
            .sort_values(['event_date', 'machine_key'])
            .reset_index(drop=True)
        )
        operation_first_date = operation.groupby('machine_key')['event_date'].min()
        operation_roster_machines = set(
            operation['machine_key'].dropna().unique().tolist()
        )
    else:
        # Condition-only and condition+history variants do not consume operation
        # features. Read only identity columns to preserve the complete fleet
        # roster, and materialize an empty operation table so the snapshot builder
        # emits stable zero-valued liveness columns without loading the wide file.
        roster_path = getattr(config, 'OPERATION_ROSTER_FILE', None)
        if roster_path is not None and Path(roster_path).exists():
            operation_roster = pd.read_csv(roster_path, usecols=['machine_key'])
            operation_roster_machines = set(
                normalize_machine_key(operation_roster['machine_key'])
                .dropna()
                .unique()
                .tolist()
            )
        else:
            operation_roster = pd.read_csv(
                config.OPERATION_FILE,
                usecols=['full_model', 'SERIAL'],
                low_memory=False,
            ).dropna(subset=['full_model', 'SERIAL'])
            operation_roster['machine_key'] = machine_key_from_model_serial(
                operation_roster['full_model'], operation_roster['SERIAL']
            )
            operation_roster_machines = set(
                operation_roster['machine_key'].dropna().unique().tolist()
            )
        empty_columns = list(dict.fromkeys(operation_columns + ['machine_key', 'event_date']))
        operation = pd.DataFrame({column: pd.Series(dtype='float64') for column in empty_columns})
        operation['machine_key'] = pd.Series(dtype='string')
        operation['event_date'] = pd.Series(dtype='datetime64[ns]')
        operation_first_date = pd.Series(dtype='datetime64[ns]', name='event_date')
    print(
        f'[sources] operation complete (features={include_operation_features}): '
        f'{time.time() - load_started:.1f}s',
        flush=True,
    )

    print(f'[sources] collecting fleet roster: {time.time() - load_started:.1f}s', flush=True)
    fault_machines = set(fault['machine_key'].dropna().unique().tolist())
    print(f'[sources] fault roster unique={len(fault_machines):,}: {time.time() - load_started:.1f}s', flush=True)
    fluid_machines = set(fluid['machine_key'].dropna().unique().tolist())
    maintenance_machines = set(maintenance['machine_key'].dropna().unique().tolist())
    operation_machines = operation_roster_machines
    all_machines = sorted(
        fault_machines | fluid_machines | maintenance_machines | operation_machines
    )
    print(f'[sources] fleet roster complete={len(all_machines):,}: {time.time() - load_started:.1f}s', flush=True)
    machine_index = pd.Index(all_machines, name='machine_key')

    print(f'[sources] building profile: {time.time() - load_started:.1f}s', flush=True)
    profile = {
        **target_tables.profile,
        'fault_rows': int(len(fault)),
        'fault_machines': int(fault['machine_key'].nunique()),
        'fluid_rows': int(len(fluid)),
        'fluid_machines': int(fluid['machine_key'].nunique()),
        'maintenance_rows': int(len(maintenance)),
        'maintenance_machines': int(maintenance['machine_key'].nunique()),
        'operation_populated_rows': int(len(operation)),
        'operation_features_loaded': bool(include_operation_features),
        'operation_machines': int(len(operation_roster_machines)),
        'fleet_roster_machines': int(len(machine_index)),
        'candidate_serious_codes': candidate_serious_codes,
        'raw_warranty_csv_used': config.TARGET_SOURCE == 'warranty',
    }

    print(f'[sources] source load complete: {time.time() - load_started:.1f}s', flush=True)
    return SnapshotSources(
        target_events=target_events,
        target_event_day=target_event_day,
        fault=fault,
        fluid=fluid,
        fluid_known=fluid_known,
        maintenance=maintenance,
        operation=operation,
        operation_first_date=operation_first_date,
        machine_index=machine_index,
        candidate_serious_codes=candidate_serious_codes,
        candidate_code_names=candidate_code_names,
        profile=profile,
    )


def build_snapshot_dataframe(
    sources: SnapshotSources,
    snapshot_dates: Iterable[pd.Timestamp | str],
    include_targets: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build features for the supplied scoring dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(list(snapshot_dates))).sort_values()
    machine_index = sources.machine_index
    lookback = pd.Timedelta(days=config.LOOKBACK_DAYS)
    horizon = pd.Timedelta(days=config.HORIZON_DAYS)
    frames: list[pd.DataFrame] = []
    start_time = time.time()

    for position, snapshot_date in enumerate(dates, start=1):
        feature = pd.DataFrame(index=machine_index)
        feature['snapshot_date'] = snapshot_date

        # Fault features.
        fault_window = sources.fault[
            sources.fault['event_date'].ge(snapshot_date - lookback)
            & sources.fault['event_date'].lt(snapshot_date)
        ].copy()
        if not fault_window.empty:
            aggregate = fault_window.groupby('machine_key').agg(
                fault_count_90d=('fault_code', 'size'),
                fault_log_occurrence_90d=('log_occ', 'sum'),
                fault_serious_count_90d=('is_serious', 'sum'),
                fault_serious_log_occurrence_90d=('serious_log_occ', 'sum'),
                fault_severity_weighted_90d=('severity_weight', 'sum'),
                fault_evidence_weighted_90d=('evidence_weight', 'sum'),
                fault_last_date=('event_date', 'max'),
                fault_l03_count_90d=('is_l03', 'sum'),
                fault_l04_count_90d=('is_l04', 'sum'),
                fault_distinct_codes_90d=('fault_code', 'nunique'),
            )
            feature = feature.join(aggregate, how='left')
            feature['fault_days_since_last'] = (
                snapshot_date - feature['fault_last_date']
            ).dt.days
            feature = feature.drop(columns='fault_last_date')

            serious_30 = fault_window[
                fault_window['event_date'].ge(snapshot_date - pd.Timedelta(days=30))
                & fault_window['is_serious'].eq(1)
            ]
            serious_prior_60 = fault_window[
                fault_window['event_date'].lt(snapshot_date - pd.Timedelta(days=30))
                & fault_window['is_serious'].eq(1)
            ]
            fault_7 = fault_window[
                fault_window['event_date'].ge(snapshot_date - pd.Timedelta(days=7))
            ]
            feature = feature.join(
                serious_30.groupby('machine_key').size().rename('fault_serious_30d'),
                how='left',
            )
            feature = feature.join(
                serious_prior_60.groupby('machine_key').size().rename('fault_serious_prev60d'),
                how='left',
            )
            feature = feature.join(
                fault_7.groupby('machine_key').size().rename('fault_count_7d'),
                how='left',
            )

            serious_window = fault_window[fault_window['is_serious'].eq(1)].copy()
            if not serious_window.empty:
                age_days = (snapshot_date - serious_window['event_date']).dt.days.clip(lower=0)
                serious_window['decay_score'] = (
                    serious_window['severity']
                    * serious_window['log_occ']
                    * np.power(
                        0.5,
                        age_days / config.FAULT_RECENCY_HALF_LIFE_DAYS,
                    )
                )
                feature = feature.join(
                    serious_window.groupby('machine_key')['decay_score']
                    .sum()
                    .rename('fault_severity_recency_90d'),
                    how='left',
                )
                last_serious = serious_window.groupby('machine_key')['event_date'].max()
                feature['fault_days_since_last_serious'] = (
                    snapshot_date - last_serious.reindex(machine_index)
                ).dt.days

                system_counts = serious_window.groupby(['machine_key', 'system']).size()
                feature = feature.join(
                    system_counts.groupby(level=0)
                    .size()
                    .rename('fault_distinct_serious_systems'),
                    how='left',
                )
                concentration = (
                    system_counts.groupby(level=0).max()
                    / system_counts.groupby(level=0).sum()
                ).rename('fault_single_system_concentration')
                feature = feature.join(concentration, how='left')
                feature = feature.join(
                    serious_window.groupby('machine_key')['is_hst']
                    .max()
                    .rename('fault_hst_serious_flag'),
                    how='left',
                )
                for label, pattern in [
                    ('eng', 'ENG'), ('hst', 'HST'), ('mon', 'MON'), ('we', 'WE|W/E')
                ]:
                    component_count = (
                        serious_window[
                            serious_window['system'].str.contains(
                                pattern, regex=True, na=False
                            )
                        ]
                        .groupby('machine_key')
                        .size()
                        .rename(f'fault_{label}_serious_count_90d')
                    )
                    feature = feature.join(component_count, how='left')

            code_window = fault_window[
                fault_window['is_serious'].eq(1)
                & fault_window['fault_code']
                .astype('string')
                .isin(sources.candidate_serious_codes)
            ]
            if not code_window.empty:
                code_pivot = (
                    code_window.groupby(['machine_key', 'fault_code'])
                    .size()
                    .unstack(fill_value=0)
                    .rename(columns=sources.candidate_code_names)
                )
                feature = feature.join(code_pivot, how='left')

        # Ensure a stable fault schema even when a code is absent in a month.
        for column in sources.candidate_code_names.values():
            if column not in feature.columns:
                feature[column] = 0.0
        expected_fault_columns = [
            'fault_count_90d', 'fault_log_occurrence_90d',
            'fault_serious_count_90d', 'fault_serious_log_occurrence_90d',
            'fault_severity_weighted_90d', 'fault_evidence_weighted_90d',
            'fault_l03_count_90d', 'fault_l04_count_90d',
            'fault_distinct_codes_90d', 'fault_serious_30d',
            'fault_serious_prev60d', 'fault_count_7d',
            'fault_severity_recency_90d', 'fault_distinct_serious_systems',
            'fault_hst_serious_flag', 'fault_eng_serious_count_90d',
            'fault_hst_serious_count_90d', 'fault_mon_serious_count_90d',
            'fault_we_serious_count_90d',
        ]
        for column in expected_fault_columns:
            if column not in feature.columns:
                feature[column] = 0.0
        zero_fault_columns = [
            column
            for column in feature.columns
            if column.startswith('fault_')
            and column not in {
                'fault_days_since_last',
                'fault_days_since_last_serious',
                'fault_single_system_concentration',
            }
        ]
        feature[zero_fault_columns] = feature[zero_fault_columns].fillna(0)
        feature['fault_days_since_last'] = (
            feature.get(
                'fault_days_since_last', pd.Series(index=machine_index, dtype=float)
            )
            .fillna(config.LOOKBACK_DAYS + 1)
            .clip(0, config.LOOKBACK_DAYS + 1)
        )
        feature['fault_days_since_last_serious'] = (
            feature.get(
                'fault_days_since_last_serious',
                pd.Series(index=machine_index, dtype=float),
            )
            .fillna(config.LOOKBACK_DAYS + 1)
            .clip(0, config.LOOKBACK_DAYS + 1)
        )
        feature['fault_single_system_concentration'] = feature.get(
            'fault_single_system_concentration',
            pd.Series(index=machine_index, dtype=float),
        ).fillna(0)
        recent_serious = feature['fault_serious_30d']
        prior_serious = feature['fault_serious_prev60d']
        feature['fault_velocity_ratio'] = (
            (recent_serious + 1.0) / (prior_serious / 2.0 + 1.0)
        ).clip(0, 20)
        feature.loc[
            (recent_serious + prior_serious).eq(0), 'fault_velocity_ratio'
        ] = 0
        feature['has_accelerating_faults'] = (
            recent_serious.ge(3)
            & recent_serious.gt(1.5 * (prior_serious / 2.0 + 0.5))
        ).astype('int8')
        feature = feature.drop(columns='fault_serious_prev60d')

        # Fluid LOCF features.
        prior_known = sources.fluid_known[
            sources.fluid_known['event_date'].lt(snapshot_date)
        ]
        if not prior_known.empty:
            latest_known = (
                prior_known.groupby('machine_key', sort=False)
                .tail(1)
                .set_index('machine_key')
            )
            staleness = (snapshot_date - latest_known['event_date']).dt.days
            active = staleness.le(config.FLUID_LOCF_EXPIRY_DAYS)
            active_aligned = active.reindex(machine_index).fillna(False)
            feature['fluid_current_severity'] = (
                latest_known['severity_known']
                .reindex(machine_index)
                .where(active_aligned, 0)
            )
            feature['fluid_worsening_trend'] = (
                latest_known['severity_trend']
                .reindex(machine_index)
                .where(active_aligned, 0)
            )
            feature['fluid_sample_staleness_days'] = (
                staleness.reindex(machine_index)
                .fillna(config.FLUID_LOCF_EXPIRY_DAYS + 1)
                .clip(0, config.FLUID_LOCF_EXPIRY_DAYS + 1)
            )
        else:
            feature['fluid_current_severity'] = 0
            feature['fluid_worsening_trend'] = 0
            feature['fluid_sample_staleness_days'] = config.FLUID_LOCF_EXPIRY_DAYS + 1

        recent_abnormal = sources.fluid_known[
            sources.fluid_known['event_date'].ge(snapshot_date - lookback)
            & sources.fluid_known['event_date'].lt(snapshot_date)
            & sources.fluid_known['is_abnormal'].eq(1)
        ]
        feature['fluid_recent_abnormal_count_90d'] = (
            recent_abnormal.groupby('machine_key')
            .size()
            .reindex(machine_index)
            .fillna(0)
        )
        prior_fluid = sources.fluid[sources.fluid['event_date'].lt(snapshot_date)]
        if not prior_fluid.empty:
            latest_fluid = (
                prior_fluid.groupby('machine_key', sort=False)
                .tail(1)
                .set_index('machine_key')
            )
            any_staleness = (snapshot_date - latest_fluid['event_date']).dt.days
            any_active = any_staleness.le(config.FLUID_LOCF_EXPIRY_DAYS)
            any_active_aligned = any_active.reindex(machine_index).fillna(False)
            feature['fluid_contaminant_flag'] = (
                latest_fluid['contaminant_flag']
                .reindex(machine_index)
                .where(any_active_aligned, 0)
            )
            feature['fluid_any_sample_staleness_days'] = (
                any_staleness.reindex(machine_index)
                .fillna(config.FLUID_LOCF_EXPIRY_DAYS + 1)
                .clip(0, config.FLUID_LOCF_EXPIRY_DAYS + 1)
            )
            feature['fluid_sample_known_flag'] = any_active_aligned.astype('int8')
            for column in [
                'Water_Water_PERCENT', 'Gly_Glycol_PERCENT',
                'EthyleneGlycol_Ethylene_Glycol_PERCENT', 'Fuel_Fuel_PERCENT',
                'Si_Silicon_PPM', 'Fe_Iron_PPM', 'Cu_Copper_PPM',
            ]:
                feature[f'fluid_locf_{safe_name(column).lower()}'] = (
                    latest_fluid[column]
                    .reindex(machine_index)
                    .where(any_active_aligned, np.nan)
                )
        else:
            feature['fluid_contaminant_flag'] = 0
            feature['fluid_any_sample_staleness_days'] = config.FLUID_LOCF_EXPIRY_DAYS + 1
            feature['fluid_sample_known_flag'] = 0
        for column in [
            'fluid_current_severity', 'fluid_worsening_trend',
            'fluid_contaminant_flag',
        ]:
            feature[column] = pd.to_numeric(feature[column], errors='coerce').fillna(0)

        # Preventative-maintenance features.
        reset_rows = sources.maintenance[
            sources.maintenance['event_date'].lt(snapshot_date)
            & sources.maintenance['is_monitor_reset']
        ]
        if not reset_rows.empty:
            last_reset = reset_rows.groupby('machine_key')['event_date'].max()
            feature['pm_days_since_last_reset'] = (
                snapshot_date - last_reset.reindex(machine_index)
            ).dt.days.fillna(1096).clip(0, 1096)
        else:
            feature['pm_days_since_last_reset'] = 1096
        maintenance_window = sources.maintenance[
            sources.maintenance['event_date'].ge(snapshot_date - lookback)
            & sources.maintenance['event_date'].lt(snapshot_date)
        ]
        if not maintenance_window.empty:
            feature['pm_overdue_count_90d'] = (
                maintenance_window[maintenance_window['is_overdue']]
                .groupby('machine_key')['EVENT_NAME_ID']
                .nunique()
                .reindex(machine_index)
                .fillna(0)
            )
            feature['pm_due_count_90d'] = (
                maintenance_window[maintenance_window['is_due_now']]
                .groupby('machine_key')['EVENT_NAME_ID']
                .nunique()
                .reindex(machine_index)
                .fillna(0)
            )
            feature['pm_reset_count_90d'] = (
                maintenance_window[maintenance_window['is_monitor_reset']]
                .groupby('machine_key')['EVENT_NAME_ID']
                .nunique()
                .reindex(machine_index)
                .fillna(0)
            )
            feature['pm_record_count_90d'] = (
                maintenance_window.groupby('machine_key')
                .size()
                .reindex(machine_index)
                .fillna(0)
            )
        else:
            for column in [
                'pm_overdue_count_90d', 'pm_due_count_90d',
                'pm_reset_count_90d', 'pm_record_count_90d',
            ]:
                feature[column] = 0

        # Operation and liveness features. They are gate-only in the strict base model
        # and become model candidates in enhanced variants.
        operation_prior = sources.operation[
            sources.operation['event_date'].lt(snapshot_date)
        ]
        if not operation_prior.empty:
            last_sensor = operation_prior.groupby('machine_key')['event_date'].max()
            feature['days_since_last_sensor_reading'] = (
                snapshot_date - last_sensor.reindex(machine_index)
            ).dt.days.fillna(366).clip(0, 366)
            feature['operation_history_days'] = (
                snapshot_date - sources.operation_first_date.reindex(machine_index)
            ).dt.days.fillna(0).clip(lower=0)
        else:
            feature['days_since_last_sensor_reading'] = 366
            feature['operation_history_days'] = 0

        for days in [30, 90]:
            operation_window = sources.operation[
                sources.operation['event_date'].ge(
                    snapshot_date - pd.Timedelta(days=days)
                )
                & sources.operation['event_date'].lt(snapshot_date)
            ]
            if not operation_window.empty:
                operation_aggregate = operation_window.groupby('machine_key').agg(
                    **{
                        f'smr_delta_{days}d': (
                            'smr_delta_clean_since_prev_obs_hours', 'sum'
                        ),
                        f'engine_hours_{days}d': ('engine_running_hours_clean', 'sum'),
                        f'work_hours_{days}d': ('actual_working_hours_clean', 'sum'),
                        f'active_days_{days}d': ('engine_running_day_flag', 'sum'),
                        f'travel_hours_{days}d': ('traveling_hours_clean', 'sum'),
                        f'idle_share_mean_{days}d': ('engine_idle_share_daily', 'mean'),
                        f'throttle_share_mean_{days}d': (
                            'throttle_full_share_clean', 'mean'
                        ),
                        f'operation_records_{days}d': ('event_date', 'size'),
                    }
                )
                feature = feature.join(operation_aggregate, how='left')
            for column in [
                f'smr_delta_{days}d', f'engine_hours_{days}d',
                f'work_hours_{days}d', f'active_days_{days}d',
                f'travel_hours_{days}d', f'operation_records_{days}d',
            ]:
                if column not in feature.columns:
                    feature[column] = 0
                feature[column] = feature[column].fillna(0)
        feature['utilization_velocity_ratio'] = (
            (feature['engine_hours_30d'] + 1)
            / (feature['engine_hours_90d'] / 3 + 1)
        ).clip(0, 10)
        feature['is_recently_live'] = (
            feature['days_since_last_sensor_reading'].le(21)
        ).astype('int8')

        # Prior target-event history is available at scoring time. It is excluded
        # from base27 and included only in the optional history variants.
        prior_events = sources.target_event_day[
            sources.target_event_day['event_date'].lt(snapshot_date)
        ]
        if not prior_events.empty:
            last_event = prior_events.groupby('machine_key')['event_date'].max()
            feature['days_since_prior_target_event'] = (
                snapshot_date - last_event.reindex(machine_index)
            ).dt.days.fillna(1096).clip(0, 1096)
        else:
            feature['days_since_prior_target_event'] = 1096
        for days in [90, 365, 730]:
            prior_window = sources.target_event_day[
                sources.target_event_day['event_date'].ge(
                    snapshot_date - pd.Timedelta(days=days)
                )
                & sources.target_event_day['event_date'].lt(snapshot_date)
            ]
            feature[f'prior_target_event_count_{days}d'] = (
                prior_window.groupby('machine_key')
                .size()
                .reindex(machine_index)
                .fillna(0)
            )

        if include_targets:
            future = sources.target_event_day[
                sources.target_event_day['event_date'].ge(snapshot_date)
                & sources.target_event_day['event_date'].lt(snapshot_date + horizon)
            ]
            future_count = future.groupby('machine_key').size()
            feature[config.FUTURE_TARGET_COUNT_COLUMN] = (
                future_count.reindex(machine_index).fillna(0).astype('int16')
            )
            feature[config.TARGET_COLUMN] = (
                feature[config.FUTURE_TARGET_COUNT_COLUMN].gt(0)
            ).astype('int8')

            # Optional source-breakout labels are diagnostic only and never model
            # features. For the original warranty target this produces only the
            # WARRANTY breakout; the physical target may also contain service/TSI.
            for source_name in ['WARRANTY', 'SERVICE_REPAIR', 'TSI']:
                source_future = sources.target_events[
                    sources.target_events['event_source']
                    .astype('string')
                    .str.upper()
                    .eq(source_name)
                    & sources.target_events['event_date'].ge(snapshot_date)
                    & sources.target_events['event_date'].lt(snapshot_date + horizon)
                ]
                feature[f'target_{source_name.lower()}_0_90'] = (
                    feature.index.isin(source_future['machine_key'].unique())
                ).astype('int8')

        feature['has_fault_evidence_90d'] = feature['fault_count_90d'].gt(0).astype('int8')
        feature['has_fluid_evidence_365d'] = feature['fluid_sample_known_flag'].gt(0).astype('int8')
        feature['has_pm_evidence_90d'] = feature['pm_record_count_90d'].gt(0).astype('int8')
        feature['has_operation_evidence_30d'] = feature['operation_records_30d'].gt(0).astype('int8')
        feature['source_evidence_count'] = feature[
            [
                'has_fault_evidence_90d', 'has_fluid_evidence_365d',
                'has_pm_evidence_90d', 'has_operation_evidence_30d',
            ]
        ].sum(axis=1)

        feature = feature.reset_index()
        frames.append(feature)
        if verbose:
            positive_text = ''
            if include_targets:
                positive_text = f", positives={int(feature[config.TARGET_COLUMN].sum()):,}"
            print(
                f'[{position:02d}/{len(dates):02d}] {snapshot_date.date()} '
                f'rows={len(feature):,}{positive_text}, '
                f'elapsed={time.time() - start_time:.1f}s',
                flush=True,
            )

    dataframe = pd.concat(frames, ignore_index=True)
    dataframe['full_model'] = dataframe['machine_key'].str.replace(
        r'-\d+$', '', regex=True
    )
    for column in dataframe.columns:
        if column not in {'machine_key', 'snapshot_date', 'full_model'}:
            dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
    # Code counts are structural zeros before a code first appears.
    code_columns = list(sources.candidate_code_names.values())
    dataframe[code_columns] = dataframe[code_columns].fillna(0)
    return dataframe
