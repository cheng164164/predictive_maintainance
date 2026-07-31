"""Shared data utilities."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize free-text values while preserving missing values."""

    return series.astype('string').str.strip().str.upper()


def clean_serial(series: pd.Series) -> pd.Series:
    """Normalize serial numbers for stable machine-key construction."""

    return clean_text(series).str.replace(r'\.0$', '', regex=True)


def make_machine_key(model: pd.Series, serial: pd.Series) -> pd.Series:
    """Combine normalized machine model and serial into an audit identifier."""

    model_clean = clean_text(model)
    serial_clean = clean_serial(serial)
    key = model_clean + '|' + serial_clean
    invalid = model_clean.isna() | serial_clean.isna() | model_clean.eq('') | serial_clean.eq('')
    return key.mask(invalid)


def four_month_period_start(dates: pd.Series) -> pd.Series:
    """Map dates to fixed January/May/September four-month period starts."""

    dates = pd.to_datetime(dates, errors='coerce')
    months = ((dates.dt.month - 1) // 4) * 4 + 1
    return pd.to_datetime({'year': dates.dt.year, 'month': months, 'day': 1}, errors='coerce')


def period_end_from_start(period_start: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
    """Return the inclusive final date for a configured segment start."""

    return pd.to_datetime(period_start) + pd.DateOffset(months=4) - pd.Timedelta(days=1)


def parse_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean text and numeric encodings to zero or one."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    normalized = series.astype('string').str.strip().str.lower()
    return normalized.map({'true': 1.0, 'false': 0.0, '1': 1.0, '0': 0.0, 'yes': 1.0, 'no': 0.0})


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two numeric series while replacing invalid results with missing values."""

    numerator = pd.to_numeric(numerator, errors='coerce')
    denominator = pd.to_numeric(denominator, errors='coerce')
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def sanitize_category(value: object) -> str:
    """Convert category values to safe lowercase feature-name tokens."""

    text = str(value).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text or 'missing'


def add_category_indicators(
    df: pd.DataFrame,
    column: str,
    prefix: str,
    expected_values: Iterable[object] | None = None,
) -> list[str]:
    """Add count columns for configured category values before aggregation."""

    values = list(expected_values) if expected_values is not None else sorted(df[column].dropna().unique().tolist())
    output_cols: list[str] = []
    normalized = df[column].astype('string').str.strip().str.upper()
    for value in values:
        col = f'{prefix}_{sanitize_category(value)}_count'
        df[col] = normalized.eq(str(value).strip().upper()).fillna(False).astype('int16')
        output_cols.append(col)
    return output_cols


def date_filter(df: pd.DataFrame, date_col: str, start: str, end: str) -> pd.DataFrame:
    """Parse a date column and keep rows inside the configured inclusive range."""

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[df[date_col].between(pd.Timestamp(start), pd.Timestamp(end))].copy()
