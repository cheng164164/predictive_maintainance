"""Create a stable customer/UI projection from full model prediction output."""
from __future__ import annotations

import pandas as pd
from transforms.api import Input, LightweightInput, LightweightOutput, Output, transform

from .foundry_paths import MODEL_PREDICTIONS_DATASET, UI_RISK_OUTPUT_DATASET

UI_COLUMNS = (
    "machine_key",
    "full_model",
    "snapshot_date",
    "raw_model_score",
    "failure_probability",
    "risk_index",
    "risk_tier",
    "candidate_risk_tier",
    "risk_rank_within_score_date",
    "model_flag_at_threshold",
    "tier_decision_reason",
    "top_risk_factors",
)


@transform.using(
    predictions=Input(MODEL_PREDICTIONS_DATASET),
    ui_output=Output(UI_RISK_OUTPUT_DATASET),
)
def compute(
    predictions: LightweightInput,
    ui_output: LightweightOutput,
) -> None:
    """Validate, order, and format predictions for downstream customer use."""
    frame = predictions.pandas()
    missing = sorted(set(UI_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Model prediction dataset is missing UI columns: {missing}")
    output = frame[list(UI_COLUMNS)].copy()
    output["snapshot_date"] = pd.to_datetime(
        output["snapshot_date"], errors="raise"
    ).dt.date.astype(str)
    output["risk_index_display"] = output["risk_index"].round(2)
    output["failure_probability_pct_display"] = (
        100.0 * output["failure_probability"]
    ).round(2)
    tier_order = pd.Categorical(
        output["risk_tier"],
        categories=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        ordered=True,
    )
    output = (
        output.assign(_tier_order=tier_order)
        .sort_values(
            ["_tier_order", "risk_index", "machine_key"],
            ascending=[True, False, True],
        )
        .drop(columns="_tier_order")
        .reset_index(drop=True)
    )
    ui_output.write_pandas(output)
