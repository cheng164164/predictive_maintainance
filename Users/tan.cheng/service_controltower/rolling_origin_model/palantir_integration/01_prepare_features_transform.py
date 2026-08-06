"""Palantir lightweight transform that builds the current machine snapshot."""
from __future__ import annotations

from transforms.api import Input, LightweightInput, LightweightOutput, Output, transform

from .data_preparation_core import build_feature_snapshot_from_frames, infer_score_date
from .foundry_paths import (
    FAULT_DATASET,
    FLUID_DATASET,
    MACHINE_ROSTER_DATASET,
    MAINTENANCE_DATASET,
    OPERATION_DATASET,
    PALANTIR_SCORE_DATE,
    PREPARED_FEATURES_DATASET,
    TARGET_HISTORY_DATASET,
)


@transform.using(
    fault=Input(FAULT_DATASET),
    fluid=Input(FLUID_DATASET),
    maintenance=Input(MAINTENANCE_DATASET),
    operation=Input(OPERATION_DATASET),
    target_history=Input(TARGET_HISTORY_DATASET),
    machine_roster=Input(MACHINE_ROSTER_DATASET),
    prepared_features=Output(PREPARED_FEATURES_DATASET),
)
def compute(
    fault: LightweightInput,
    fluid: LightweightInput,
    maintenance: LightweightInput,
    operation: LightweightInput,
    target_history: LightweightInput,
    machine_roster: LightweightInput,
    prepared_features: LightweightOutput,
) -> None:
    """Read Foundry sources, build one snapshot, and write model-ready features."""
    operation_frame = operation.pandas()
    score_date = infer_score_date(operation_frame, override=PALANTIR_SCORE_DATE)
    snapshot = build_feature_snapshot_from_frames(
        source_frames={
            "fault": fault.pandas(),
            "fluid": fluid.pandas(),
            "maintenance": maintenance.pandas(),
            "operation": operation_frame,
            "target": target_history.pandas(),
        },
        machine_roster=machine_roster.pandas(),
        score_date=score_date,
    )
    prepared_features.write_pandas(snapshot)
