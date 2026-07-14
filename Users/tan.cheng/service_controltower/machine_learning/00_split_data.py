"""Step 00: create chronological train / validation / test split reports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from ml_utils import (
    chronological_split,
    ensure_dir,
    make_split_summary,
    read_snapshot,
    resolve_input_path,
    source_features_for_prepared_features,
    validate_source_features,
    write_json,
)



def _ordered_unique(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def _export_reduced_snapshot_dataframe(df, split, step_dir):
    """Optionally save a source-level snapshot dataframe for one model variant."""

    if not getattr(config, "SAVE_REDUCED_SNAPSHOT_DATAFRAME", False):
        return None

    variant = str(getattr(config, "REDUCED_SNAPSHOT_MODEL_VARIANT", "D"))
    if variant not in config.FEATURE_SETS:
        raise ValueError(
            f"REDUCED_SNAPSHOT_MODEL_VARIANT='{variant}' is not defined in config.FEATURE_SETS."
        )

    selected_prepared = config.FEATURE_SETS[variant]
    required_source_features = source_features_for_prepared_features(
        selected_prepared,
        config.PREPARED_TO_SOURCE_FEATURE,
    )
    present_source_features, missing_source_features = validate_source_features(
        df=df,
        source_features=required_source_features,
        error_on_missing=config.ERROR_ON_MISSING_SOURCE_FEATURES,
    )

    base_cols = _ordered_unique(
        list(config.ID_COLS or [])
        + [config.DATE_COL, config.TARGET_COL]
        + list(getattr(config, "SECONDARY_SORT_COLS", []) or [])
    )
    base_cols = [c for c in base_cols if c in df.columns]
    export_cols = _ordered_unique(base_cols + present_source_features)

    pieces = []
    for split_name, split_df in [
        ("training_main", split.train),
        ("validation_holdout", split.validation),
        ("test_holdout", split.test),
    ]:
        part = split_df[export_cols].copy()
        if getattr(config, "REDUCED_SNAPSHOT_INCLUDE_SPLIT_COLUMN", True):
            part.insert(len(base_cols), "split", split_name)
        pieces.append(part)

    reduced_df = pd.concat(pieces, ignore_index=True)

    output_filename = getattr(
        config,
        "REDUCED_SNAPSHOT_OUTPUT_FILENAME",
        f"06_snapshot_dataframe_model_{variant}_reduced.csv",
    )
    output_path = step_dir / output_filename
    reduced_df.to_csv(output_path, index=False)

    metadata = {
        "model_variant": variant,
        "output_path": str(output_path),
        "rows": int(len(reduced_df)),
        "columns": int(len(reduced_df.columns)),
        "selected_prepared_feature_count": int(len(selected_prepared)),
        "required_source_feature_count": int(len(required_source_features)),
        "present_source_feature_count": int(len(present_source_features)),
        "missing_source_feature_count": int(len(missing_source_features)),
        "missing_source_features": missing_source_features,
        "base_columns": base_cols,
        "source_feature_columns": present_source_features,
        "include_split_column": bool(getattr(config, "REDUCED_SNAPSHOT_INCLUDE_SPLIT_COLUMN", True)),
    }
    write_json(metadata, step_dir / "06_snapshot_dataframe_model_reduced_metadata.json")

    print(
        f"Reduced source snapshot for Model {variant} saved: {output_path} "
        f"({len(reduced_df)} rows, {len(reduced_df.columns)} columns)"
    )
    return metadata

def run() -> None:
    step_dir = config.OUTPUT_DIR / "00_data_split"
    ensure_dir(step_dir)

    resolved_input = resolve_input_path(config.INPUT_DATA_PATH)
    df = read_snapshot(config.INPUT_DATA_PATH, config.DATE_COL)

    source_cols_to_drop = [
        c for c in getattr(config, "SOURCE_COLUMNS_TO_DROP_BEFORE_MODELING", []) if c in df.columns
    ]
    if source_cols_to_drop:
        df = df.drop(columns=source_cols_to_drop)

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL='{config.TARGET_COL}' was not found in input data.")

    split, effective_ratios = chronological_split(
        df=df,
        date_col=config.DATE_COL,
        train_ratio=config.TRAIN_RATIO,
        validation_ratio=config.VALIDATION_RATIO,
        test_ratio=config.TEST_RATIO,
        secondary_sort_cols=config.SECONDARY_SORT_COLS,
    )

    split_summary = make_split_summary(split, config.TARGET_COL, config.DATE_COL, config.ID_COLS)
    split_summary.to_csv(step_dir / "01_split_summary.csv", index=False)
    split.split_assignments.to_csv(step_dir / "02_split_assignments.csv", index=False)

    feature_set_summary = []
    feature_set_detail_rows = []
    for variant, features in config.FEATURE_SETS.items():
        sources = []
        for feature_idx, feature in enumerate(features, start=1):
            source = config.PREPARED_TO_SOURCE_FEATURE[feature]
            if source not in sources:
                sources.append(source)

            feature_set_detail_rows.append(
                {
                    "model_variant": variant,
                    "feature_index": feature_idx,
                    "prepared_feature": feature,
                    "source_feature": source,
                    "is_machine_context_feature": bool(
                        feature in set(getattr(config, "PROTECTED_MACHINE_CONTEXT_PREPARED_FEATURES", []))
                    ),
                    "included_in_model_A": feature in set(config.FEATURE_SETS.get("A", [])),
                    "included_in_model_B": feature in set(config.FEATURE_SETS.get("B", [])),
                    "included_in_model_C": feature in set(config.FEATURE_SETS.get("C", [])),
                    "included_in_model_D": feature in set(config.FEATURE_SETS.get("D", [])),
                }
            )

        missing_sources = [c for c in sources if c not in df.columns]
        feature_set_summary.append(
            {
                "model_variant": variant,
                "selected_prepared_feature_count": len(features),
                "required_source_feature_count": len(sources),
                "missing_required_source_feature_count": len(missing_sources),
                "missing_required_source_features": ";".join(missing_sources),
            }
        )

    pd.DataFrame(feature_set_summary).to_csv(step_dir / "03_feature_set_summary.csv", index=False)

    feature_set_detail_df = pd.DataFrame(feature_set_detail_rows)
    feature_set_detail_df.to_csv(step_dir / "04_feature_sets_prepared_features.csv", index=False)

    if "D" in config.FEATURE_SETS:
        model_d_features_df = feature_set_detail_df[
            feature_set_detail_df["model_variant"] == "D"
        ].copy()
        model_d_features_df.insert(
            5,
            "model_d_feature_source_reason",
            model_d_features_df.apply(
                lambda row: "retained_from_model_C"
                if row["prepared_feature"] in set(config.FEATURE_SETS.get("C", []))
                else "added_back_machine_context",
                axis=1,
            ),
        )
        model_d_features_df.to_csv(step_dir / "05_model_D_prepared_features.csv", index=False)
        print(
            f"Model D prepared feature list saved: "
            f"{step_dir / '05_model_D_prepared_features.csv'} "
            f"({len(model_d_features_df)} features)"
        )


    reduced_snapshot_metadata = _export_reduced_snapshot_dataframe(df, split, step_dir)

    write_json(
        {
            "step": "00_split_data",
            "input_data_path_configured": str(config.INPUT_DATA_PATH),
            "input_data_path_resolved": str(resolved_input),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target_col": config.TARGET_COL,
            "date_col": config.DATE_COL,
            "configured_split_ratios": {
                "train": config.TRAIN_RATIO,
                "validation": config.VALIDATION_RATIO,
                "test": config.TEST_RATIO,
            },
            "effective_normalized_split_ratios": effective_ratios,
            "source_columns_dropped_before_modeling": source_cols_to_drop,
            "reduced_snapshot_export_enabled": bool(getattr(config, "SAVE_REDUCED_SNAPSHOT_DATAFRAME", False)),
            "reduced_snapshot_export": reduced_snapshot_metadata,
            "output_dir": str(step_dir),
        },
        step_dir / "00_run_summary.json",
    )
    print(f"00_split_data completed. Outputs: {step_dir}")


if __name__ == "__main__":
    run()
