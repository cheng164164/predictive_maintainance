"""Step 11: score the newest exact 90-day segment for the full fleet."""
from __future__ import annotations

import joblib
import pandas as pd

import config
from modeling_90d import apply_platt, rank_predictions, score_model


def main() -> None:
    config.STEP_11_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(config.FULL_DATASET_PATH)
    artifact = joblib.load(config.FINAL_MODEL_ARTIFACT_PATH)
    latest_start = pd.Timestamp(data["period_start"].max())
    latest = data[data["period_start"].eq(latest_start)].copy()
    raw = score_model(artifact["preprocessor"], artifact["model"], latest, artifact["features"])
    calibrated = apply_platt(artifact["calibrator"], raw)
    ranked = rank_predictions(latest, raw, calibrated, float(artifact["threshold"]))
    ranked.to_csv(config.STEP_11_OUTPUT_DIR / "latest_ranked_fleet.csv.gz", index=False, compression="gzip")
    compact = [
        "machine_key", "full_model", "period_start", "period_end", "true_label",
        "predicted_label", "prediction_outcome", "raw_score", "calibrated_probability",
        "risk_index", "rank_within_anchor", "target_first_claim_date_90d",
        "days_to_next_target_event",
    ] + [f"selected_top_{int(rate*100)}pct" for rate in config.TOP_K_RATES] + [
        f"selected_top_n_{count}" for count in config.TOP_N_COUNTS
    ]
    ranked[[c for c in compact if c in ranked.columns]].to_csv(
        config.STEP_11_OUTPUT_DIR / "latest_ranked_fleet_compact.csv", index=False
    )
    print(f"Latest segment start: {latest_start.date()}")
    print(ranked.head(20)[["machine_key", "calibrated_probability", "risk_index", "true_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
