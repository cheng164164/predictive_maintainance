"""Step 04: freeze the configured reviewed feature list."""
from __future__ import annotations

import pandas as pd

import config
from modeling_90d import feature_list


def main() -> None:
    config.STEP_04_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.FULL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Run Step 03 first: {config.FULL_DATASET_PATH}")
    data = pd.read_pickle(config.FULL_DATASET_PATH)
    features = feature_list(data, config.MODEL_VARIANT)
    table = pd.DataFrame(
        {
            "feature_order": range(1, len(features) + 1),
            "feature": features,
            "model_variant": config.MODEL_VARIANT,
            "target_source": config.TARGET_SOURCE,
        }
    )
    table.to_csv(config.MODEL_FEATURE_LIST_PATH, index=False)
    print(f"Selected {len(features)} features for {config.MODEL_VARIANT}.")
    print(f"Feature list: {config.MODEL_FEATURE_LIST_PATH}")


if __name__ == "__main__":
    main()
