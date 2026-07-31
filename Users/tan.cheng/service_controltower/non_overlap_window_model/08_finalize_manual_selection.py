"""Step 08: copy the config-selected calibrated model to the final artifact."""
from __future__ import annotations

import shutil

import config


def main() -> None:
    config.STEP_08_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.SELECTED_MODEL_ARTIFACT_PATH, config.FINAL_MODEL_ARTIFACT_PATH)
    (config.STEP_08_OUTPUT_DIR / "final_model_path.txt").write_text(
        str(config.FINAL_MODEL_ARTIFACT_PATH), encoding="utf-8"
    )
    print(f"Final model: {config.FINAL_MODEL_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
