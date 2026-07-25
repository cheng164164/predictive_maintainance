"""Run the standard fixed-split case-control modeling workflow.

This builds source profiles, claim episodes, fixed machine-level
train/validation/test datasets, training-only cross-validation diagnostics, and
the validation report. It does not score the test split.
"""
from __future__ import annotations

import importlib
import traceback

import config

STEPS = [
    "00_profile_sources",
    "01_build_claim_episodes",
    "02_build_case_control_dataset",
    "03_cross_validation",
    "04_fit_validate_model_report",
]


def main() -> None:
    config.refresh_derived_config()
    for i, module_name in enumerate(STEPS, start=1):
        print("\n" + "=" * 88)
        print(f"Running workflow step {i}/{len(STEPS)}: {module_name}")
        print("=" * 88, flush=True)
        module = importlib.import_module(module_name)
        module = importlib.reload(module)
        try:
            module.run()
        except Exception:
            print(f"Step failed: {module_name}")
            traceback.print_exc()
            raise
    print("\nStandard validation workflow completed.")


if __name__ == "__main__":
    main()
