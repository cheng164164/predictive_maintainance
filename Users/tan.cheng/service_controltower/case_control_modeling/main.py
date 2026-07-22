"""Run the standard validation-only case-control modeling workflow.

This main workflow builds the source profiles, claim episodes, case-control
windows, cross-validation diagnostics, and validation report. It does not run
Phase 1 design sweep, hyperparameter tuning, or final test evaluation.
"""
from __future__ import annotations

import importlib
import traceback

STEPS = [
    "00_profile_sources",
    "01_build_claim_episodes",
    "02_build_case_control_dataset",
    "03_cross_validation",
    "04_fit_validate_model_report",
]


def main() -> None:
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
