"""Run the window-based case-control modeling workflow end to end."""
from __future__ import annotations

import importlib
import traceback

STEPS = [
    ("00_profile_sources", "00_profile_sources"),
    ("01_build_claim_episodes", "01_build_claim_episodes"),
    ("02_build_case_control_dataset", "02_build_case_control_dataset"),
    ("03_smoke_run", "03_smoke_run"),
    ("04_cross_validation", "04_cross_validation"),
    ("05_validation_prediction_report", "05_validation_prediction_report"),
]


def main() -> None:
    for label, module_name in STEPS:
        print("\n" + "=" * 80)
        print(f"Running {label}")
        print("=" * 80)
        module = importlib.import_module(module_name)
        try:
            module.run()
        except Exception:
            print(f"Step failed: {label}")
            traceback.print_exc()
            raise
    print("\nWindow-based case-control workflow completed.")


if __name__ == "__main__":
    main()
