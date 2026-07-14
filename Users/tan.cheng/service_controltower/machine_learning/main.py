"""Run the standalone machine-learning workflow end to end."""
from __future__ import annotations

import importlib
import traceback

STEPS = [
    ("00_split_data", "00_split_data"),
    ("01_cross_validation", "01_cross_validation"),
    ("02_train_validate_test", "02_train_validate_test"),
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
    print("\nMachine-learning workflow completed.")


if __name__ == "__main__":
    main()
