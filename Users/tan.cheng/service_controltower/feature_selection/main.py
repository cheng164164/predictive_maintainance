"""
Run all feature-selection steps in order.

This file is kept as a convenience option. To inspect each method step by step,
run the numbered scripts individually instead:

    python 00_prepare_data.py
    python 01_unsupervised_selection.py
    python 02_correlation_analysis.py
    python 03_statistical_tests.py
    python 04_xgboost_importance.py
    python 05_permutation_importance.py
    python 06_shap_analysis.py
    python 07_consensus_report.py
"""
from workflow import run_all_steps


if __name__ == "__main__":
    run_all_steps()
