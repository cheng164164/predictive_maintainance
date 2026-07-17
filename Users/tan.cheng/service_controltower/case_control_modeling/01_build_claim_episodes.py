"""Step 01: build warranty claim episodes."""
from __future__ import annotations

import pandas as pd

import config
from cc_utils import build_claim_episodes, ensure_dir, load_warranty, write_json


def run() -> None:
    step_dir = config.OUTPUT_DIR / "01_claim_episodes"
    ensure_dir(step_dir)

    warranty = load_warranty(config)
    episodes = build_claim_episodes(warranty, gap_days=config.CLAIM_EPISODE_GAP_DAYS)

    warranty.to_csv(step_dir / "cleaned_warranty_claims.csv", index=False)
    episodes.to_csv(step_dir / "claim_episodes.csv", index=False)

    summary = {
        "step": "01_build_claim_episodes",
        "output_dir": str(step_dir),
        "raw_or_filtered_claim_rows": int(len(warranty)),
        "claim_episode_rows": int(len(episodes)),
        "unique_claim_machines": int(episodes["machine_key"].nunique(dropna=True)) if len(episodes) else 0,
        "claim_episode_gap_days": int(config.CLAIM_EPISODE_GAP_DAYS),
        "keep_only_valid_critical_part_claims": bool(config.KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS),
        "claim_date_min": episodes["claim_date"].min() if len(episodes) else None,
        "claim_date_max": episodes["claim_date"].max() if len(episodes) else None,
    }
    write_json(summary, step_dir / "run_summary.json")

    print(f"01_build_claim_episodes completed. Outputs: {step_dir}")
    print(f"  claim rows: {len(warranty):,}")
    print(f"  claim episodes: {len(episodes):,}")


if __name__ == "__main__":
    run()
