"""Step 01: build claim episodes from the cleaned warranty source."""
from __future__ import annotations

import config
from cc_utils import build_claim_episodes, ensure_dir, load_warranty, write_json


def run() -> None:
    config.refresh_derived_config()
    step_dir = config.OUTPUT_DIR / "01_claim_episodes"
    ensure_dir(step_dir)

    # Claim construction needs only the warranty source.
    warranty, filter_audit = load_warranty(
        config, return_filter_audit=True
    )
    warranty = warranty.copy()
    episodes = build_claim_episodes(
        warranty, gap_days=config.CLAIM_EPISODE_GAP_DAYS
    )

    warranty.to_csv(step_dir / "cleaned_warranty_claims.csv", index=False)
    filter_audit.to_csv(step_dir / "warranty_claim_filter_audit.csv", index=False)
    filter_summary = (
        filter_audit.groupby(
            [
                "warranty_claim_filter_mode",
                "warranty_claim_filter_keep",
                "warranty_claim_filter_reason",
                "claim_type_description",
            ],
            dropna=False,
        )
        .agg(
            claim_rows=("claim_number", "size"),
            unique_machines=("machine_key", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["warranty_claim_filter_keep", "claim_rows"],
            ascending=[True, False],
            kind="mergesort",
        )
    )
    filter_summary.to_csv(step_dir / "warranty_claim_filter_summary.csv", index=False)
    episodes.to_csv(step_dir / "claim_episodes.csv", index=False)

    summary = {
        "step": "01_build_claim_episodes",
        "output_dir": str(step_dir),
        "raw_claim_rows": int(len(filter_audit)),
        "claim_rows": int(len(warranty)),
        "filtered_out_claim_rows": int(
            filter_audit["warranty_claim_filter_keep"].eq(0).sum()
        ),
        "warranty_claim_filter_mode": str(
            getattr(config, "WARRANTY_CLAIM_FILTER_MODE", "none")
        ),
        "claim_episode_rows": int(len(episodes)),
        "unique_claim_machines": (
            int(episodes["machine_key"].nunique(dropna=True)) if len(episodes) else 0
        ),
        "claim_episode_gap_days": int(config.CLAIM_EPISODE_GAP_DAYS),
        "keep_only_valid_critical_part_claims": bool(
            config.KEEP_ONLY_VALID_CRITICAL_PART_CLAIMS
        ),
        "claim_date_min": episodes["claim_date"].min() if len(episodes) else None,
        "claim_date_max": episodes["claim_date"].max() if len(episodes) else None,
    }
    write_json(summary, step_dir / "run_summary.json")

    print(f"01_build_claim_episodes completed. Outputs: {step_dir}")
    print(f"  raw claim rows: {len(filter_audit):,}")
    print(f"  retained claim rows: {len(warranty):,}")
    print(
        f"  filtered-out claim rows: "
        f"{int(filter_audit['warranty_claim_filter_keep'].eq(0).sum()):,}"
    )
    print(f"  claim episodes: {len(episodes):,}")


if __name__ == "__main__":
    run()
