"""Prepare compact operation caches for faster repeated model runs.

The uploaded partial operation CSV may contain a long blank tail. This step
streams the source once, writes only rows with a populated LOCAL_DATE, and
creates a tiny unique machine-roster file. All settings come from config.py.
"""
from __future__ import annotations

import csv
import os
import re
import time
from pathlib import Path

import config


def _machine_key(model: str, serial: str) -> str | None:
    model_text = str(model).strip().upper()
    match = re.search(r"(\d+)", str(serial))
    if not model_text or match is None:
        return None
    return f"{model_text}-{match.group(1)}"


def main() -> None:
    raw_path = Path(config.OPERATION_RAW_FILE)
    clean_path = Path(config.OPERATION_CLEAN_FILE)
    roster_path = Path(config.OPERATION_ROSTER_FILE)
    force = bool(getattr(config, "OPERATION_CACHE_FORCE_REBUILD", False))

    if not raw_path.exists():
        raise FileNotFoundError(f"Operation source not found: {raw_path}")
    if clean_path.exists() and roster_path.exists() and not force:
        print(
            "Operation caches already exist. Set "
            "OPERATION_CACHE_FORCE_REBUILD=True to recreate them."
        )
        return

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    clean_tmp = clean_path.with_suffix(clean_path.suffix + ".tmp")
    roster_tmp = roster_path.with_suffix(roster_path.suffix + ".tmp")

    started = time.time()
    rows_seen = 0
    rows_kept = 0
    machines: set[str] = set()
    csv.field_size_limit(min(2**31 - 1, 1_000_000_000))

    try:
        with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as source, clean_tmp.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source)
            writer = csv.writer(target, lineterminator="\n")
            header = next(reader)
            index = {name: position for position, name in enumerate(header)}
            required = ["LOCAL_DATE", "full_model", "SERIAL"]
            missing = [name for name in required if name not in index]
            if missing:
                raise ValueError(
                    "Operation source is missing required columns: " + ", ".join(missing)
                )
            writer.writerow(header)
            max_required_index = max(index[name] for name in required)

            for row in reader:
                rows_seen += 1
                if len(row) <= max_required_index:
                    continue
                if not row[index["LOCAL_DATE"]].strip():
                    continue
                writer.writerow(row)
                rows_kept += 1
                key = _machine_key(row[index["full_model"]], row[index["SERIAL"]])
                if key:
                    machines.add(key)
                if rows_seen % 250_000 == 0:
                    print(
                        f"Scanned {rows_seen:,} rows; kept {rows_kept:,}; "
                        f"elapsed {time.time() - started:.1f}s",
                        flush=True,
                    )

        with roster_tmp.open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(["machine_key"])
            writer.writerows([[key] for key in sorted(machines)])

        os.replace(clean_tmp, clean_path)
        os.replace(roster_tmp, roster_path)
    finally:
        for temporary in [clean_tmp, roster_tmp]:
            if temporary.exists():
                temporary.unlink()

    print("Operation cache build complete")
    print(f"  source rows scanned: {rows_seen:,}")
    print(f"  populated rows retained: {rows_kept:,}")
    print(f"  unique machines: {len(machines):,}")
    print(f"  cleaned file: {clean_path}")
    print(f"  roster file: {roster_path}")
    print(f"  elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
