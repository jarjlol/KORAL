#!/usr/bin/env python3
"""
convert_smart_to_koral.py — Convert NVMe SMART poller output to KORAL Stage II input format.

Your smart_log.csv has flat time-series rows (one row per poll):
  timestamp, composite_temp_c, media_errors, ...

KORAL expects a CSV where:
  - Each ROW is one "analysis window" (e.g., 30 data points)
  - Each COLUMN is named r_<SMART_ID> (e.g., r_194 for temperature)
  - Each CELL contains a JSON list of values for that window

This script:
  1. Reads smart_log.csv
  2. Maps NVMe attributes to SMART attribute IDs
  3. Slices the time-series into windows (default: 30 points per window)
  4. Outputs a KORAL-compatible CSV

Usage:
    python convert_smart_to_koral.py \
        --input smart_log.csv \
        --output koral_kv_dataset.csv \
        --window_size 30
"""

import argparse
import json
import csv
import sys
from pathlib import Path


# ── NVMe → SMART ID mapping ──────────────────────────────────────────
# NVMe attributes don't have traditional SMART IDs (those are for SATA).
# KORAL's infer_smart_columns() looks for r_<number> columns.
# We map NVMe attributes to commonly-used SMART-like IDs:

NVME_TO_SMART_ID = {
    "composite_temp_c":               194,  # Temperature
    "available_spare_pct":            231,  # SSD Life Left / Available Spare
    "available_spare_threshold_pct":  232,  # Available Spare Threshold
    "percentage_used":                233,  # Media Wearout / Percentage Used
    "data_units_read":                241,  # Total LBAs Read
    "data_units_written":             242,  # Total LBAs Written
    "host_read_commands":             246,  # Host Read Commands
    "host_write_commands":            247,  # Host Write Commands
    "media_errors":                   199,  # Uncorrectable Errors (closest match)
    "error_log_entries":              175,  # Error Log Entries (SSD-like)
    "power_on_hours":                   9,  # Power-On Hours
    "power_cycles":                    12,  # Power Cycle Count
    "unsafe_shutdowns":               192,  # Unsafe Shutdowns / Power-Off Retract
    "critical_warning":                 1,  # Critical Warning (mapped to raw read error)
    "controller_busy_time_min":       190,  # Controller Busy Time
    "warn_temp_time_min":             195,  # Warning Temp Time
    "crit_temp_time_min":             196,  # Critical Temp Time
    "temp_sensor_1":                  194,  # (duplicate → skip, handled below)
    "temp_sensor_2":                  190,  # (reuse 190 slot for sensor 2)
}

# Some NVMe fields map to the same ID — handle duplicates
# We'll skip temp_sensor_1 (same as composite_temp_c → r_194)
SKIP_COLUMNS = {"timestamp", "temp_sensor_1"}


def convert(input_path: str, output_path: str, window_size: int = 30):
    """Convert flat NVMe SMART log to KORAL windowed format."""

    # Read input
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("ERROR: No data rows in input", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(rows)} data points from {input_path}")

    # Get column names (excluding timestamp and skipped)
    all_cols = [c for c in rows[0].keys() if c not in SKIP_COLUMNS]

    # Build column → SMART ID mapping
    col_to_id = {}
    used_ids = set()
    for col in all_cols:
        if col in NVME_TO_SMART_ID:
            sid = NVME_TO_SMART_ID[col]
            if sid not in used_ids:
                col_to_id[col] = sid
                used_ids.add(sid)
            else:
                print(f"  Skipping '{col}' (SMART ID {sid} already used)")
        else:
            print(f"  Warning: no SMART mapping for '{col}', skipping")

    print(f"Mapped {len(col_to_id)} NVMe attributes to SMART IDs")

    # Create windows
    num_windows = len(rows) // window_size
    if num_windows == 0:
        print(f"ERROR: Not enough data ({len(rows)} rows) for window_size={window_size}")
        print(f"  Try a smaller window_size (e.g., --window_size {max(1, len(rows) // 3)})")
        sys.exit(1)

    print(f"Creating {num_windows} windows of {window_size} points each")
    leftover = len(rows) - (num_windows * window_size)
    if leftover:
        print(f"  ({leftover} leftover points will be discarded)")

    # Build output rows
    output_rows = []
    for w in range(num_windows):
        start = w * window_size
        end = start + window_size
        window_rows = rows[start:end]

        out_row = {
            "sample_id": f"kv_window_{w}",
            "dataset_type": "KV_TWITTER",
            "disk_id": "nvme0n1",
            "failure": 0,  # no failure label (healthy drive)
        }

        for col, sid in col_to_id.items():
            # Extract series of values for this window
            series = []
            for r in window_rows:
                try:
                    val = float(r[col])
                    series.append(val)
                except (ValueError, KeyError):
                    pass

            out_row[f"r_{sid}"] = json.dumps(series)

        output_rows.append(out_row)

    # Write output CSV
    if not output_rows:
        print("ERROR: No windows generated", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(output_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nWrote {len(output_rows)} windows to {output_path}")
    print(f"Columns: {', '.join(f'r_{sid}' for sid in sorted(col_to_id.values()))}")
    print(f"\nReady for KORAL! Run:")
    print(f"  python -m stage_II.cli --dataset_type KV_TWITTER \\")
    print(f"    --input_csv {output_path} \\")
    print(f"    --tasks descriptive,prescriptive,whatif \\")
    print(f"    --out_name kv_baseline")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert NVMe SMART log to KORAL format")
    p.add_argument("--input", required=True, help="Path to smart_log.csv")
    p.add_argument("--output", default="koral_kv_dataset.csv", help="Output path")
    p.add_argument("--window_size", type=int, default=30,
                   help="Points per window (default: 30)")
    args = p.parse_args()

    convert(args.input, args.output, args.window_size)
