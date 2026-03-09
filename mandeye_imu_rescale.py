#!/usr/bin/env python3
"""
mandeye_imu_rescale.py — Post-process MandEye IMU CSV files to rescale units.

Useful when IMU data was already extracted from a bag to MandEye format but
the unit conversion was wrong or skipped.  Operates directly on the CSV files
without needing to re-run the full bag extraction.

Usage:
  python mandeye_imu_rescale.py --dir <directory> [options]

Options:
  --dir <path>          Directory containing IMU CSV files (required)
  --acc-conv <conv>     Accelerometer conversion:
                          g2ms   : g       → m/s²   (× 9.80665)
                          ms2g   : m/s²    → g       (÷ 9.80665)
  --gyro-conv <conv>    Gyroscope conversion:
                          rad2deg : rad/s  → deg/s   (× 180/π)
                          deg2rad : deg/s  → rad/s   (× π/180)
  --time-conv <conv>    Timestamp conversion preset:
                          ns2sec  : ns    → s     (×  1e-9)
                          sec2ns  : s     → ns    (×  1e9)
                          ms2sec  : ms    → s     (×  1e-3)
                          sec2ms  : s     → ms    (×  1e3)
  --time-factor <X>     Custom timestamp multiplication factor (any float)
  --pattern <glob>      File glob to match (default: imu_*.csv)
  --backup              Save original files as <file>.bak before modifying
  --dry-run             Print what would change without writing anything

CSV format (both formats are supported):
  Legacy:  timestamp gyroX gyroY gyroZ accX accY accZ [imuId]
  Header:  timestamp,gyroX,gyroY,gyroZ,accX,accY,accZ,imuId  (with header row)

Examples:
  # Preview what would happen (no files written):
  python mandeye_imu_rescale.py --dir ./extracted --acc-conv ms2g --gyro-conv rad2deg --dry-run

  # Convert in-place with backup:
  python mandeye_imu_rescale.py --dir ./extracted --acc-conv ms2g --gyro-conv rad2deg --backup

  # Only rescale gyro, leave acc untouched:
  python mandeye_imu_rescale.py --dir ./extracted --gyro-conv rad2deg --backup
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

_G = 9.80665  # m/s² per g

# (from_label, to_label, factor)
_ACC_CONV: dict[str, tuple[str, str, float]] = {
    "g2ms":  ("g",    "m/s²",  _G),
    "ms2g":  ("m/s²", "g",     1.0 / _G),
}
_GYRO_CONV: dict[str, tuple[str, str, float]] = {
    "rad2deg": ("rad/s", "deg/s", 180.0 / math.pi),
    "deg2rad": ("deg/s", "rad/s", math.pi / 180.0),
}
_TIME_CONV: dict[str, tuple[str, str, float]] = {
    "ns2sec": ("ns",  "s",   1e-9),
    "sec2ns": ("s",   "ns",  1e9),
    "ms2sec": ("ms",  "s",   1e-3),
    "sec2ms": ("s",   "ms",  1e3),
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _detect_delimiter(line: str) -> str:
    """Pick the most likely delimiter from the first non-empty line."""
    for d in (",", "\t", " "):
        if d in line:
            return d
    return " "


def _has_header(first_line: str) -> bool:
    lower = first_line.lower()
    return "timestamp" in lower or "gyrox" in lower


def _parse_lines(text: str) -> tuple[Optional[str], str, list[list[str]]]:
    """Return (header_line_or_None, delimiter, list_of_value_rows)."""
    lines = text.splitlines()
    if not lines:
        return None, " ", []

    # strip empty lines at the top
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None, " ", []

    delim = _detect_delimiter(lines[0])
    header = None
    data_lines = lines

    if _has_header(lines[0]):
        header = lines[0]
        data_lines = lines[1:]

    rows: list[list[str]] = []
    for raw in data_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        rows.append(stripped.split(delim) if delim != " " else stripped.split())

    return header, delim, rows


def _apply_factors(
    rows: list[list[str]],
    acc_factor: Optional[float],
    gyro_factor: Optional[float],
    header: Optional[str],
    time_factor: Optional[float] = None,
) -> list[list[str]]:
    """Apply conversion factors to timestamp (col 0), gyro (cols 1-3), and/or acc (cols 4-6).

    Column layout (0-based):
      0         : timestamp (rescaled when time_factor is given)
      1, 2, 3   : gyroX, gyroY, gyroZ
      4, 5, 6   : accX,  accY,  accZ
      7         : imuId  (optional)
    """
    # Resolve column indices from header if present
    gyro_cols = [1, 2, 3]
    acc_cols  = [4, 5, 6]

    if header:
        delim = _detect_delimiter(header)
        cols = [c.strip().lower() for c in header.split(delim)]
        _GYRO_NAMES = ("gyrox", "gyroy", "gyroz")
        _ACC_NAMES  = ("accx",  "accy",  "accz")
        gyro_cols = [cols.index(n) for n in _GYRO_NAMES if n in cols] or gyro_cols
        acc_cols  = [cols.index(n) for n in _ACC_NAMES  if n in cols] or acc_cols

    result: list[list[str]] = []
    for row in rows:
        new_row = list(row)
        if time_factor is not None and len(new_row) > 0:
            new_row[0] = repr(float(new_row[0]) * time_factor)
        if gyro_factor is not None:
            for i in gyro_cols:
                if i < len(new_row):
                    new_row[i] = repr(float(new_row[i]) * gyro_factor)
        if acc_factor is not None:
            for i in acc_cols:
                if i < len(new_row):
                    new_row[i] = repr(float(new_row[i]) * acc_factor)
        result.append(new_row)
    return result


def _rows_to_text(header: Optional[str], delim: str, rows: list[list[str]]) -> str:
    lines = []
    if header:
        lines.append(header)
    for row in rows:
        lines.append(delim.join(row))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    acc_factor: Optional[float],
    gyro_factor: Optional[float],
    backup: bool,
    dry_run: bool,
    time_factor: Optional[float] = None,
) -> dict:
    """Process a single CSV file.  Returns a small stats dict."""
    text = path.read_text(encoding="utf-8")
    header, delim, rows = _parse_lines(text)

    if not rows:
        print(f"  {path.name}: empty / no data rows — skipped")
        return {"file": path.name, "rows": 0, "skipped": True}

    new_rows = _apply_factors(rows, acc_factor, gyro_factor, header, time_factor)

    # Build a short preview (first data row before → after)
    def _fmt(row: list[str]) -> str:
        ts = f"ts={float(row[0]):.6g}"
        vals = ", ".join(f"{float(v):.6g}" for v in row[1:7])
        return f"  [{ts} | {vals}]"

    print(f"  {path.name}  ({len(rows)} rows)")
    print(f"    before: {_fmt(rows[0])}")
    print(f"    after:  {_fmt(new_rows[0])}")

    if dry_run:
        print(f"    (dry-run — not written)")
        return {"file": path.name, "rows": len(rows), "written": False}

    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"    backup: {bak.name}")

    new_text = _rows_to_text(header, delim, new_rows)
    path.write_text(new_text, encoding="utf-8")
    print(f"    written.")
    return {"file": path.name, "rows": len(rows), "written": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescale IMU values in MandEye CSV files (post-extraction unit fix).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conversion keys:
  --acc-conv   g2ms   : g     → m/s²   (multiply by 9.80665)
               ms2g   : m/s²  → g      (divide  by 9.80665)
  --gyro-conv  rad2deg: rad/s → deg/s  (multiply by 180/π ≈ 57.296)
               deg2rad: deg/s → rad/s  (multiply by π/180 ≈ 0.01745)
  --time-conv  ns2sec : ns  → s     (×  1e-9)
               sec2ns : s   → ns    (×  1e9)
               ms2sec : ms  → s     (×  1e-3)
               sec2ms : s   → ms    (×  1e3)
  --time-factor <X> : custom multiplier applied to timestamp column

Examples:
  # Dry-run preview — nothing is written:
  python mandeye_imu_rescale.py --dir ./extracted --acc-conv ms2g --gyro-conv rad2deg --dry-run

  # Apply conversions in-place with backup:
  python mandeye_imu_rescale.py --dir ./extracted --acc-conv ms2g --gyro-conv rad2deg --backup

  # Only fix gyro, leave acc untouched:
  python mandeye_imu_rescale.py --dir ./extracted --gyro-conv rad2deg --backup

  # Convert timestamps from nanoseconds to seconds:
  python mandeye_imu_rescale.py --dir ./extracted --time-conv ns2sec --backup

  # Custom timestamp factor (e.g. ms → s):
  python mandeye_imu_rescale.py --dir ./extracted --time-factor 0.001 --backup""",
""",
    )
    parser.add_argument("--dir", required=True, metavar="PATH",
                        help="Directory containing IMU CSV files")
    parser.add_argument("--acc-conv", choices=list(_ACC_CONV), default=None,
                        help="Accelerometer conversion (omit = no change)")
    parser.add_argument("--gyro-conv", choices=list(_GYRO_CONV), default=None,
                        help="Gyroscope conversion (omit = no change)")
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--time-conv", choices=list(_TIME_CONV), default=None,
        help="Timestamp conversion preset: ns2sec, sec2ns, ms2sec, sec2ms",
    )
    time_group.add_argument(
        "--time-factor", type=float, default=None, metavar="X",
        help="Custom timestamp multiplication factor (mutually exclusive with --time-conv)",
    )
    parser.add_argument("--pattern", default="imu_*.csv", metavar="GLOB",
                        help="File glob pattern (default: imu_*.csv)")
    parser.add_argument("--backup", action="store_true",
                        help="Keep originals as <file>.bak")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show changes without writing files")
    args = parser.parse_args()

    if args.acc_conv is None and args.gyro_conv is None and args.time_conv is None and args.time_factor is None:
        parser.error("Nothing to do: specify at least one of --acc-conv, --gyro-conv, --time-conv, --time-factor.")

    work_dir = Path(args.dir)
    if not work_dir.is_dir():
        parser.error(f"Not a directory: {args.dir}")

    csv_files = sorted(work_dir.glob(args.pattern))
    if not csv_files:
        print(f"No files matching '{args.pattern}' found in {work_dir}")
        return

    acc_factor:  Optional[float] = None
    gyro_factor: Optional[float] = None

    if args.acc_conv:
        src, dst, acc_factor = _ACC_CONV[args.acc_conv]
        print(f"Accel:  {src} → {dst}  (× {acc_factor:.6g})")
    else:
        print("Accel:  no conversion")

    if args.gyro_conv:
        src, dst, gyro_factor = _GYRO_CONV[args.gyro_conv]
        print(f"Gyro:   {src} → {dst}  (× {gyro_factor:.6g})")
    else:
        print("Gyro:   no conversion")

    time_factor: Optional[float] = None
    if args.time_conv:
        src, dst, time_factor = _TIME_CONV[args.time_conv]
        print(f"Time:   {src} → {dst}  (× {time_factor:.6g})")
    elif args.time_factor is not None:
        time_factor = args.time_factor
        print(f"Time:   custom factor × {time_factor:.6g}")
    else:
        print("Time:   no conversion")

    if args.dry_run:
        print("Mode:   DRY-RUN (files will NOT be modified)\n")
    elif args.backup:
        print("Mode:   in-place with .bak backup\n")
    else:
        print("Mode:   in-place (no backup)\n")

    total_rows = 0
    total_files = 0
    for f in csv_files:
        stats = process_file(f, acc_factor, gyro_factor,
                             backup=args.backup, dry_run=args.dry_run,
                             time_factor=time_factor)
        total_rows  += stats.get("rows", 0)
        total_files += 1

    print(f"\nDone: {total_files} file(s), {total_rows} data row(s) processed.")


if __name__ == "__main__":
    main()
