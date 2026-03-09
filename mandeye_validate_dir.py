#!/usr/bin/env python3
"""
mandeye_validate_dir.py — Validate a MandEye / HDMapping data folder.

Checks that the folder is ready for conversion to a ROS bag:
  - Matching .csv / .laz pairs (same indices)
  - CSV internal consistency (column count, timestamp monotonicity)
  - LAZ files readable and point counts > 0
  - Timestamp overlap between CSV and LAZ
  - .sn serial files presence

Usage:
  python mandeye_validate_dir.py <folder>
  python mandeye_validate_dir.py <folder> --verbose
  python mandeye_validate_dir.py <folder> --json report.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colours (Windows-safe)
# ---------------------------------------------------------------------------
def _enable_ansi() -> None:
    if sys.platform != "win32":
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong(0)
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass
    # Reconfigure stdout/stderr to UTF-8 so Unicode symbols (✓ ⚠ ✗) work
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


_enable_ansi()

_OK    = "\033[32m✓\033[0m"
_WARN  = "\033[93m⚠\033[0m"
_FAIL  = "\033[91m✗\033[0m"
_COL_OK   = "\033[32m"
_COL_WARN = "\033[93m"
_COL_FAIL = "\033[91m"
_RESET    = "\033[0m"


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return _OK
    return _WARN if warn else _FAIL


# ---------------------------------------------------------------------------
# LAZ reading (optional — degrades gracefully)
# ---------------------------------------------------------------------------
@dataclass
class LazInfo:
    path: Path
    point_count: int = 0
    ts_min: Optional[float] = None
    ts_max: Optional[float] = None
    file_source_id: Optional[int] = None
    global_encoding: Optional[int] = None
    scales: Optional[List[float]] = None
    # user_data distribution: {sensor_id: point_count}
    sensor_counts: Dict[int, int] = field(default_factory=dict)
    # classification distribution: {tag_value: point_count} (top entries)
    class_counts:  Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    skipped: bool = False  # True when laspy not installed

    @property
    def ts_span_s(self) -> Optional[float]:
        if self.ts_min is not None and self.ts_max is not None:
            return self.ts_max - self.ts_min
        return None

    @property
    def sensor_count(self) -> int:
        return len(self.sensor_counts)


def _validate_laz(path: Path) -> LazInfo:
    """Read LAZ/LAS file and return full LazInfo."""
    info = LazInfo(path=path)
    try:
        import laspy  # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]
    except ImportError:
        info.skipped = True
        info.warnings.append("laspy not installed — LAZ content check skipped")
        return info
    try:
        with laspy.open(str(path)) as lf:
            info.point_count = int(lf.header.point_count)
            las = lf.read()

        # Header metadata
        info.file_source_id = int(las.header.file_source_id)
        info.global_encoding = int(las.header.global_encoding.value)
        info.scales = [float(s) for s in las.header.scales]

        if info.file_source_id != 4711:
            info.warnings.append(
                f"file_source_id={info.file_source_id} (expected 4711)"
            )
        if (info.global_encoding & 1) == 0:
            info.warnings.append(
                f"global_encoding bit0=0 (GPS time not in Adjusted Standard GPS Time)"
            )
        expected_scale = 0.0001
        if info.scales and any(abs(s - expected_scale) > 1e-6 for s in info.scales):
            info.warnings.append(
                f"Unexpected scale factors: {info.scales} (expected {expected_scale})"
            )

        # GPS time
        if hasattr(las, "gps_time") and len(las.gps_time) > 0:
            ts = las.gps_time
            info.ts_min = float(ts.min())
            info.ts_max = float(ts.max())

        # user_data → sensor ID distribution
        if hasattr(las, "user_data"):
            ud = np.array(las.user_data)
            vals, counts = np.unique(ud, return_counts=True)
            info.sensor_counts = {int(v): int(c) for v, c in zip(vals, counts)}

        # classification → Livox tag distribution (top 6 values)
        if hasattr(las, "classification"):
            cl = np.array(las.classification)
            vals, counts = np.unique(cl, return_counts=True)
            top = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: -x[1])[:6]
            info.class_counts = {int(v): int(c) for v, c in top}

        if info.point_count == 0:
            info.errors.append("LAZ has 0 points")

    except Exception as exc:
        info.errors.append(f"LAZ read error: {exc}")
    return info


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------
_CSV_MIN_COLS = 8   # timestamp, gx, gy, gz, ax, ay, az, imuId

@dataclass
class CsvInfo:
    path: Path
    rows: int = 0
    cols_ok: bool = True
    monotone: bool = True
    bad_cols_rows: List[int] = field(default_factory=list)
    first_ts_ns: Optional[int] = None
    last_ts_ns: Optional[int] = None
    errors: List[str] = field(default_factory=list)

    @property
    def ts_span_s(self) -> Optional[float]:
        if self.first_ts_ns is not None and self.last_ts_ns is not None:
            return (self.last_ts_ns - self.first_ts_ns) / 1e9
        return None


def _validate_csv(path: Path, delim: str = ",") -> CsvInfo:
    info = CsvInfo(path=path)
    prev_ts: Optional[int] = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            # Auto-detect delimiter: peek first non-empty line
            first_line = ""
            for raw in f:
                first_line = raw.strip()
                if first_line:
                    break
            if first_line:
                # If user-specified delim doesn't yield >= 8 cols, try space/comma
                if len(first_line.split(delim)) < _CSV_MIN_COLS:
                    for candidate in (" ", ",", "\t"):
                        if len(first_line.split(candidate)) >= _CSV_MIN_COLS:
                            delim = candidate
                            break
            f.seek(0)
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(delim)
                if len(parts) < _CSV_MIN_COLS:
                    info.cols_ok = False
                    info.bad_cols_rows.append(lineno)
                    if len(info.bad_cols_rows) >= 5:
                        info.errors.append(
                            f"Too few columns on rows (showing first 5): "
                            f"{info.bad_cols_rows}"
                        )
                        break
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    info.errors.append(f"Row {lineno}: non-integer timestamp '{parts[0][:20]}'")
                    continue
                if prev_ts is not None and ts < prev_ts:
                    info.monotone = False
                    if not info.errors:
                        info.errors.append(
                            f"Non-monotone timestamp at row {lineno}: "
                            f"{ts} < {prev_ts}"
                        )
                if info.first_ts_ns is None:
                    info.first_ts_ns = ts
                info.last_ts_ns = ts
                prev_ts = ts
                info.rows += 1
    except OSError as exc:
        info.errors.append(f"Cannot read file: {exc}")
    return info


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------
@dataclass
class FileCheck:
    index: int
    csv_path: Optional[Path] = None
    laz_path: Optional[Path] = None
    sn_path:  Optional[Path] = None
    csv_info: Optional[CsvInfo] = None
    laz_info: Optional[LazInfo] = None
    issues: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.issues) == 0


def _parse_index(name: str) -> Optional[int]:
    """Extract leading 4-digit index from filename stem."""
    m = re.search(r"(\d{4})", name)
    return int(m.group(1)) if m else None


def validate_dir(
    folder: Path,
    csv_delim: str = ",",
    verbose: bool = False,
) -> Dict[str, Any]:
    folder = Path(folder)
    if not folder.is_dir():
        print(f"{_FAIL}  Not a directory: {folder}", flush=True)
        sys.exit(1)

    csv_files = sorted(folder.glob("imu_*.csv"))
    laz_files = sorted(folder.glob("pointcloud_*.laz"))
    sn_files  = sorted(folder.glob("*.sn"))

    print(f"\n{'='*72}")
    print(f"  MANDEYE FOLDER VALIDATION")
    print(f"  Folder: {folder}")
    print(f"{'='*72}")
    print(f"  Found: {len(csv_files)} CSV  |  {len(laz_files)} LAZ  |  {len(sn_files)} SN")
    print()

    # --- index maps ---
    csv_map: Dict[int, Path] = {i: f for f in csv_files if (i := _parse_index(f.stem)) is not None}  # type: ignore[misc]
    laz_map: Dict[int, Path] = {i: f for f in laz_files if (i := _parse_index(f.stem)) is not None}  # type: ignore[misc]
    sn_map:  Dict[int, Path] = {i: f for f in sn_files  if (i := _parse_index(f.stem)) is not None}  # type: ignore[misc]

    all_indices: List[int] = sorted(set(csv_map.keys()) | set(laz_map.keys()))

    checks: List[FileCheck] = []
    global_issues: List[str] = []

    if not all_indices:
        global_issues.append("No imu_NNNN.csv or pointcloud_NNNN.laz files found.")

    for idx in all_indices:
        fc = FileCheck(index=idx)
        fc.csv_path = csv_map.get(idx)
        fc.laz_path = laz_map.get(idx)
        fc.sn_path  = sn_map.get(idx)

        if fc.csv_path is None:
            fc.issues.append(f"Missing CSV for index {idx:04d}")
        if fc.laz_path is None:
            fc.issues.append(f"Missing LAZ for index {idx:04d}")

        if fc.csv_path is not None:
            fc.csv_info = _validate_csv(fc.csv_path, csv_delim)
            if not fc.csv_info.cols_ok:
                fc.issues.append(f"CSV has rows with < {_CSV_MIN_COLS} columns")
            if not fc.csv_info.monotone:
                fc.issues.append("CSV timestamps not monotone")
            if fc.csv_info.rows == 0:
                fc.issues.append("CSV is empty")
            fc.issues.extend(fc.csv_info.errors[:3])  # first 3 errors only

        if fc.laz_path is not None:
            fc.laz_info = _validate_laz(fc.laz_path)
            li = fc.laz_info
            fc.issues.extend(li.errors)
            # warnings are non-fatal — shown in verbose but don't fail

        # Timestamp alignment check: CSV ns (header stamp) vs LAZ GPS s (header stamp)
        # Both are on the same clock domain (Livox sensor clock), so absolute
        # alignment should be within a few seconds.
        li = fc.laz_info
        ci = fc.csv_info
        if (ci is not None and li is not None
                and li.ts_min is not None and li.ts_max is not None
                and ci.first_ts_ns is not None and ci.last_ts_ns is not None):
            csv_s = ci.first_ts_ns / 1e9
            csv_e = ci.last_ts_ns  / 1e9
            laz_s = li.ts_min
            laz_e = li.ts_max
            csv_span = csv_e - csv_s
            laz_span = laz_e - laz_s
            # Span ratio check
            if csv_span > 0 and laz_span > 0:
                ratio = csv_span / laz_span
                if ratio < 0.5 or ratio > 2.0:
                    fc.issues.append(
                        f"Time span mismatch: CSV {csv_span:.1f}s vs "
                        f"LAZ {laz_span:.1f}s  (ratio {ratio:.2f})"
                    )
            # Absolute alignment check: start times should be close (same chunk)
            abs_diff = abs(csv_s - laz_s)
            if abs_diff > 5.0:  # more than 5s apart → different clocks or mismatched files
                fc.issues.append(
                    f"Timestamp misalignment: CSV start {csv_s:.3f}s vs "
                    f"LAZ GPS start {laz_s:.3f}s  (diff {abs_diff:.1f}s)"
                )

        checks.append(fc)

    # --- Index continuity ---
    if len(all_indices) >= 2:
        for a, b in zip(all_indices, all_indices[1:]):
            if b - a != 1:
                global_issues.append(f"Gap in index sequence: {a:04d} → {b:04d}")

    # --- SN files ---
    sn_ok = len(sn_files) >= 1
    if not sn_ok:
        global_issues.append("No .sn serial-number files found (may cause issues with some tools)")

    # --- Print per-file results ---
    fail_count = 0
    warn_count = 0
    for fc in checks:
        stat = _status(fc.ok)
        if not fc.ok:
            fail_count += 1

        li = fc.laz_info
        laz_pts_str = (
            f"{li.point_count:,}" if li is not None and not li.errors
            else ("N/A" if li is not None and li.skipped else "?")
        )

        if verbose or not fc.ok:
            csv_rows = fc.csv_info.rows if fc.csv_info else "?"
            csv_name = fc.csv_path.name if fc.csv_path else f"MISSING imu_{fc.index:04d}.csv"
            laz_name = fc.laz_path.name if fc.laz_path else f"MISSING pointcloud_{fc.index:04d}.laz"
            print(f"  {stat}  [{fc.index:04d}]  {csv_name}  ({csv_rows} rows)  |  {laz_name}  ({laz_pts_str} pts)")
            for issue in fc.issues:
                print(f"         {_FAIL}  {issue}")
            if li is not None:
                for w in li.warnings:
                    print(f"         {_WARN}  LAZ: {w}")
                if verbose and li.sensor_counts:
                    if li.sensor_count > 1:
                        counts_str = "  ".join(
                            f"sensor{s}:{c:,}" for s, c in sorted(li.sensor_counts.items())
                        )
                        print(f"              sensors (user_data): {counts_str}")
                    if li.class_counts:
                        top = sorted(li.class_counts.items(), key=lambda x: -x[1])[:4]
                        tag_str = "  ".join(f"tag{v}:{c:,}" for v, c in top)
                        print(f"              classification:       {tag_str}")

    if not verbose and fail_count == 0:
        print(f"  {_OK}  All {len(checks)} file pairs OK")
        if checks:
            total_rows = sum(fc.csv_info.rows for fc in checks if fc.csv_info)
            total_pts  = sum(
                fc.laz_info.point_count for fc in checks
                if fc.laz_info is not None and not fc.laz_info.errors
            )
            # Report multi-sensor setup if detected
            all_sensor_ids: set[int] = set()
            for fc in checks:
                if fc.laz_info:
                    all_sensor_ids |= set(fc.laz_info.sensor_counts.keys())
            sensor_note = (
                f"  |  {len(all_sensor_ids)} sensors (lidar_id {min(all_sensor_ids)}-{max(all_sensor_ids)})"
                if len(all_sensor_ids) > 1 else ""
            )
            print(f"     Total: {total_rows:,} IMU rows  |  {total_pts:,} LiDAR points{sensor_note}")

    # --- SN summary ---
    if sn_files:
        print(f"\n  {_OK}  Serial number files: {', '.join(f.name for f in sn_files)}")
    else:
        warn_count += 1
        print(f"\n  {_WARN}  No .sn files found")

    # --- Global issues ---
    if global_issues:
        print(f"\n  Global issues:")
        for gi in global_issues:
            print(f"    {_FAIL}  {gi}")
        fail_count += len([g for g in global_issues if "Gap" in g or "No imu" in g])

    # --- Final verdict ---
    print(f"\n{'─'*72}")
    total_issues = fail_count + len([g for g in global_issues
                                     if "Gap" in g or "Missing" in g])
    if total_issues == 0:
        col = _COL_OK
        verdict = "VALID — folder is ready for conversion"
    elif fail_count == 0:
        col = _COL_WARN
        verdict = f"WARNINGS ({warn_count}) — conversion may still work"
    else:
        col = _COL_FAIL
        verdict = f"INVALID — {fail_count} critical issue(s) found"

    print(f"  {col}{verdict}{_RESET}")
    print()

    result: Dict[str, Any] = {
        "folder": str(folder),
        "csv_count": len(csv_files),
        "laz_count": len(laz_files),
        "sn_count":  len(sn_files),
        "pair_count": len(checks),
        "fail_count": fail_count,
        "warn_count": warn_count,
        "verdict": verdict,
        "global_issues": global_issues,
        "files": [
            {
                "index": fc.index,
                "csv": str(fc.csv_path) if fc.csv_path else None,
                "laz": str(fc.laz_path) if fc.laz_path else None,
                "csv_rows": fc.csv_info.rows if fc.csv_info else None,
                "csv_monotone": fc.csv_info.monotone if fc.csv_info else None,
                "laz_points": fc.laz_info.point_count if fc.laz_info else None,
                "laz_ts_min": fc.laz_info.ts_min if fc.laz_info else None,
                "laz_ts_max": fc.laz_info.ts_max if fc.laz_info else None,
                "laz_file_source_id": fc.laz_info.file_source_id if fc.laz_info else None,
                "laz_global_encoding": fc.laz_info.global_encoding if fc.laz_info else None,
                "laz_sensors": fc.laz_info.sensor_counts if fc.laz_info else None,
                "laz_classification": fc.laz_info.class_counts if fc.laz_info else None,
                "laz_warnings": fc.laz_info.warnings if fc.laz_info else [],
                "issues": fc.issues,
            }
            for fc in checks
        ],
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a MandEye / HDMapping data folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("folder", help="Path to the MandEye data folder")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-file details even when OK",
    )
    parser.add_argument(
        "--json", metavar="FILE",
        help="Write validation report as JSON to FILE",
    )
    parser.add_argument(
        "--delim", default=",", metavar="CHAR",
        help="CSV delimiter (default: ',')",
    )
    args = parser.parse_args()

    result = validate_dir(
        Path(args.folder),
        csv_delim=args.delim,
        verbose=args.verbose,
    )

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON report written to: {out}")

    sys.exit(0 if result["fail_count"] == 0 else 1)


if __name__ == "__main__":
    main()
