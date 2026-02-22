"""
mandeye_bag_common.py — Shared utilities for MandEye bag tools.

Provides ROS type-system setup (including Livox custom types), safe
deserialization helpers, IMU unit detection / conversion, bag-sequence
detection, topic-type classification, and ROS time utilities.

Used by:
  - mandeye_bag_audit.py
  - mandeye_bag_convert.py
  - mandeye_bag_extract.py
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# rosbags imports (fail-fast if missing)
# ---------------------------------------------------------------------------
try:
    from rosbags.rosbag1 import Reader as Reader1
    from rosbags.rosbag1.reader import ReaderError as Reader1Error
    from rosbags.rosbag2 import Reader as Reader2
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit("ERROR: 'rosbags' is required.  Install with:  pip install rosbags")

# Optional writers — only convert / extract need them
try:
    from rosbags.rosbag1 import Writer as Writer1
except ImportError:
    Writer1 = None  # type: ignore[assignment,misc]

try:
    from rosbags.rosbag2 import Writer as Writer2
except ImportError:
    Writer2 = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Type-system setup (shared singleton)
# ---------------------------------------------------------------------------
typestore = get_typestore(Stores.ROS2_HUMBLE)

# Bind (de)serialization functions
serialize_cdr = typestore.serialize_cdr
deserialize_cdr = typestore.deserialize_cdr
serialize_ros1 = typestore.serialize_ros1
deserialize_ros1 = typestore.deserialize_ros1
cdr_to_ros1 = typestore.cdr_to_ros1
ros1_to_cdr = typestore.ros1_to_cdr

# ---------------------------------------------------------------------------
# Register Livox custom message types
# ---------------------------------------------------------------------------
try:
    from rosbags.typesys.msg import get_types_from_msg

    _CUSTOM_POINT_DEF = (
        "uint32 offset_time\nfloat32 x\nfloat32 y\nfloat32 z\n"
        "uint8 reflectivity\nuint8 tag\nuint8 line"
    )
    _CUSTOM_MSG_DEF = (
        "std_msgs/Header header\nuint64 timebase\nuint32 point_num\n"
        "uint8 lidar_id\nuint8[3] rsvd\n{pkg}/CustomPoint[] points"
    )
    for _pkg in ("livox_ros_driver", "livox_ros_driver2"):
        _pt = get_types_from_msg(_CUSTOM_POINT_DEF, f"{_pkg}/msg/CustomPoint")
        typestore.register(_pt)
        _mt = get_types_from_msg(
            _CUSTOM_MSG_DEF.format(pkg=_pkg), f"{_pkg}/msg/CustomMsg"
        )
        typestore.register(_mt)
except Exception:
    pass  # non-critical: only needed when bags contain Livox CustomMsg

# ---------------------------------------------------------------------------
# Convenience type references
# ---------------------------------------------------------------------------
Header = typestore.types["std_msgs/msg/Header"]
Time = typestore.types["builtin_interfaces/msg/Time"]
Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
Imu = typestore.types["sensor_msgs/msg/Imu"]
PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
PointField = typestore.types["sensor_msgs/msg/PointField"]


# ============================================================================
# Safe deserialization
# ============================================================================

def deserialize_ros1_safe(rawdata: bytes, msgtype: str):
    """Deserialize ROS1 raw data with automatic fallback.

    Tries ``deserialize_ros1`` first; on failure converts via
    ``ros1_to_cdr`` and then ``deserialize_cdr``.
    """
    try:
        return deserialize_ros1(rawdata, msgtype)
    except (UnicodeDecodeError, Exception):
        cdr = ros1_to_cdr(rawdata, msgtype)
        return deserialize_cdr(cdr, msgtype)


def deserialize_ros1_tracked(raw: bytes, msgtype: str):
    """Deserialize ROS1 data with status tracking.

    Returns ``(msg | None, status)`` where *status* is one of
    ``"ok"``, ``"fallback"``, ``"fail"``.
    """
    try:
        msg = deserialize_ros1(raw, msgtype)
        return msg, "ok"
    except Exception:
        pass
    try:
        msg = deserialize_cdr(ros1_to_cdr(raw, msgtype), msgtype)
        return msg, "fallback"
    except Exception:
        return None, "fail"


# ============================================================================
# ROS time utilities
# ============================================================================

def rostime_to_sec(stamp) -> float:
    """Convert a ROS ``Time`` stamp to seconds (float)."""
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def rostime_to_nsec(stamp) -> int:
    """Convert a ROS ``Time`` stamp to nanoseconds (int)."""
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def sec_to_rostime(t: float):
    """Convert seconds (float) to a ROS ``Time`` message."""
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    return Time(sec=sec, nanosec=nsec)


def sec_to_nsec(t: float) -> int:
    """Convert seconds (float) to nanoseconds (int)."""
    return int(t * 1e9)


# ============================================================================
# Topic-type classifiers
# ============================================================================

def is_imu_type(msgtype: str) -> bool:
    """Return *True* if *msgtype* looks like an IMU message."""
    return "Imu" in msgtype


def is_pc_type(msgtype: str) -> bool:
    """Return *True* if *msgtype* is a point-cloud type (``PointCloud2`` or Livox ``CustomMsg``)."""
    return "PointCloud2" in msgtype or "CustomMsg" in msgtype


def is_custom_msg(msgtype: str) -> bool:
    """Return *True* if *msgtype* is a Livox ``CustomMsg``."""
    return "CustomMsg" in msgtype


def is_image_type(msgtype: str) -> bool:
    """Return *True* for ``sensor_msgs/Image``."""
    return "sensor_msgs" in msgtype and msgtype.endswith("/Image")


def is_compressed_image_type(msgtype: str) -> bool:
    """Return *True* for ``sensor_msgs/CompressedImage``."""
    return "CompressedImage" in msgtype


def is_navsatfix_type(msgtype: str) -> bool:
    """Return *True* for ``sensor_msgs/NavSatFix``."""
    return "NavSatFix" in msgtype


def is_nmea_type(msgtype: str) -> bool:
    """Return *True* for ``nmea_msgs/Sentence``."""
    return "nmea_msgs" in msgtype and "Sentence" in msgtype


def is_odometry_type(msgtype: str) -> bool:
    """Return *True* for ``nav_msgs/Odometry``."""
    return "Odometry" in msgtype


def is_tf_type(msgtype: str) -> bool:
    """Return *True* for ``tf2_msgs/TFMessage`` or ``tf/tfMessage``."""
    return "TFMessage" in msgtype or "tfMessage" in msgtype


def classify_topic(msgtype: str) -> str:
    """Return a human-readable category string for a ROS message type.

    Categories: ``imu``, ``pointcloud``, ``image``, ``compressed_image``,
    ``navsatfix``, ``nmea``, ``odometry``, ``tf``, ``other``.
    """
    if is_imu_type(msgtype):
        return "imu"
    if is_pc_type(msgtype):
        return "pointcloud"
    if is_compressed_image_type(msgtype):
        return "compressed_image"
    if is_image_type(msgtype):
        return "image"
    if is_navsatfix_type(msgtype):
        return "navsatfix"
    if is_nmea_type(msgtype):
        return "nmea"
    if is_odometry_type(msgtype):
        return "odometry"
    if is_tf_type(msgtype):
        return "tf"
    return "other"


# ============================================================================
# IMU unit detection & conversion
# ============================================================================

_G = 9.80665  # m/s² per g


def guess_acc_unit(magnitudes: np.ndarray) -> Tuple[str, float]:
    """Guess accelerometer unit from |acc| distribution.

    Returns ``(unit_label, scale_to_mps2)``.
    """
    if len(magnitudes) == 0:
        return "?", 1.0
    med = float(np.median(magnitudes))
    if 7.0 <= med <= 12.5:
        return "m/s²", 1.0
    if 0.7 <= med <= 1.4:
        return "g", _G
    if 800 <= med <= 1300:
        return "mg", _G / 1000.0
    if 7000 <= med <= 12500:
        return "mm/s²", 0.001
    return "?", 1.0


def guess_gyro_unit(magnitudes: np.ndarray) -> Tuple[str, float]:
    """Guess gyroscope unit from |gyro| distribution.

    Returns ``(unit_label, scale_to_radps)``.
    """
    if len(magnitudes) == 0:
        return "?", 1.0
    p99 = float(np.percentile(magnitudes, 99))
    if p99 < 20:
        return "rad/s", 1.0
    if 20 <= p99 <= 2000:
        return "deg/s", math.pi / 180.0
    return "?", 1.0


def compute_imu_factors(
    acc_unit: str,
    acc_scale_to_mps2: float,
    gyro_unit: str,
    gyro_scale_to_radps: float,
) -> Tuple[float, float]:
    """Compute multiplication factors so that:

    - ``raw_acc  * acc_factor  → g``
    - ``raw_gyro * gyro_factor → deg/s``
    """
    acc_factor = acc_scale_to_mps2 / _G
    gyro_factor = gyro_scale_to_radps * (180.0 / math.pi)
    return acc_factor, gyro_factor


# Known unit name → (canonical_name, scale_to_SI) lookup tables
_ACC_UNIT_TABLE: Dict[str, Tuple[str, float]] = {
    "m/s²": ("m/s²", 1.0),
    "m/s2": ("m/s²", 1.0),
    "g":    ("g", _G),
    "mg":   ("mg", _G / 1000.0),
    "mm/s²": ("mm/s²", 0.001),
    "mm/s2": ("mm/s²", 0.001),
}

_GYRO_UNIT_TABLE: Dict[str, Tuple[str, float]] = {
    "rad/s":  ("rad/s", 1.0),
    "deg/s":  ("deg/s", math.pi / 180.0),
    "mdeg/s": ("mdeg/s", math.pi / 180000.0),
}


def guess_acc_unit_by_name(name: str) -> Tuple[str, float]:
    """Look up accelerometer unit by name.

    Returns ``(canonical_name, scale_to_mps2)``.
    """
    key = name.strip().lower().replace("²", "2")
    for k, v in _ACC_UNIT_TABLE.items():
        if k.lower().replace("²", "2") == key:
            return v
    print(f"  WARNING: Unknown acc unit '{name}', assuming g (no conversion)")
    return ("g", _G)


def guess_gyro_unit_by_name(name: str) -> Tuple[str, float]:
    """Look up gyroscope unit by name.

    Returns ``(canonical_name, scale_to_radps)``.
    """
    key = name.strip().lower()
    for k, v in _GYRO_UNIT_TABLE.items():
        if k.lower() == key:
            return v
    print(f"  WARNING: Unknown gyro unit '{name}', assuming deg/s (no conversion)")
    return ("deg/s", math.pi / 180.0)


# ============================================================================
# Bag sequence detection (multi-volume / split recordings)
# ============================================================================

def extract_seq_prefix(stem: str) -> Tuple[str, Optional[int]]:
    """Extract ``(prefix, index)`` from a bag filename stem.

    Handles common rosbag-split naming conventions::

        recording           -> ("recording", None)
        recording_0         -> ("recording", 0)
        recording_003       -> ("recording", 3)
        rec_2022-01-01-12-00-00_0  -> ("rec", 0)
    """
    # prefix_DATETIME_INDEX
    m = re.match(
        r'^(.+?)_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(\d+)$', stem,
    )
    if m:
        return m.group(1), int(m.group(2))
    # prefix_INDEX
    m = re.match(r'^(.+?)_(\d+)$', stem)
    if m:
        return m.group(1), int(m.group(2))
    # no index
    return stem, None


def detect_bag_sequence(bag_path: Path) -> List[Path]:
    """Find all ``.bag`` siblings that share the same recording prefix.

    Returns a list sorted by sequence index.  Always contains at least
    *bag_path* itself.  For non-``.bag`` inputs returns ``[bag_path]``.
    """
    if bag_path.suffix != ".bag":
        return [bag_path]

    prefix, _ = extract_seq_prefix(bag_path.stem)
    parent = bag_path.parent

    candidates: List[Tuple[int, Path]] = []
    for f in sorted(parent.glob("*.bag")):
        p, idx = extract_seq_prefix(f.stem)
        if p == prefix:
            candidates.append((idx if idx is not None else -1, f))

    if len(candidates) <= 1:
        return [bag_path]

    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates]


@dataclass
class SequenceInfo:
    """Per-bag timing info within a multi-volume sequence."""
    path: Path
    start_ns: int
    end_ns: int
    duration_s: float
    gap_from_prev_s: Optional[float] = None


def validate_bag_sequence(bags: List[Path]) -> List[SequenceInfo]:
    """Open each bag to read start/end times and compute inter-bag gaps.

    Returns a list of :class:`SequenceInfo` sorted by start time.
    """
    infos: List[SequenceInfo] = []
    for bag_path in bags:
        try:
            with Reader1(bag_path) as reader:
                start = reader.start_time   # nanoseconds
                end = reader.end_time
                duration = (end - start) / 1e9
                infos.append(SequenceInfo(
                    path=bag_path,
                    start_ns=start,
                    end_ns=end,
                    duration_s=duration,
                ))
        except Exception as exc:
            print(f"  WARNING: Cannot read {bag_path.name}: {exc}",
                  file=sys.stderr)

    infos.sort(key=lambda x: x.start_ns)
    for i in range(1, len(infos)):
        infos[i].gap_from_prev_s = (
            infos[i].start_ns - infos[i - 1].end_ns
        ) / 1e9

    return infos


def print_sequence_summary(seq_infos: List[SequenceInfo]) -> None:
    """Print a human-readable sequence summary to stdout."""
    if not seq_infos:
        return
    total_dur = sum(si.duration_s for si in seq_infos)
    total_start = seq_infos[0].start_ns / 1e9

    print(f"\n{'='*72}")
    print(f"  SEQUENCE SUMMARY  ({len(seq_infos)} bags, "
          f"total duration: {total_dur:.1f}s)")
    print(f"{'='*72}")
    for i, si in enumerate(seq_infos):
        rel_start = (si.start_ns / 1e9) - total_start
        rel_end = (si.end_ns / 1e9) - total_start
        gap_str = ""
        if si.gap_from_prev_s is not None:
            g = si.gap_from_prev_s
            ok = "✓" if abs(g) < 1.0 else ("⚠ overlap" if g < 0 else "⚠ gap")
            gap_str = f"  gap: {g:.3f}s {ok}"
        print(f"  [{i}] {si.path.name:40s}  "
              f"[{rel_start:8.1f}s .. {rel_end:8.1f}s]  "
              f"dur: {si.duration_s:7.1f}s{gap_str}")
    print()


def print_sequence_info(bags: List[Path]) -> None:
    """Print a quick sequence summary using bag metadata (no SequenceInfo)."""
    total_dur = 0.0
    for i, bag_path in enumerate(bags):
        try:
            with Reader1(bag_path) as reader:
                start = reader.start_time
                end = reader.end_time
                dur = (end - start) / 1e9
                total_dur += dur
                gap_str = ""
                if i > 0:
                    try:
                        with Reader1(bags[i - 1]) as prev:
                            gap = (start - prev.end_time) / 1e9
                            ok = "✓" if abs(gap) < 1.0 else (
                                "⚠ overlap" if gap < 0 else "⚠ gap")
                            gap_str = f"  gap: {gap:.3f}s {ok}"
                    except Exception:
                        pass
                print(f"    [{i}] {bag_path.name:40s}  dur: {dur:7.1f}s{gap_str}")
        except Exception as exc:
            print(f"    [{i}] {bag_path.name:40s}  ERROR: {exc}")

    print(f"    Total duration: {total_dur:.1f}s")


# ============================================================================
# Output path helpers
# ============================================================================

def resolve_unique_output_path(path: str) -> str:
    """If *path* already exists, append ``_1``, ``_2``, … until unique."""
    if not Path(path).exists():
        return path
    base = path.rstrip("/\\")
    for i in range(1, 10000):
        candidate = f"{base}_{i}"
        if not Path(candidate).exists():
            return candidate
    return f"{base}_dup"


# ============================================================================
# Sequence CLI arg helpers
# ============================================================================

def add_sequence_args(parser) -> None:
    """Add ``--sequence`` / ``--no-sequence`` to an :class:`ArgumentParser`."""
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--sequence", action="store_true", default=None,
        help="Process all bags in the detected sequence (multi-volume split)",
    )
    grp.add_argument(
        "--no-sequence", action="store_true",
        help="Suppress sequence detection; process only the given file",
    )


def detect_dir_bags(input_path: Path) -> List[Path]:
    """If *input_path* is a directory of ``.bag`` files (not ROS2), return them sorted."""
    if input_path.is_dir() and not any(input_path.glob("metadata.yaml")):
        bags = sorted(input_path.glob("*.bag"))
        if bags:
            return bags
    return []
