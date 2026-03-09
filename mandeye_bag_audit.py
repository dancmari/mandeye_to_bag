#!/usr/bin/env python3
"""
mandeye_bag_audit.py  v0.8 — Enhanced bag auditor for HDMapping / MandEye.

Reads a ROS1 (.bag) or ROS2 bag, samples messages, and scores every possible
(pointcloud_topic, imu_topic) pair against 7 weighted criteria.  Outputs a
human-readable report and, optionally, a machine-readable JSON that can be
fed directly to mandeye_bag_convert.py via its --audit-json flag.

Key capabilities:
  - Bag-time (recording timestamp) tracking for ALL messages — eliminates
    sampling-asymmetry artefacts when IMU and PC have very different rates
  - Clock-axis analysis: bag_time vs header.stamp vs Livox timebase per topic,
    with cross-topic clock-domain comparison (same / different clock)
  - C1/C2/C6 scoring uses overlapping time windows and full bag_time ranges
    instead of only sampled header.stamp windows
  - IMU unit autodetect (m/s² vs g vs mg vs mm/s², rad/s vs deg/s)
    with scale factors exported to JSON for use by mandeye_bag_convert.py
  - Livox multi-timestamp analysis (header.stamp vs timebase vs offset_time)
  - PointCloud2 per-point timestamp data-buffer verification
  - Explicit deserialization error tracking (ok / fallback / fail per topic)
  - Graceful handling of truncated/corrupt bag files (partial data recovery)
  - Multi-volume / split bag sequence detection (--sequence / --no-sequence)
  - JSON export for pipeline integration with mandeye_bag_convert.py

JSON export includes per-topic IMU unit info (acc_unit, acc_scale, gyro_unit,
gyro_scale) which mandeye_bag_convert.py reads to skip auto-detection and
apply correct conversion factors (output: Acc → g, Gyro → deg/s).

Dependencies:  pip install rosbags numpy

Usage:
  python mandeye_bag_audit.py <bag_path> [options]
  python mandeye_bag_audit.py recording.bag -v
  python mandeye_bag_audit.py recording.bag --json audit.json
  python mandeye_bag_audit.py recording.bag --max_msgs 5000 --top 10

Multi-volume sequences:
  python mandeye_bag_audit.py recording_0.bag --sequence
  python mandeye_bag_audit.py ./bag_directory/

Pipeline (audit → convert with auto unit conversion):
  python mandeye_bag_audit.py  recording.bag --json audit.json
  python mandeye_bag_convert.py recording.bag output ros1-to-hdmapping --audit-json audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mandeye_bag_common import (
    Reader1, Reader1Error, Reader2,
    deserialize_cdr, deserialize_ros1, ros1_to_cdr,
    deserialize_ros1_tracked,
    rostime_to_sec as _rostime_sec,
    is_imu_type as _is_imu_type,
    is_pc_type as _is_pc_type,
    is_custom_msg as _is_custom_msg,
    guess_acc_unit as _guess_acc_unit,
    guess_gyro_unit as _guess_gyro_unit,
    _G,
    extract_seq_prefix as _extract_seq_prefix,
    detect_bag_sequence,
    SequenceInfo,
    validate_bag_sequence,
    print_sequence_summary as _print_sequence_summary,
    add_sequence_args,
    detect_dir_bags,
)

# ---------------------------------------------------------------------------
# Deserialization with explicit error tracking
# ---------------------------------------------------------------------------
_DS_OK = "ok"
_DS_FALLBACK = "fallback"
_DS_FAIL = "fail"

# Aliases — the tracked variant from mandeye_bag_common returns (msg|None, status)
_audit_deserialize_ros1 = deserialize_ros1_tracked


def _audit_deserialize_cdr(raw: bytes, msgtype: str):
    """CDR deserializer with fail tracking."""
    try:
        msg = deserialize_cdr(raw, msgtype)
        return msg, _DS_OK
    except Exception:
        return None, _DS_FAIL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _header_ts(msg) -> Optional[float]:
    try:
        return _rostime_sec(msg.header.stamp)
    except Exception:
        return None


# PointCloud2 field-type sizes (ROS datatype enum → (struct_char, byte_size))
_PC2_DTYPE = {
    1: ("b", 1),   # INT8
    2: ("B", 1),   # UINT8
    3: ("h", 2),   # INT16
    4: ("H", 2),   # UINT16
    5: ("i", 4),   # INT32
    6: ("I", 4),   # UINT32
    7: ("f", 4),   # FLOAT32
    8: ("d", 8),   # FLOAT64
}


def _pc2_read_time_field(msg, max_points: int = 64) -> Optional[List[float]]:
    """Actually parse the PointCloud2 data buffer and extract values from the
    first time-like field.  Returns a list of floats or None.

    This avoids false-positives/negatives from only checking field names.
    """
    # Identify candidate time field
    time_names = {
        "t", "time", "timestamp", "timestamps", "time_stamp",
        "offset_time", "stamp", "offset_us", "offset_ns",
    }
    time_field = None
    for f in msg.fields:
        if f.name.lower() in time_names:
            time_field = f
            break
    if time_field is None:
        return None

    dtype_info = _PC2_DTYPE.get(time_field.datatype)
    if dtype_info is None:
        return None

    fmt_char, fsize = dtype_info
    point_step = msg.point_step
    data = bytes(msg.data)
    n_pts = min(max_points, msg.width * msg.height)
    offset = time_field.offset

    values: List[float] = []
    for i in range(n_pts):
        start = i * point_step + offset
        end = start + fsize
        if end > len(data):
            break
        val = struct.unpack_from(f"<{fmt_char}", data, start)[0]
        values.append(float(val))

    return values if values else None


# _guess_acc_unit, _guess_gyro_unit imported from mandeye_bag_common

# ---------------------------------------------------------------------------
# Data collected per topic
# ---------------------------------------------------------------------------
@dataclass
class TopicStats:
    topic: str
    msgtype: str
    msgcount: int

    # Sampling
    header_timestamps: List[float] = field(default_factory=list)
    frame_ids: List[str] = field(default_factory=list)

    # Deserialisation tracking
    deser_ok: int = 0
    deser_fallback: int = 0
    deser_fail: int = 0

    # --- IMU ---
    acc_magnitudes: List[float] = field(default_factory=list)     # raw
    gyro_magnitudes: List[float] = field(default_factory=list)    # raw
    acc_unit: str = "?"
    acc_scale: float = 1.0          # multiply raw to get m/s²
    gyro_unit: str = "?"
    gyro_scale: float = 1.0        # multiply raw to get rad/s

    # --- PC general ---
    point_counts: List[int] = field(default_factory=list)
    has_per_point_ts: Optional[bool] = None
    per_point_ts_range: List[float] = field(default_factory=list)
    per_point_ts_verified: Optional[bool] = None   # True if data-buffer parsing confirmed
    per_point_ts_field: str = ""                    # field name
    n_fields: int = 0
    field_names: List[str] = field(default_factory=list)

    # --- Livox CustomMsg multi-timestamp ---
    livox_timebases: List[float] = field(default_factory=list)          # timebase (sec)
    livox_point_time_min: List[float] = field(default_factory=list)     # timebase + min(offset) sec
    livox_point_time_max: List[float] = field(default_factory=list)     # timebase + max(offset) sec

    # --- Bag-time tracking (recording timestamps, no deserialization needed) ---
    bag_timestamps: List[float] = field(default_factory=list)   # bag_time for sampled msgs
    bag_time_start: float = 0.0     # earliest bag_time across ALL msgs
    bag_time_end: float = 0.0       # latest bag_time across ALL msgs
    bag_time_count: int = 0         # total relevant msgs seen
    bag_header_offset_med: float = 0.0   # median(bag_time - header.stamp) on samples
    bag_header_offset_std: float = 0.0

    # --- Derived (computed after sampling) ---
    epoch_sec: float = 0.0
    duration_sec: float = 0.0
    frequency_hz: float = 0.0
    dt_mean: float = 0.0
    dt_std: float = 0.0
    monotone: bool = True
    dominant_frame_id: str = ""
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------
@dataclass
class PairScore:
    pc_topic: str
    imu_topic: str
    scores: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    total: float = 0.0
    imu_time_offset_sec: float = 0.0
    recommended_pc_clock: str = ""   # "header" / "timebase" / "point_time"

    WEIGHTS = {
        "time_domain": 20,
        "delta_t_offset": 20,
        "hw_sync": 5,           # reduced: purely heuristic
        "freq_density": 15,
        "kinematic": 15,        # increased: now unit-aware, more reliable
        "correlation": 15,
        "per_point_ts": 10,
    }

    def compute_total(self):
        w_sum = sum(self.WEIGHTS.values())
        self.total = sum(
            self.scores.get(k, 0) * w for k, w in self.WEIGHTS.items()
        ) / w_sum


# ============================================================================
# SAMPLING
# ============================================================================
def _sample_bag(bag_path: Path, max_msgs: int, is_ros1: bool) -> Dict[str, TopicStats]:
    """Open the bag and sample up to max_msgs per topic."""
    topics: Dict[str, TopicStats] = {}
    counters: Dict[str, int] = {}

    ReaderCls = Reader1 if is_ros1 else Reader2

    # First pass: gather connections
    with ReaderCls(bag_path) as reader:
        for c in reader.connections:
            if c.topic not in topics:
                cnt = getattr(c, "msgcount", 0) or 0
                topics[c.topic] = TopicStats(
                    topic=c.topic, msgtype=c.msgtype, msgcount=cnt,
                )
                counters[c.topic] = 0

    # Keep only IMU / PC for deep sampling
    relevant = {
        t: s for t, s in topics.items()
        if _is_imu_type(s.msgtype) or _is_pc_type(s.msgtype)
    }
    if not relevant:
        return topics

    deser_fn = _audit_deserialize_ros1 if is_ros1 else _audit_deserialize_cdr
    pc2_ts_checked: Dict[str, bool] = {}   # topic → already checked data buffer?

    # Second pass: sample messages
    truncated = False
    with ReaderCls(bag_path) as reader:
      try:
        for conn, _timestamp, rawdata in reader.messages():
            if conn.topic not in relevant:
                continue
            stats = relevant[conn.topic]

            # Track bag_time for ALL relevant messages (free, no deser)
            bt_sec = _timestamp / 1e9
            if stats.bag_time_count == 0:
                stats.bag_time_start = bt_sec
            stats.bag_time_end = bt_sec
            stats.bag_time_count += 1

            if counters[conn.topic] >= max_msgs:
                continue
            counters[conn.topic] += 1

            msg, status = deser_fn(rawdata, conn.msgtype)
            # Track deser status
            if status == _DS_OK:
                stats.deser_ok += 1
            elif status == _DS_FALLBACK:
                stats.deser_fallback += 1
            else:
                stats.deser_fail += 1
                continue   # skip unparseable

            ts = _header_ts(msg)
            if ts is not None and ts > 0:
                stats.header_timestamps.append(ts)
                stats.bag_timestamps.append(bt_sec)

            try:
                stats.frame_ids.append(msg.header.frame_id)
            except Exception:
                pass

            # ---- IMU ----
            if _is_imu_type(conn.msgtype):
                try:
                    ax = msg.linear_acceleration.x
                    ay = msg.linear_acceleration.y
                    az = msg.linear_acceleration.z
                    stats.acc_magnitudes.append(math.sqrt(ax*ax + ay*ay + az*az))
                    gx = msg.angular_velocity.x
                    gy = msg.angular_velocity.y
                    gz = msg.angular_velocity.z
                    stats.gyro_magnitudes.append(math.sqrt(gx*gx + gy*gy + gz*gz))
                except Exception:
                    pass

            # ---- PointCloud ----
            if _is_pc_type(conn.msgtype):
                if _is_custom_msg(conn.msgtype):
                    # -- Livox CustomMsg --
                    try:
                        n_pts = len(msg.points)
                        stats.point_counts.append(n_pts)
                        tb_sec = float(msg.timebase) / 1e9
                        stats.livox_timebases.append(tb_sec)
                        if n_pts > 0:
                            offsets = np.array(
                                [p.offset_time for p in msg.points], dtype=np.float64
                            )
                            pt_min = tb_sec + float(np.min(offsets)) / 1e9
                            pt_max = tb_sec + float(np.max(offsets)) / 1e9
                            stats.livox_point_time_min.append(pt_min)
                            stats.livox_point_time_max.append(pt_max)
                            if float(np.max(offsets)) > 0:
                                stats.has_per_point_ts = True
                                stats.per_point_ts_range.append(
                                    float(np.max(offsets) - np.min(offsets)) / 1e9
                                )
                            elif stats.has_per_point_ts is None:
                                stats.has_per_point_ts = False
                    except Exception:
                        pass
                else:
                    # -- PointCloud2 --
                    try:
                        n_pts = msg.width * msg.height
                        stats.point_counts.append(n_pts)
                        fnames = [f.name for f in msg.fields]
                        stats.field_names = fnames
                        stats.n_fields = len(fnames)

                        # Data-buffer verification (once per topic)
                        if conn.topic not in pc2_ts_checked:
                            pc2_ts_checked[conn.topic] = True
                            vals = _pc2_read_time_field(msg, max_points=64)
                            if vals is not None and len(vals) >= 4:
                                arr = np.array(vals)
                                # Check: values should vary (not all-zero)
                                val_range = float(np.ptp(arr))
                                non_zero = float(np.count_nonzero(arr))
                                has_variation = val_range > 0 and non_zero > len(arr) * 0.5
                                if has_variation:
                                    stats.has_per_point_ts = True
                                    stats.per_point_ts_verified = True
                                    # Find which field
                                    for f in msg.fields:
                                        if f.name.lower() in {
                                            "t", "time", "timestamp", "timestamps",
                                            "time_stamp", "offset_time", "stamp",
                                            "offset_us", "offset_ns",
                                        }:
                                            stats.per_point_ts_field = f.name
                                            break
                                else:
                                    stats.per_point_ts_verified = False
                            elif vals is not None:
                                # Field exists but very few values
                                stats.per_point_ts_verified = False
                            else:
                                # No time-like field at all
                                if stats.has_per_point_ts is None:
                                    stats.has_per_point_ts = False
                                    stats.per_point_ts_verified = False
                    except Exception:
                        pass

      except (Reader1Error, Exception) as exc:
          truncated = True
          print(f"WARNING: Bag file read error (truncated/corrupt?): {exc}",
                file=sys.stderr)
          print("         Continuing with data read so far ...", file=sys.stderr)

    if truncated:
        # Mark all topics so the report can flag it
        for stats in relevant.values():
            stats.notes.append("BAG FILE TRUNCATED – partial data only")

    # ---- Post-sampling derivations ----
    for stats in relevant.values():
        ts = np.array(stats.header_timestamps)
        if len(ts) > 1:
            stats.epoch_sec = float(np.median(ts))
            stats.duration_sec = float(ts[-1] - ts[0])
            diffs = np.diff(ts)
            stats.dt_mean = float(np.mean(diffs))
            stats.dt_std = float(np.std(diffs))
            stats.frequency_hz = 1.0 / stats.dt_mean if stats.dt_mean > 0 else 0.0
            stats.monotone = bool(np.all(diffs >= 0))
        elif len(ts) == 1:
            stats.epoch_sec = float(ts[0])

        # Compute bag-header offset
        if stats.bag_timestamps and stats.header_timestamps:
            n = min(len(stats.bag_timestamps), len(stats.header_timestamps))
            bh_offsets = (
                np.array(stats.bag_timestamps[:n])
                - np.array(stats.header_timestamps[:n])
            )
            stats.bag_header_offset_med = float(np.median(bh_offsets))
            stats.bag_header_offset_std = float(np.std(bh_offsets))

        if stats.frame_ids:
            c = Counter(stats.frame_ids)
            stats.dominant_frame_id = c.most_common(1)[0][0]

        # IMU unit autodetect
        if _is_imu_type(stats.msgtype):
            if stats.acc_magnitudes:
                stats.acc_unit, stats.acc_scale = _guess_acc_unit(
                    np.array(stats.acc_magnitudes)
                )
            if stats.gyro_magnitudes:
                stats.gyro_unit, stats.gyro_scale = _guess_gyro_unit(
                    np.array(stats.gyro_magnitudes),
                    np.array(stats.acc_magnitudes) if stats.acc_magnitudes else None,
                )

    topics.update(relevant)
    return topics


# ============================================================================
# CRITERION SCORERS
# ============================================================================

def _score_time_domain(pc: TopicStats, imu: TopicStats) -> Tuple[float, str]:
    """C1: Same epoch, both monotone, low jitter.
    Uses bag_time ranges (full coverage) for overlap when available,
    falling back to sampled header.stamp.
    """
    score = 1.0
    notes: List[str] = []

    if not pc.header_timestamps or not imu.header_timestamps:
        return 0.0, "No timestamps sampled"

    # Determine whether bag_time is available for full-range overlap
    use_bag_time = (pc.bag_time_count > 0 and imu.bag_time_count > 0)

    if use_bag_time:
        # Check if both topics' header clocks are in the same domain
        bh_diff = abs(pc.bag_header_offset_med - imu.bag_header_offset_med)
        same_header_clock = (bh_diff < 1.0)

        # Overlap from bag_time (covers ALL messages, avoids sampling asymmetry)
        bt_pc_min, bt_pc_max = pc.bag_time_start, pc.bag_time_end
        bt_imu_min, bt_imu_max = imu.bag_time_start, imu.bag_time_end
        overlap = max(0, min(bt_pc_max, bt_imu_max) - max(bt_pc_min, bt_imu_min))
        span = max(bt_pc_max, bt_imu_max) - min(bt_pc_min, bt_imu_min)
        ov_ratio = overlap / span if span > 0 else 0

        # Epoch from bag_time midpoints (avoids sampling-window asymmetry)
        bt_pc_epoch = (bt_pc_min + bt_pc_max) / 2
        bt_imu_epoch = (bt_imu_min + bt_imu_max) / 2
        epoch_diff = abs(bt_pc_epoch - bt_imu_epoch)

        if same_header_clock:
            notes.append(
                f"Same header clock (\u0394 bag-hdr offset={bh_diff*1000:.1f}ms)"
            )
        else:
            notes.append(
                f"Different header clocks (\u0394 bag-hdr offset={bh_diff:.1f}s)"
            )
    else:
        epoch_diff = abs(pc.epoch_sec - imu.epoch_sec)
        pc_min, pc_max = min(pc.header_timestamps), max(pc.header_timestamps)
        imu_min, imu_max = min(imu.header_timestamps), max(imu.header_timestamps)
        overlap = max(0, min(pc_max, imu_max) - max(pc_min, imu_min))
        span = max(pc_max, imu_max) - min(pc_min, imu_min)
        ov_ratio = overlap / span if span > 0 else 0

    if epoch_diff > 3600:
        score *= 0.0
        notes.append(f"Epoch diff {epoch_diff:.0f}s (>1 h!)")
    elif epoch_diff > 60:
        score *= 0.3
        notes.append(f"Epoch diff {epoch_diff:.0f}s (>1 min)")
    elif epoch_diff > 10:
        score *= 0.7
        notes.append(f"Epoch diff {epoch_diff:.1f}s")
    else:
        notes.append(f"Epoch diff {epoch_diff:.3f}s OK")

    if ov_ratio < 0.5:
        score *= 0.2
        notes.append(f"Overlap {ov_ratio:.0%} (poor)")
    elif ov_ratio < 0.9:
        score *= 0.7
        notes.append(f"Overlap {ov_ratio:.0%}")
    else:
        notes.append(f"Overlap {ov_ratio:.0%} OK")

    if use_bag_time:
        notes.append(
            f"(bag_time overlap, {pc.bag_time_count}+{imu.bag_time_count} total msgs)"
        )

    if not pc.monotone:
        score *= 0.5
        notes.append("PC ts NOT monotone")
    if not imu.monotone:
        score *= 0.5
        notes.append("IMU ts NOT monotone")

    for label, st in [("PC", pc), ("IMU", imu)]:
        if st.dt_mean > 0:
            cv = st.dt_std / st.dt_mean
            if cv > 0.5:
                score *= 0.5
                notes.append(f"{label} jitter CV={cv:.2f} (high)")
            elif cv > 0.1:
                score *= 0.8
                notes.append(f"{label} jitter CV={cv:.2f}")

    return score, "; ".join(notes)


def _nearest_offset(src_ts: np.ndarray, ref_ts: np.ndarray,
                     max_src: int = 500) -> np.ndarray:
    """For each timestamp in src (up to max_src), find nearest in ref and
    return the array of offsets (src - nearest_ref)."""
    ref_sorted = np.sort(ref_ts)
    deltas = []
    for pt in src_ts[:max_src]:
        idx = np.searchsorted(ref_sorted, pt)
        cands = []
        if idx > 0:
            cands.append(ref_sorted[idx - 1])
        if idx < len(ref_sorted):
            cands.append(ref_sorted[idx])
        if cands:
            nearest = min(cands, key=lambda x: abs(x - pt))
            deltas.append(pt - nearest)
    return np.array(deltas) if deltas else np.array([])


def _offset_quality(deltas: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (median, std, max_dev, score) for an array of offsets."""
    if len(deltas) == 0:
        return 0.0, 999.0, 999.0, 0.0
    med = float(np.median(deltas))
    std = float(np.std(deltas))
    maxdev = float(np.max(np.abs(deltas - med)))
    score = 1.0
    if std > 0.5:
        score *= 0.1
    elif std > 0.1:
        score *= 0.5
    elif std > 0.01:
        score *= 0.8
    if maxdev > 1.0:
        score *= 0.3
    elif maxdev > 0.1:
        score *= 0.7
    return med, std, maxdev, score


def _score_delta_t_offset(
    pc: TopicStats, imu: TopicStats,
) -> Tuple[float, str, float, str]:
    """C2: Constant Δt offset.  For Livox topics also evaluates timebase and
    point-time clocks.  Restricts analysis to the overlapping time window
    to avoid sampling-asymmetry artefacts.
    Returns (score, note, best_offset_sec, best_clock_source).
    """
    if not pc.header_timestamps or not imu.header_timestamps:
        return 0.0, "No timestamps", 0.0, "header"

    imu_ts = np.array(imu.header_timestamps)

    def _windowed_offset_quality(
        src_ts: np.ndarray, ref_ts: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Restrict to overlapping window, then compute offset quality."""
        ov_start = max(float(src_ts[0]), float(ref_ts[0]))
        ov_end = min(float(src_ts[-1]), float(ref_ts[-1]))
        if ov_end <= ov_start:
            return None
        src_ov = src_ts[(src_ts >= ov_start) & (src_ts <= ov_end)]
        ref_ov = ref_ts[(ref_ts >= ov_start) & (ref_ts <= ov_end)]
        if len(src_ov) < 2 or len(ref_ov) < 2:
            return None
        d = _nearest_offset(src_ov, ref_ov)
        if len(d) == 0:
            return None
        return _offset_quality(d)

    # Evaluate multiple clock sources for the PC side
    clock_results: Dict[str, Tuple[float, float, float, float]] = {}

    # (a) header.stamp  (always available)
    header_ts = np.array(pc.header_timestamps)
    res = _windowed_offset_quality(header_ts, imu_ts)
    if res is not None:
        clock_results["header"] = res

    # (b) Livox timebase
    if pc.livox_timebases:
        tb_ts = np.array(pc.livox_timebases)
        res = _windowed_offset_quality(tb_ts, imu_ts)
        if res is not None:
            clock_results["timebase"] = res

    # (c) Livox mean point-time  (timebase + mean(offset_time))
    if pc.livox_point_time_min and pc.livox_point_time_max:
        pt_mid = (np.array(pc.livox_point_time_min) +
                  np.array(pc.livox_point_time_max)) / 2.0
        res = _windowed_offset_quality(pt_mid, imu_ts)
        if res is not None:
            clock_results["point_time"] = res

    if not clock_results:
        # No overlap in any clock — check if same clock via bag_time
        if (pc.bag_time_count > 0 and imu.bag_time_count > 0
                and abs(pc.bag_header_offset_med - imu.bag_header_offset_med) < 1.0):
            return (0.8,
                    "Same clock (bag-time confirms) but sample windows don't "
                    "overlap; try increasing --max_msgs",
                    0.0, "header")
        return 0.0, "No overlap for Δt analysis", 0.0, "header"

    # Pick the clock with best score (then lowest std as tie-breaker)
    best_clock = max(
        clock_results,
        key=lambda k: (clock_results[k][3], -clock_results[k][1]),
    )
    med, std, maxdev, score = clock_results[best_clock]

    notes = [f"Best clock: {best_clock}"]
    notes.append(f"Median offset={med*1000:.2f}ms, STD={std*1000:.2f}ms")

    # Annotate all clocks if Livox multi-clock is available
    if len(clock_results) > 1:
        for ck, (cm, cs, _, csc) in clock_results.items():
            tag = " <<<" if ck == best_clock else ""
            notes.append(
                f"  {ck}: offset={cm*1000:.2f}ms "
                f"std={cs*1000:.2f}ms score={csc:.0%}{tag}"
            )

    if std > 0.5:
        notes.append("Very unstable offset")
    elif std > 0.1:
        notes.append("Noisy offset")
    elif std < 0.01:
        notes.append("Excellent stability")

    return score, "; ".join(notes), med, best_clock


def _score_hw_sync(pc: TopicStats, imu: TopicStats) -> Tuple[float, str]:
    """C3: Hardware sync likelihood — HEURISTIC ONLY.
    Uses frame_id prefix and topic namespace.  Cannot confirm actual clock sync.
    """
    score = 0.5
    notes = ["(heuristic only — cannot confirm HW clock sync)"]

    pc_fid = pc.dominant_frame_id.lower()
    imu_fid = imu.dominant_frame_id.lower()

    if pc_fid and imu_fid:
        if pc_fid == imu_fid:
            score = 0.8
            notes.append(f"Same frame_id '{pc.dominant_frame_id}'")
        else:
            common = ""
            for a, b in zip(pc_fid, imu_fid):
                if a == b:
                    common += a
                else:
                    break
            common = common.rstrip("_/- ")
            if common and len(common) >= 3:
                score = 0.7
                notes.append(
                    f"Frame IDs share prefix '{common}' "
                    f"('{pc.dominant_frame_id}' / '{imu.dominant_frame_id}')"
                )
            else:
                score = 0.3
                notes.append(
                    f"Different frame_ids: '{pc.dominant_frame_id}' vs "
                    f"'{imu.dominant_frame_id}'"
                )
    elif not pc_fid and not imu_fid:
        notes.append("No frame_id on either topic")
    else:
        notes.append(
            f"frame_id: PC='{pc.dominant_frame_id or '(empty)'}', "
            f"IMU='{imu.dominant_frame_id or '(empty)'}'"
        )

    pc_ns = pc.topic.strip("/").split("/")
    imu_ns = imu.topic.strip("/").split("/")
    if len(pc_ns) >= 2 and len(imu_ns) >= 2:
        if pc_ns[0] == imu_ns[0]:
            score = min(score + 0.2, 1.0)
            notes.append(f"Same namespace '/{pc_ns[0]}'")
        else:
            notes.append(f"Diff namespaces '/{pc_ns[0]}' vs '/{imu_ns[0]}'")

    return score, "; ".join(notes)


def _score_freq_density(pc: TopicStats, imu: TopicStats) -> Tuple[float, str]:
    """C4: Frequency and density checks."""
    score = 1.0
    notes: List[str] = []

    if imu.frequency_hz > 0:
        if imu.frequency_hz >= 100:
            notes.append(f"IMU {imu.frequency_hz:.0f} Hz OK")
        elif imu.frequency_hz >= 50:
            score *= 0.7
            notes.append(f"IMU {imu.frequency_hz:.0f} Hz (low, want ≥100)")
        else:
            score *= 0.3
            notes.append(f"IMU {imu.frequency_hz:.1f} Hz (very low)")
    else:
        score *= 0.1
        notes.append("IMU freq unknown")

    if pc.frequency_hz > 0:
        if 3 <= pc.frequency_hz <= 50:
            notes.append(f"PC {pc.frequency_hz:.1f} Hz OK")
        elif pc.frequency_hz > 50:
            score *= 0.8
            notes.append(f"PC {pc.frequency_hz:.1f} Hz (unusually high)")
        else:
            score *= 0.5
            notes.append(f"PC {pc.frequency_hz:.2f} Hz (low)")
    else:
        score *= 0.1
        notes.append("PC freq unknown")

    if pc.point_counts:
        avg_pts = float(np.mean(pc.point_counts))
        if avg_pts >= 1000:
            notes.append(f"Avg {avg_pts:.0f} pts/msg OK")
        elif avg_pts >= 100:
            score *= 0.7
            notes.append(f"Avg {avg_pts:.0f} pts/msg (sparse)")
        else:
            score *= 0.3
            notes.append(f"Avg {avg_pts:.0f} pts/msg (very sparse)")

    if pc.frequency_hz > 0 and imu.frequency_hz > 0:
        ratio = imu.frequency_hz / pc.frequency_hz
        if ratio >= 5:
            notes.append(f"IMU/PC ratio {ratio:.0f}:1 OK")
        else:
            score *= 0.6
            notes.append(f"IMU/PC ratio {ratio:.1f}:1 (low, want ≥5:1)")

    return score, "; ".join(notes)


def _score_kinematic(pc: TopicStats, imu: TopicStats) -> Tuple[float, str]:
    """C5: Kinematic consistency — unit-aware.
    Autodetects m/s² vs g (and rad/s vs deg/s), normalises to SI, then checks
    that gravity ≈ 9.81 m/s² and gyro is alive.
    """
    score = 1.0
    notes: List[str] = []

    # --- Accelerometer ---
    if imu.acc_magnitudes:
        raw = np.array(imu.acc_magnitudes)
        raw_mean = float(np.mean(raw))
        unit = imu.acc_unit
        scale = imu.acc_scale

        si = raw * scale
        si_mean = float(np.mean(si))
        si_std = float(np.std(si))

        if unit == "?":
            # Unknown unit — try to score raw, but flag it
            notes.append(
                f"|acc| raw={raw_mean:.2f} (unit unknown — cannot verify gravity)"
            )
            # Still give partial credit if it's vaguely in the right ballpark
            if 7.0 <= raw_mean <= 12.5:
                notes.append("  (looks like m/s² anyway)")
            elif 0.7 <= raw_mean <= 1.4:
                notes.append("  (looks like g)")
            else:
                score *= 0.5
                notes.append("  (does not match m/s² or g)")
        else:
            notes.append(f"|acc| raw={raw_mean:.3f} {unit} → {si_mean:.2f}±{si_std:.2f} m/s²")
            if 8.5 <= si_mean <= 11.0:
                notes.append("  Gravity check OK")
            elif 6.0 <= si_mean <= 14.0:
                score *= 0.6
                notes.append("  Gravity unusual (expect ≈9.81 m/s²)")
            else:
                score *= 0.2
                notes.append("  Gravity FAIL (wrong scale / broken sensor?)")
    else:
        score *= 0.3
        notes.append("No accel data")

    # --- Gyroscope ---
    if imu.gyro_magnitudes:
        raw = np.array(imu.gyro_magnitudes)
        raw_mean = float(np.mean(raw))
        unit = imu.gyro_unit
        scale = imu.gyro_scale

        si = raw * scale
        si_mean = float(np.mean(si))

        if unit == "?":
            notes.append(f"|gyro| raw={raw_mean:.4f} (unit unknown)")
        else:
            notes.append(f"|gyro| raw={raw_mean:.4f} {unit} → {si_mean:.5f} rad/s")

        # Check stuck-at-zero
        if si_mean < 1e-8:
            score *= 0.3
            notes.append("  Gyro stuck at 0!")
        elif si_mean < 0.001:
            notes.append("  Very still, but alive")
    else:
        score *= 0.3
        notes.append("No gyro data")

    return score, "; ".join(notes)


def _score_correlation(pc: TopicStats, imu: TopicStats) -> Tuple[float, str]:
    """C6: Binned offset drift — checks clock stability across recording.
    Restricts analysis to the overlapping header-time window.
    """
    if len(pc.header_timestamps) < 10 or len(imu.header_timestamps) < 10:
        return 0.5, "Insufficient samples for correlation"

    pc_ts_all = np.array(pc.header_timestamps)
    imu_ts_all = np.array(imu.header_timestamps)

    # Restrict to overlapping header-time window
    ov_start = max(float(pc_ts_all[0]), float(imu_ts_all[0]))
    ov_end = min(float(pc_ts_all[-1]), float(imu_ts_all[-1]))

    if ov_end <= ov_start:
        if (pc.bag_time_count > 0 and imu.bag_time_count > 0
                and abs(pc.bag_header_offset_med - imu.bag_header_offset_med) < 1.0):
            return 0.8, "Same clock but sample windows don't overlap; try --max_msgs"
        return 0.3, "No overlap in header timestamps for correlation"

    pc_ts = pc_ts_all[(pc_ts_all >= ov_start) & (pc_ts_all <= ov_end)]
    imu_ts = imu_ts_all[(imu_ts_all >= ov_start) & (imu_ts_all <= ov_end)]

    n_bins = min(20, len(pc_ts) // 5)
    if n_bins < 3:
        return 0.5, "Too few PC messages in overlap window for binned analysis"

    edges = np.linspace(ov_start, ov_end, n_bins + 1)
    bin_offsets = []

    imu_sorted = np.sort(imu_ts)
    for i in range(n_bins):
        mask = (pc_ts >= edges[i]) & (pc_ts < edges[i + 1])
        bp = pc_ts[mask]
        if len(bp) == 0:
            continue
        pt = bp[0]
        idx = np.searchsorted(imu_sorted, pt)
        cands = []
        if idx > 0:
            cands.append(imu_sorted[idx - 1])
        if idx < len(imu_sorted):
            cands.append(imu_sorted[idx])
        if cands:
            nearest = min(cands, key=lambda x: abs(x - pt))
            bin_offsets.append(pt - nearest)

    if len(bin_offsets) < 3:
        return 0.5, "Too few bins with data"

    bo = np.array(bin_offsets)
    drift = float(np.ptp(bo))

    notes = [f"Drift across recording: {drift*1000:.2f}ms"]
    score = 1.0
    if drift > 1.0:
        score *= 0.1
        notes.append("Severe clock drift (>1 s)")
    elif drift > 0.1:
        score *= 0.5
        notes.append("Moderate drift (>100 ms)")
    elif drift > 0.01:
        score *= 0.8
        notes.append("Small drift")
    else:
        notes.append("Stable synchronisation")

    return score, "; ".join(notes)


def _score_per_point_ts(pc: TopicStats) -> Tuple[float, str]:
    """C7: Per-point timestamps (bonus).
    For PointCloud2, requires data-buffer verification (not just field name).
    """
    # Livox CustomMsg — directly verified via offset_time
    if _is_custom_msg(pc.msgtype):
        if pc.has_per_point_ts:
            if pc.per_point_ts_range:
                avg_r = float(np.mean(pc.per_point_ts_range))
                return 1.0, f"Livox offset_time present (intra-msg span ≈ {avg_r*1000:.1f} ms)"
            return 1.0, "Livox offset_time confirmed"
        return 0.0, "Livox CustomMsg but offset_time all zero"

    # PointCloud2 — require buffer verification
    if pc.per_point_ts_verified is True:
        note = f"Per-pt ts verified in data buffer (field '{pc.per_point_ts_field}')"
        return 1.0, note
    elif pc.per_point_ts_verified is False and pc.has_per_point_ts:
        # Field name matched but data didn't look like timestamps
        return 0.2, (
            f"Field '{pc.per_point_ts_field or '?'}' exists but data-buffer "
            "values look constant / zero — possible false-positive"
        )
    return 0.0, "No per-point timestamps detected"


# ============================================================================
# MAIN SCORING
# ============================================================================
def score_pair(pc: TopicStats, imu: TopicStats) -> PairScore:
    ps = PairScore(pc_topic=pc.topic, imu_topic=imu.topic)

    s, n = _score_time_domain(pc, imu)
    ps.scores["time_domain"] = s
    ps.notes["time_domain"] = n

    s, n, offset, clock = _score_delta_t_offset(pc, imu)
    ps.scores["delta_t_offset"] = s
    ps.notes["delta_t_offset"] = n
    ps.imu_time_offset_sec = offset
    ps.recommended_pc_clock = clock

    s, n = _score_hw_sync(pc, imu)
    ps.scores["hw_sync"] = s
    ps.notes["hw_sync"] = n

    s, n = _score_freq_density(pc, imu)
    ps.scores["freq_density"] = s
    ps.notes["freq_density"] = n

    s, n = _score_kinematic(pc, imu)
    ps.scores["kinematic"] = s
    ps.notes["kinematic"] = n

    s, n = _score_correlation(pc, imu)
    ps.scores["correlation"] = s
    ps.notes["correlation"] = n

    s, n = _score_per_point_ts(pc)
    ps.scores["per_point_ts"] = s
    ps.notes["per_point_ts"] = n

    ps.compute_total()
    return ps


# ============================================================================
# REPORT
# ============================================================================
CRITERION_LABELS = {
    "time_domain":    "1. Time-domain alignment",
    "delta_t_offset": "2. Constant dt offset",
    "hw_sync":        "3. HW sync (heuristic)",
    "freq_density":   "4. Freq & density",
    "kinematic":      "5. Kinematic (unit-aware)",
    "correlation":    "6. Numerical correlation",
    "per_point_ts":   "7. Per-point timestamps",
}


def _grade(total: float) -> str:
    if total >= 0.85:
        return "EXCELLENT"
    if total >= 0.70:
        return "GOOD"
    if total >= 0.50:
        return "FAIR"
    if total >= 0.30:
        return "POOR"
    return "REJECT"


def _grade_color(total: float) -> str:
    g = _grade(total)
    colors = {
        "EXCELLENT": "\033[92m",
        "GOOD": "\033[32m",
        "FAIR": "\033[33m",
        "POOR": "\033[91m",
        "REJECT": "\033[31m",
    }
    return f"{colors.get(g, '')}{g}\033[0m"


def print_report(
    bag_path: str,
    all_topics: Dict[str, TopicStats],
    pairs: List[PairScore],
    top_k: int,
    verbose: bool,
):
    hr = "=" * 80
    print(f"\n{hr}")
    print("  MANDEYE BAG AUDIT REPORT  v0.6")
    print(f"  Bag: {bag_path}")
    print(hr)

    # ---- Topic listing ----
    print(f"\n  {'TOPIC':<43} {'TYPE':<38} {'#MSGS':>8}")
    print("  " + "-" * 91)
    for t in sorted(all_topics.values(), key=lambda x: x.topic):
        short = t.msgtype.split("/")[-1]
        role = ""
        if _is_pc_type(t.msgtype):
            role = " [PC]"
        elif _is_imu_type(t.msgtype):
            role = " [IMU]"
        print(f"  {t.topic:<43} {short + role:<38} {t.msgcount:>8}")

    # ---- Per-topic stats ----
    relevant = {
        t: s for t, s in all_topics.items()
        if _is_imu_type(s.msgtype) or _is_pc_type(s.msgtype)
    }
    if relevant:
        print(f"\n  {'─' * 76}")
        print("  TOPIC STATISTICS (sampled)")
        print(f"  {'─' * 76}")
        for s in sorted(relevant.values(), key=lambda x: x.topic):
            role = "PC" if _is_pc_type(s.msgtype) else "IMU"
            print(f"\n  [{role}] {s.topic}  ({s.msgtype})")
            total_deser = s.deser_ok + s.deser_fallback + s.deser_fail
            if total_deser > 0:
                parts = []
                if s.deser_ok:
                    parts.append(f"{s.deser_ok} ok")
                if s.deser_fallback:
                    parts.append(f"\033[33m{s.deser_fallback} fallback\033[0m")
                if s.deser_fail:
                    parts.append(f"\033[91m{s.deser_fail} FAIL\033[0m")
                print(f"        Deser:       {' / '.join(parts)}  (of {total_deser})")
            print(f"        Sampled ts:  {len(s.header_timestamps)}")
            if s.bag_time_count > 0:
                bt_dur = s.bag_time_end - s.bag_time_start
                print(
                    f"        Bag time:    {s.bag_time_start:.3f}..{s.bag_time_end:.3f} "
                    f"({bt_dur:.1f}s, {s.bag_time_count} msgs)"
                )
                if s.bag_header_offset_med != 0:
                    print(
                        f"        Bag-Hdr \u0394:   med={s.bag_header_offset_med:.3f}s "
                        f"std={s.bag_header_offset_std*1000:.2f}ms"
                    )
            if s.header_timestamps:
                print(f"        Epoch:       {s.epoch_sec:.3f} s")
                print(f"        Duration:    {s.duration_sec:.2f} s (sampled)")
                print(f"        Frequency:   {s.frequency_hz:.1f} Hz")
                print(f"        dt mean/std: {s.dt_mean*1000:.2f} / {s.dt_std*1000:.2f} ms")
                print(f"        Monotone:    {'YES' if s.monotone else 'NO'}")
            if s.dominant_frame_id:
                print(f"        Frame ID:    '{s.dominant_frame_id}'")
            if s.point_counts:
                avg = float(np.mean(s.point_counts))
                print(f"        Pts/msg:     {avg:.0f} avg, {min(s.point_counts)}..{max(s.point_counts)}")

            # Livox multi-timestamp
            if s.livox_timebases:
                tb = np.array(s.livox_timebases)
                hdr = np.array(s.header_timestamps[:len(tb)])
                diff_hdr_tb = hdr - tb if len(hdr) == len(tb) else None
                print(f"        Livox timebase: min={tb[0]:.6f}  max={tb[-1]:.6f}")
                if diff_hdr_tb is not None and len(diff_hdr_tb) > 0:
                    print(
                        f"        header-timebase offset: "
                        f"med={np.median(diff_hdr_tb)*1000:.2f} ms  "
                        f"std={np.std(diff_hdr_tb)*1000:.2f} ms"
                    )
            if s.livox_point_time_min:
                ptmin = np.array(s.livox_point_time_min)
                ptmax = np.array(s.livox_point_time_max)
                span = ptmax - ptmin
                print(
                    f"        Point-time span/msg: "
                    f"med={np.median(span)*1000:.2f} ms  "
                    f"min={np.min(span)*1000:.2f} ms  "
                    f"max={np.max(span)*1000:.2f} ms"
                )

            # Per-point ts
            if s.has_per_point_ts is not None:
                tag = "YES" if s.has_per_point_ts else "NO"
                extra = ""
                if s.per_point_ts_verified is True:
                    extra = f" (data-buffer VERIFIED, field='{s.per_point_ts_field}')"
                elif s.per_point_ts_verified is False and s.has_per_point_ts:
                    extra = " (field exists but data NOT verified)"
                print(f"        Per-pt ts:   {tag}{extra}")

            # IMU units
            if s.acc_magnitudes:
                raw_m = float(np.mean(s.acc_magnitudes))
                si_m = raw_m * s.acc_scale
                g_m = si_m / _G
                conv_hint = (
                    "already g — no conversion needed"
                    if s.acc_unit == "g"
                    else f"export target: g  → use --acc_unit {s.acc_unit}"
                    if s.acc_unit != "?"
                    else "unit unknown — specify --acc_unit"
                )
                print(
                    f"        |accel|:     {raw_m:.3f} {s.acc_unit} "
                    f"(= {si_m:.2f} m/s²  /  {g_m:.3f} g)  [{conv_hint}]"
                )
            if s.gyro_magnitudes:
                raw_m = float(np.mean(s.gyro_magnitudes))
                si_m = raw_m * s.gyro_scale
                degps_m = si_m * (180.0 / math.pi)
                conv_hint = (
                    "already deg/s — no conversion needed"
                    if s.gyro_unit == "deg/s"
                    else f"export target: deg/s  → use --gyro_unit {s.gyro_unit}"
                    if s.gyro_unit != "?"
                    else "unit unknown — specify --gyro_unit"
                )
                print(
                    f"        |gyro|:      {raw_m:.5f} {s.gyro_unit} "
                    f"(= {si_m:.5f} rad/s  /  {degps_m:.4f} deg/s)  [{conv_hint}]"
                )
            if s.field_names:
                print(f"        PC2 fields:  {', '.join(s.field_names)}")
            if s.notes:
                for note in s.notes:
                    print(f"        \033[93mNOTE: {note}\033[0m")

    # ---- Deserialization warnings ----
    warn_topics = [
        s for s in relevant.values()
        if s.deser_fail > 0 or s.deser_fallback > 0
    ]
    if warn_topics:
        print(f"\n  {'─' * 76}")
        print("  DESERIALIZATION WARNINGS")
        print(f"  {'─' * 76}")
        for s in warn_topics:
            total = s.deser_ok + s.deser_fallback + s.deser_fail
            if s.deser_fail > 0:
                pct = s.deser_fail / total * 100
                print(
                    f"  \033[91mFAIL\033[0m  {s.topic}: {s.deser_fail}/{total} "
                    f"({pct:.0f}%) messages could not be deserialised"
                )
            if s.deser_fallback > 0:
                pct = s.deser_fallback / total * 100
                print(
                    f"  \033[33mWARN\033[0m  {s.topic}: {s.deser_fallback}/{total} "
                    f"({pct:.0f}%) needed ros1→cdr fallback (Python 3.14 compat?)"
                )

    # ---- Clock axis analysis ----
    clock_topics = [
        s for s in relevant.values()
        if s.bag_time_count > 0 and s.header_timestamps
    ]
    if clock_topics:
        print(f"\n  {'\u2500' * 76}")
        print("  CLOCK AXIS ANALYSIS")
        print(f"  {'\u2500' * 76}")
        for s in sorted(clock_topics, key=lambda x: x.topic):
            role = "PC" if _is_pc_type(s.msgtype) else "IMU"
            hdr_min = min(s.header_timestamps)
            hdr_max = max(s.header_timestamps)
            hdr_dur = hdr_max - hdr_min
            bt_dur = s.bag_time_end - s.bag_time_start
            print(f"\n  [{role}] {s.topic}")
            print(
                f"     bag_time     : {s.bag_time_start:.3f} .. "
                f"{s.bag_time_end:.3f}  ({bt_dur:.1f}s, ALL {s.bag_time_count} msgs)"
            )
            print(
                f"     header.stamp : {hdr_min:.3f} .. "
                f"{hdr_max:.3f}  ({hdr_dur:.1f}s, {len(s.header_timestamps)} sampled)"
            )
            print(
                f"     bag-hdr \u0394    : {s.bag_header_offset_med:.3f}s "
                f"\u00b1 {s.bag_header_offset_std*1000:.2f}ms"
            )
            if s.livox_timebases:
                tb = np.array(s.livox_timebases)
                hdr = np.array(s.header_timestamps[:len(tb)])
                if len(hdr) == len(tb):
                    d = hdr - tb
                    print(
                        f"     hdr-timebase : {np.median(d)*1000:.2f}ms "
                        f"\u00b1 {np.std(d)*1000:.2f}ms"
                    )

        # Cross-topic clock comparison
        if len(clock_topics) >= 2:
            offsets = [
                (s.topic, s.bag_header_offset_med) for s in clock_topics
            ]
            print(f"\n  Cross-topic clock comparison:")
            for i in range(len(offsets)):
                for j in range(i + 1, len(offsets)):
                    diff = abs(offsets[i][1] - offsets[j][1])
                    same = "SAME clock" if diff < 1.0 else "DIFFERENT clocks"
                    print(f"     {offsets[i][0]} vs {offsets[j][0]}")
                    print(
                        f"       bag-hdr offset diff = {diff:.3f}s \u2192 {same}"
                    )

    # ---- Pair results ----
    if not pairs:
        print("\n  No valid (PC, IMU) pairs found.")
        return

    pairs_sorted = sorted(pairs, key=lambda p: p.total, reverse=True)
    top = pairs_sorted[:top_k]

    print(f"\n  {'=' * 76}")
    print(f"  TOP {min(top_k, len(top))} PAIRS (of {len(pairs)} evaluated)")
    print(f"  {'=' * 76}")

    for rank, ps in enumerate(top, 1):
        g = _grade_color(ps.total)
        print(f"\n  #{rank}  Score: {ps.total:.1%}  [{g}]")
        print(f"       PC:  {ps.pc_topic}")
        print(f"       IMU: {ps.imu_topic}")
        print(f"       Est. offset: {ps.imu_time_offset_sec*1000:.2f} ms  "
              f"(clock: {ps.recommended_pc_clock})")

        if verbose:
            for key, label in CRITERION_LABELS.items():
                sc = ps.scores.get(key, 0)
                w = PairScore.WEIGHTS[key]
                note = ps.notes.get(key, "")
                bar = "#" * int(sc * 10) + "." * (10 - int(sc * 10))
                print(f"         {label:<28} [{bar}] {sc:.0%} (w={w})")
                if note:
                    for line in textwrap.wrap(note, width=62):
                        print(f"           > {line}")

    # ---- Recommended command ----
    best = top[0]
    if best.total >= 0.30:
        print(f"\n  {'─' * 76}")
        print("  RECOMMENDED COMMAND:")
        # Resolve IMU unit hints for the best pair
        imu_stats = relevant.get(best.imu_topic)
        acc_hint = ""
        gyro_hint = ""
        if imu_stats is not None:
            if imu_stats.acc_unit not in ("g", "?"):
                acc_hint = f" --acc_unit {imu_stats.acc_unit}"
            if imu_stats.gyro_unit not in ("deg/s", "?"):
                gyro_hint = f" --gyro_unit {imu_stats.gyro_unit}"
        print(
            f"  python mandeye_bag_convert.py <bag> <output> ros1-to-hdmapping "
            f"--pointcloud_topic {best.pc_topic} "
            f"--imu_topic {best.imu_topic}"
            f"{acc_hint}{gyro_hint}"
        )
        if acc_hint or gyro_hint:
            print(
                f"        (unit flags added: output will be converted to "
                f"Acc → g, Gyro → deg/s)"
            )
        if abs(best.imu_time_offset_sec) > 0.001:
            print(
                f"        (note: estimated IMU time offset "
                f"~{best.imu_time_offset_sec*1000:.1f} ms, clock={best.recommended_pc_clock})"
            )
    print()


# Sequence detection imported from mandeye_bag_common:
#   _extract_seq_prefix, detect_bag_sequence, SequenceInfo,
#   validate_bag_sequence, _print_sequence_summary


# ============================================================================
# CLI
# ============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Audit a ROS bag for valid (LiDAR, IMU) pairs — HDMapping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Scores every (PointCloud, IMU) pair against 7 weighted criteria:
          1. Time-domain alignment  (bag_time overlap, epoch, monotone, jitter)
          2. Constant dt offset     (multi-clock: header / timebase / point-time;
                                     windowed to overlapping samples only)
          3. HW sync heuristic      (frame_id, namespace — low weight)
          4. Freq & density         (IMU >= 100 Hz, LiDAR >= 5 Hz, point count)
          5. Kinematic consistency  (unit-aware: m/s² vs g, rad/s vs deg/s)
          6. Numerical correlation  (binned offset drift, windowed overlap)
          7. Per-point timestamps   (Livox offset_time / PC2 data-buffer verified)

        Additional report sections:
          - CLOCK AXIS ANALYSIS: bag_time vs header.stamp vs Livox timebase
            per topic, cross-topic same/different clock detection
          - DESERIALIZATION WARNINGS: ok / ros1→cdr fallback / fail counts

        JSON output (--json):
          Writes a machine-readable file containing all topic stats, pair
          scores/notes, and a 'recommended' block with the best pair's
          pointcloud_topic, imu_topic, clock, and offset.  For IMU topics
          the JSON also includes acc_unit, acc_scale, gyro_unit, gyro_scale
          which mandeye_bag_convert.py reads (via --audit-json) to apply
          correct unit conversion (output: Acc → g, Gyro → deg/s).

        Truncated / corrupt bags:
          If the bag file is truncated or corrupt, the tool recovers as
          much data as possible and marks affected topics with a NOTE.

        Examples:
          python mandeye_bag_audit.py recording.bag
          python mandeye_bag_audit.py recording.bag -v
          python mandeye_bag_audit.py recording.bag --json audit.json
          python mandeye_bag_audit.py recording.bag --max_msgs 5000 --top 10
          python mandeye_bag_audit.py ros2_bag_folder/

        Pipeline (audit → convert with auto unit conversion):
          python mandeye_bag_audit.py  rec.bag --json audit.json
          python mandeye_bag_convert.py rec.bag out ros1-to-hdmapping --audit-json audit.json

        Manual unit override (skip auto-detection):
          python mandeye_bag_convert.py rec.bag out ros1-to-hdmapping --acc_unit g --gyro_unit deg/s

        Multi-volume / split bags (--sequence):
          Automatically detects sibling .bag files that belong to the same
          recording (e.g. recording_0.bag, recording_1.bag, …).
          Use --sequence to audit them all and get a combined summary.
          Use --no-sequence to suppress detection and process a single file.
          
          python mandeye_bag_audit.py recording_0.bag --sequence
          python mandeye_bag_audit.py ./recordings/ --sequence
        """),
    )
    p.add_argument("bag", help="Path to ROS1 .bag file, ROS2 bag folder, "
                   "or directory containing .bag files")
    p.add_argument("--max_msgs", type=int, default=2000,
                   help="Max messages to sample per topic (default: 2000)")
    p.add_argument("--top", type=int, default=5,
                   help="Show top K pairs (default: 5)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show detailed per-criterion breakdown")
    p.add_argument("--json", metavar="FILE",
                   help="Write machine-readable audit results to a JSON file")
    add_sequence_args(p)
    args = p.parse_args()

    bag_path = Path(args.bag)

    # --- Variant C: directory of .bag files ---
    dir_bags = detect_dir_bags(bag_path)
    is_dir_of_bags = bool(dir_bags)
    if is_dir_of_bags:
        bag_path = dir_bags[0]  # use first as anchor
        print(f"Directory contains {len(dir_bags)} .bag file(s)")

    is_ros1 = bag_path.suffix == ".bag"

    if is_ros1 and not bag_path.exists():
        sys.exit(f"ERROR: File not found: {bag_path}")
    if not is_ros1 and not bag_path.is_dir():
        sys.exit(f"ERROR: ROS2 bag folder not found: {bag_path}")

    # --- Sequence detection (Variants A / B) ---
    sequence: List[Path] = [bag_path]
    seq_infos: List[SequenceInfo] = []
    if is_ros1 and not args.no_sequence:
        if is_dir_of_bags:
            detected = dir_bags          # Variant C: all bags in directory
        else:
            detected = detect_bag_sequence(bag_path)  # Variant A/B: auto-detect
        if len(detected) > 1:
            seq_infos = validate_bag_sequence(detected)
            _print_sequence_summary(seq_infos)
            if args.sequence or is_dir_of_bags:
                # Variant B or C: process all bags
                sequence = detected
                print(f"  Processing all {len(sequence)} bags in sequence.\n")
            else:
                # Variant A: inform only
                print(f"  INFO: Detected {len(detected)} bags in sequence.")
                print(f"         Use --sequence to audit them all.\n")

    # --- Audit each bag ---
    all_reports: List[Tuple[str, Dict[str, TopicStats], List[PairScore]]] = []
    for bag_file in sequence:
        label = bag_file.name if len(sequence) > 1 else str(bag_file)
        if len(sequence) > 1:
            print(f"\n{'─'*72}")
            print(f"  Auditing: {bag_file.name}")
            print(f"{'─'*72}")

        cur_is_ros1 = bag_file.suffix == ".bag"
        print(f"Sampling up to {args.max_msgs} messages per topic ...")
        all_topics = _sample_bag(bag_file, args.max_msgs, cur_is_ros1)

        pc_topics = [
            s for s in all_topics.values()
            if _is_pc_type(s.msgtype) and s.header_timestamps
        ]
        imu_topics = [
            s for s in all_topics.values()
            if _is_imu_type(s.msgtype) and s.header_timestamps
        ]

        print(f"Found {len(pc_topics)} PC topic(s), {len(imu_topics)} IMU topic(s)")
        n_pairs = len(pc_topics) * len(imu_topics)
        print(f"Evaluating {n_pairs} pair(s) ...")

        pairs: List[PairScore] = []
        for pc in pc_topics:
            for imu in imu_topics:
                pairs.append(score_pair(pc, imu))

        print_report(label, all_topics, pairs, args.top, args.verbose)
        all_reports.append((str(bag_file), all_topics, pairs))

    # --- JSON export (first / only bag, or combined) ---
    if args.json:
        bag_label, topics, pairs = all_reports[0]
        _write_json(args.json, bag_label, topics, pairs,
                     seq_infos=seq_infos if len(sequence) > 1 else None)


# ============================================================================
# JSON EXPORT
# ============================================================================
def _topic_to_dict(s: TopicStats) -> Dict:
    """Serialise TopicStats to a JSON-friendly dict."""
    d: Dict = {
        "topic": s.topic,
        "msgtype": s.msgtype,
        "msgcount": s.msgcount,
        "sampled": len(s.header_timestamps),
        "deser_ok": s.deser_ok,
        "deser_fallback": s.deser_fallback,
        "deser_fail": s.deser_fail,
    }
    if s.header_timestamps:
        d["epoch_sec"] = s.epoch_sec
        d["duration_sec"] = s.duration_sec
        d["frequency_hz"] = round(s.frequency_hz, 3)
        d["dt_mean"] = s.dt_mean
        d["dt_std"] = s.dt_std
        d["monotone"] = s.monotone
    if s.bag_time_count > 0:
        d["bag_time_start"] = s.bag_time_start
        d["bag_time_end"] = s.bag_time_end
        d["bag_time_count"] = s.bag_time_count
        d["bag_header_offset_med"] = s.bag_header_offset_med
        d["bag_header_offset_std"] = s.bag_header_offset_std
    if s.dominant_frame_id:
        d["frame_id"] = s.dominant_frame_id
    if _is_imu_type(s.msgtype):
        d["acc_unit"] = s.acc_unit
        d["acc_scale"] = s.acc_scale
        d["gyro_unit"] = s.gyro_unit
        d["gyro_scale"] = s.gyro_scale
        if s.acc_magnitudes:
            d["acc_mean_raw"] = float(np.mean(s.acc_magnitudes))
            d["acc_mean_si"] = float(np.mean(s.acc_magnitudes)) * s.acc_scale
        if s.gyro_magnitudes:
            d["gyro_mean_raw"] = float(np.mean(s.gyro_magnitudes))
            d["gyro_mean_si"] = float(np.mean(s.gyro_magnitudes)) * s.gyro_scale
    if _is_pc_type(s.msgtype):
        if s.point_counts:
            d["pts_per_msg_avg"] = float(np.mean(s.point_counts))
        d["has_per_point_ts"] = s.has_per_point_ts
        d["per_point_ts_verified"] = s.per_point_ts_verified
        if s.per_point_ts_field:
            d["per_point_ts_field"] = s.per_point_ts_field
        if s.livox_timebases:
            d["livox_timebase_min"] = float(np.min(s.livox_timebases))
            d["livox_timebase_max"] = float(np.max(s.livox_timebases))
    return d


def _pair_to_dict(ps: PairScore) -> Dict:
    """Serialise PairScore to a JSON-friendly dict."""
    return {
        "pc_topic": ps.pc_topic,
        "imu_topic": ps.imu_topic,
        "total": round(ps.total, 4),
        "grade": _grade(ps.total),
        "imu_time_offset_sec": ps.imu_time_offset_sec,
        "recommended_pc_clock": ps.recommended_pc_clock,
        "scores": {k: round(v, 4) for k, v in ps.scores.items()},
        "notes": dict(ps.notes),
    }


def _write_json(
    json_path: str,
    bag_path: str,
    all_topics: Dict[str, TopicStats],
    pairs: List[PairScore],
    seq_infos: Optional[List[SequenceInfo]] = None,
) -> None:
    """Write the full audit results to a JSON file."""
    relevant = {
        t: s for t, s in all_topics.items()
        if _is_imu_type(s.msgtype) or _is_pc_type(s.msgtype)
    }
    pairs_sorted = sorted(pairs, key=lambda p: p.total, reverse=True)

    data: Dict[str, Any] = {
        "audit_version": "0.8",
        "bag": bag_path,
        "topics": [
            {"topic": s.topic, "msgtype": s.msgtype, "msgcount": s.msgcount}
            for s in sorted(all_topics.values(), key=lambda x: x.topic)
        ],
        "analysed_topics": [
            _topic_to_dict(s)
            for s in sorted(relevant.values(), key=lambda x: x.topic)
        ],
        "pairs": [_pair_to_dict(p) for p in pairs_sorted],
    }

    # Add sequence info if present
    if seq_infos and len(seq_infos) > 1:
        data["sequence"] = [
            {
                "file": si.path.name,
                "start_ns": si.start_ns,
                "end_ns": si.end_ns,
                "duration_s": round(si.duration_s, 3),
                "gap_from_prev_s": (round(si.gap_from_prev_s, 6)
                                    if si.gap_from_prev_s is not None
                                    else None),
            }
            for si in seq_infos
        ]

    # Add recommended convert command from best pair
    if pairs_sorted and pairs_sorted[0].total >= 0.30:
        best = pairs_sorted[0]
        cmd = {
            "pointcloud_topic": best.pc_topic,
            "imu_topic": best.imu_topic,
            "recommended_pc_clock": best.recommended_pc_clock,
            "imu_time_offset_sec": best.imu_time_offset_sec,
        }
        data["recommended"] = cmd

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  Audit JSON written to: {json_path}")


if __name__ == "__main__":
    main()
