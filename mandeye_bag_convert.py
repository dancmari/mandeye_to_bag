#!/usr/bin/env python3
"""
mandeye_bag_convert.py — Standalone Python converter for MandEye / HDMapping.

Converts between MandEye datasets (LAZ point clouds + CSV IMU data)
and ROS1/ROS2 bag files — without needing a ROS installation.

Supported modes:
  hdmapping-to-ros1   : MandEye folder  -> ROS1 bag
  ros1-to-hdmapping   : ROS1 bag        -> MandEye folder
  hdmapping-to-ros2   : MandEye folder  -> ROS2 bag
  ros2-to-hdmapping   : ROS2 bag folder -> MandEye folder

IMU unit handling (bag → MandEye export):
  When exporting from bag to MandEye, IMU values are converted to:
    - Accelerometer → g  (1 g ≈ 9.80665 m/s²)
    - Gyroscope     → deg/s
  Unit detection priority:
    1. --acc_unit / --gyro_unit   (explicit CLI override)
    2. --audit-json               (from mandeye_bag_audit.py JSON export)
    3. Auto-detection             (samples first 500 IMU messages from bag)

Multi-volume / split bag sequences:
  When a single .bag is given, the tool auto-detects sibling bags that
  belong to the same recording (e.g. recording_0.bag … recording_N.bag).
  Use --sequence to process them all as a continuous dataset.
  Use --no-sequence to suppress detection.
  A directory of .bag files is treated as an implicit sequence.

Truncated / corrupt bags:
  If the bag file is truncated or corrupt, the tool recovers as much data
  as possible and saves partial chunks instead of crashing.

Dependencies (pip install):
  laspy[lazrs]   – read/write LAZ/LAS files
  rosbags        – read/write ROS1 & ROS2 bags (no ROS needed)
  numpy

Usage:
  python mandeye_bag_convert.py <input> <output> <mode> [options]

Options:
  --lines <N>             Number of lidar scan lines (default: 8)
  --pointcloud_topic <t>  Topic for point clouds   (default: /livox/lidar)
  --imu_topic <t>         Topic for IMU messages    (default: /livox/imu)
  --chunk_len <sec>       Chunk length in seconds   (default: 20)
  --emulate_point_ts      Interpolate per-point timestamps from header ts
  --acc_unit <unit>       Accelerometer unit: m/s2, g, mg, mm/s2 (auto-detect)
  --gyro_unit <unit>      Gyroscope unit: rad/s, deg/s (auto-detect)
  --audit-json <file>     Read audit JSON for auto topic + unit selection
  --sequence              Process all bags in detected sequence
  --no-sequence           Suppress sequence detection
  --start_index <N>       Starting chunk index for exported files (default: 0)

Output (bag → MandEye modes):
  The output directory contains LAZ, CSV, and .sn files plus a
  convert_report.json with conversion metadata, chunk counts,
  point/IMU totals, timing, and any warnings.

Pipeline (audit → convert, bag → MandEye):
  python mandeye_bag_audit.py  recording.bag --json audit.json
  python mandeye_bag_convert.py recording.bag output ros1-to-hdmapping --audit-json audit.json

Pipeline (MandEye → bag):
  python mandeye_bag_convert.py ./my_dataset ./output.bag hdmapping-to-ros1
  python mandeye_bag_convert.py ./my_dataset ./output_ros2 hdmapping-to-ros2
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("ERROR: 'laspy' is required.  Install with:  pip install laspy[lazrs]")

from mandeye_bag_common import (
    Reader1, Reader1Error, Reader2,
    Writer1, Writer2,
    typestore,
    serialize_cdr, deserialize_cdr,
    serialize_ros1, deserialize_ros1,
    cdr_to_ros1, ros1_to_cdr,
    deserialize_ros1_safe as _deserialize_ros1_safe,
    rostime_to_sec, rostime_to_nsec, sec_to_rostime, sec_to_nsec,
    Header, Time, Vector3, Quaternion, Imu, PointCloud2, PointField,
    is_imu_type, is_pc_type, is_custom_msg,
    guess_acc_unit as _guess_acc_unit,
    guess_gyro_unit as _guess_gyro_unit,
    compute_imu_factors as _compute_imu_factors,
    guess_acc_unit_by_name as _guess_acc_unit_by_name,
    guess_gyro_unit_by_name as _guess_gyro_unit_by_name,
    _G, _ACC_UNIT_TABLE, _GYRO_UNIT_TABLE,
    extract_seq_prefix as _extract_seq_prefix,
    detect_bag_sequence,
    print_sequence_info as _print_sequence_info,
    resolve_unique_output_path,
    add_sequence_args,
    detect_dir_bags,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class PointXYZIT(NamedTuple):
    x: float
    y: float
    z: float
    intensity: float
    timestamp: float  # seconds


class ImuSample(NamedTuple):
    timestamp: float  # seconds
    gyr_x: float
    gyr_y: float
    gyr_z: float
    acc_x: float
    acc_y: float
    acc_z: float




# IMU unit functions, sequence detection, resolve_unique_output_path,
# add_sequence_args, detect_dir_bags  — all imported from mandeye_bag_common


# ---------------------------------------------------------------------------
# LAZ I/O
# ---------------------------------------------------------------------------
def load_laz(path: str) -> List[PointXYZIT]:
    """Read a LAZ/LAS file and return a list of PointXYZIT."""
    las = laspy.read(path)
    xs = np.array(las.x, dtype=np.float64)
    ys = np.array(las.y, dtype=np.float64)
    zs = np.array(las.z, dtype=np.float64)
    intensities = np.array(las.intensity, dtype=np.float32)
    # GPS time may not exist in all files
    if hasattr(las, "gps_time"):
        timestamps = np.array(las.gps_time, dtype=np.float64)
    else:
        timestamps = np.zeros(len(xs), dtype=np.float64)

    points = []
    for i in range(len(xs)):
        points.append(PointXYZIT(xs[i], ys[i], zs[i], intensities[i], timestamps[i]))
    return points


def save_laz(path: str, points: List[PointXYZIT]) -> None:
    """Write a list of PointXYZIT to a LAZ file."""
    if not points:
        return

    xs = np.array([p.x for p in points], dtype=np.float64)
    ys = np.array([p.y for p in points], dtype=np.float64)
    zs = np.array([p.z for p in points], dtype=np.float64)
    intensities = np.array([p.intensity for p in points], dtype=np.uint16)
    timestamps = np.array([p.timestamp for p in points], dtype=np.float64)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = [np.min(xs), np.min(ys), np.min(zs)]
    header.scales = [0.0001, 0.0001, 0.0001]

    las = laspy.LasData(header)
    las.x = xs
    las.y = ys
    las.z = zs
    las.intensity = intensities
    las.gps_time = timestamps * 1e-9  # store as seconds (matching C++ saveLaz)

    las.write(path)
    print(f"  Saved {len(points)} points -> {path}")


# ---------------------------------------------------------------------------
# IMU CSV I/O
# ---------------------------------------------------------------------------
def load_imu_csv(path: str, imu_to_use: int = 0) -> List[ImuSample]:
    """Load IMU data from a CSV file (supports both legacy and header formats)."""
    samples: List[ImuSample] = []

    with open(path, "r") as f:
        first_line = f.readline().strip()

    # Detect if the file has a header row
    is_header = "timestamp" in first_line.lower() or "gyrox" in first_line.lower()

    if is_header:
        with open(path, "r") as f:
            # Try multiple delimiters
            content = f.read()

        for delim in [",", " ", "\t"]:
            try:
                reader = csv.DictReader(io.StringIO(content), delimiter=delim)
                cols = reader.fieldnames
                if cols and "timestamp" in cols:
                    break
            except Exception:
                continue
        else:
            print(f"  WARNING: Could not parse header CSV: {path}")
            return samples

        for row in reader:
            imu_id = int(row.get("imuId", -1))
            if imu_id >= 0 and imu_id != imu_to_use:
                continue
            ts = float(row["timestamp"]) / 1e9
            gx = float(row["gyroX"])
            gy = float(row["gyroY"])
            gz = float(row["gyroZ"])
            ax = float(row["accX"])
            ay = float(row["accY"])
            az = float(row["accZ"])
            if ts > 0:
                samples.append(ImuSample(ts, gx, gy, gz, ax, ay, az))
    else:
        # Legacy whitespace-separated format:
        # timestamp gyroX gyroY gyroZ accX accY accZ [imuId] [timestampUnix]
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                try:
                    vals = [float(x) for x in parts[:7]]
                except ValueError:
                    continue
                imu_id = int(parts[7]) if len(parts) > 7 else 0
                if vals[0] > 0 and imu_id == imu_to_use:
                    ts = vals[0] / 1e9
                    samples.append(
                        ImuSample(ts, vals[1], vals[2], vals[3], vals[4], vals[5], vals[6])
                    )

    return samples


def save_imu_csv(path: str, samples: List[str], delim: str = ",", imu_id: int = 0) -> None:
    """Write IMU lines to a CSV file with the given delimiter, appending imuId."""
    with open(path, "w") as f:
        for line in samples:
            f.write(line + f"{delim}{imu_id}" + "\n")
    print(f"  Saved {len(samples)} IMU samples -> {path}")


# ROS time utilities imported from mandeye_bag_common:
#   sec_to_rostime, rostime_to_sec, rostime_to_nsec, sec_to_nsec


# ---------------------------------------------------------------------------
# Build ROS messages
# ---------------------------------------------------------------------------
def make_imu_msg(sample: ImuSample, frame_id: str = "livox") -> Imu:
    stamp = sec_to_rostime(sample.timestamp)
    return Imu(
        header=Header(stamp=stamp, frame_id=frame_id),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=np.zeros(9, dtype=np.float64),
        angular_velocity=Vector3(x=sample.gyr_x, y=sample.gyr_y, z=sample.gyr_z),
        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
        linear_acceleration=Vector3(x=sample.acc_x, y=sample.acc_y, z=sample.acc_z),
        linear_acceleration_covariance=np.zeros(9, dtype=np.float64),
    )


def make_pointcloud2_msg(
    points: List[PointXYZIT], stamp_sec: float, frame_id: str = "livox"
) -> PointCloud2:
    """Create a sensor_msgs/PointCloud2 from a list of points."""
    n = len(points)
    # 4 fields: x, y, z, intensity — each float32 (4 bytes)
    point_step = 16
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
        PointField(name="intensity", offset=12, datatype=7, count=1),
    ]

    data = bytearray(n * point_step)
    for i, p in enumerate(points):
        struct.pack_into("<ffff", data, i * point_step, p.x, p.y, p.z, p.intensity)

    stamp = sec_to_rostime(stamp_sec)
    return PointCloud2(
        header=Header(stamp=stamp, frame_id=frame_id),
        height=1,
        width=n,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n,
        data=np.frombuffer(bytes(data), dtype=np.uint8),
        is_dense=True,
    )


# ---------------------------------------------------------------------------
# Parse Livox CustomMsg back to points
# ---------------------------------------------------------------------------
def parse_custom_msg(msg) -> List[PointXYZIT]:
    """Extract PointXYZIT list from a livox_ros_driver(2)/CustomMsg."""
    points = []
    timebase = msg.timebase  # nanoseconds
    for p in msg.points:
        ts_ns = timebase + p.offset_time
        points.append(PointXYZIT(p.x, p.y, p.z, float(p.reflectivity), ts_ns))
    return points


# ---------------------------------------------------------------------------
# Parse PointCloud2 message back to points
# ---------------------------------------------------------------------------
def parse_pointcloud2(msg, emulate_ts: bool = False, frame_rate: float = 0.0) -> List[PointXYZIT]:
    """Extract PointXYZIT list from a PointCloud2 message."""
    points = []
    # Find field offsets
    field_map = {f.name: f for f in msg.fields}
    x_off = field_map["x"].offset
    y_off = field_map["y"].offset
    z_off = field_map["z"].offset
    i_off = field_map.get("intensity", field_map.get("i", None))
    i_offset = i_off.offset if i_off else -1

    data = bytes(msg.data)
    ps = msg.point_step
    n = msg.width * msg.height
    header_ts = rostime_to_sec(msg.header.stamp)

    for idx in range(n):
        base = idx * ps
        x = struct.unpack_from("<f", data, base + x_off)[0]
        y = struct.unpack_from("<f", data, base + y_off)[0]
        z = struct.unpack_from("<f", data, base + z_off)[0]
        intensity = struct.unpack_from("<f", data, base + i_offset)[0] if i_offset >= 0 else 0.0

        if emulate_ts and frame_rate > 0:
            ts = (header_ts + (idx * frame_rate) / n) * 1e9
        else:
            ts = rostime_to_nsec(msg.header.stamp)

        points.append(PointXYZIT(x, y, z, intensity, ts))

    return points


# ---------------------------------------------------------------------------
# Mode: hdmapping -> ROS1 bag
# ---------------------------------------------------------------------------
def hdmapping_to_ros1(
    input_dir: str,
    output_bag: str,
    num_lines: int = 8,
    imu_topic: str = "/livox/imu",
    pc_topic: str = "/livox/lidar",
) -> None:
    print(f"Converting MandEye -> ROS1 bag")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_bag}")

    input_path = Path(input_dir)
    files_imu = sorted(input_path.glob("*.csv"))
    files_laz = sorted(input_path.glob("*.laz"))

    print(f"  Found {len(files_imu)} IMU files, {len(files_laz)} LAZ files")

    with Writer1(output_bag) as writer:
        imu_conn = writer.add_connection(imu_topic, Imu.__msgtype__)
        pc_conn = writer.add_connection(pc_topic, PointCloud2.__msgtype__)

        # Write IMU data
        total_imu = 0
        for imu_fn in files_imu:
            print(f"  Loading IMU: {imu_fn.name}")
            samples = load_imu_csv(str(imu_fn))
            for s in samples:
                if s.timestamp == 0:
                    continue
                msg = make_imu_msg(s)
                raw = serialize_ros1(msg, msg.__class__.__msgtype__)
                writer.write(imu_conn, sec_to_nsec(s.timestamp), raw)
                total_imu += 1

        print(f"  Wrote {total_imu} IMU messages")

        # Write point cloud data
        NUM_POINT = 19968
        total_pc = 0
        buffer: List[PointXYZIT] = []

        for laz_fn in files_laz:
            print(f"  Loading LAZ: {laz_fn.name}")
            points = load_laz(str(laz_fn))
            for p in points:
                if p.timestamp == 0:
                    continue
                buffer.append(p)
                if len(buffer) >= NUM_POINT:
                    stamp = buffer[0].timestamp
                    pc_msg = make_pointcloud2_msg(buffer, stamp)
                    raw = serialize_ros1(pc_msg, pc_msg.__class__.__msgtype__)
                    writer.write(pc_conn, sec_to_nsec(stamp), raw)
                    buffer.clear()
                    total_pc += 1

        # Flush remaining
        if buffer:
            stamp = buffer[0].timestamp
            pc_msg = make_pointcloud2_msg(buffer, stamp)
            raw = serialize_ros1(pc_msg, pc_msg.__class__.__msgtype__)
            writer.write(pc_conn, sec_to_nsec(stamp), raw)
            total_pc += 1

        print(f"  Wrote {total_pc} PointCloud2 messages")

    print("Done!")


# ---------------------------------------------------------------------------
# Mode: hdmapping -> ROS2 bag
# ---------------------------------------------------------------------------
def hdmapping_to_ros2(
    input_dir: str,
    output_bag: str,
    num_lines: int = 8,
    imu_topic: str = "/livox/imu",
    pc_topic: str = "/livox/pointcloud",
) -> None:
    print(f"Converting MandEye -> ROS2 bag")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_bag}")

    input_path = Path(input_dir)
    files_imu = sorted(input_path.glob("*.csv"))
    # For ROS2, filter only files with "lidar" in the name (matching C++ code)
    files_laz_all = sorted(input_path.glob("*.laz"))
    files_laz = [f for f in files_laz_all if "lidar" in f.name.lower()]
    if not files_laz:
        # Fallback: use all LAZ files
        files_laz = files_laz_all

    print(f"  Found {len(files_imu)} IMU files, {len(files_laz)} LAZ files")

    with Writer2(output_bag) as writer:
        imu_conn = writer.add_connection(
            imu_topic,
            Imu.__msgtype__,
            typestore=typestore,
        )
        pc_conn = writer.add_connection(
            pc_topic,
            PointCloud2.__msgtype__,
            typestore=typestore,
        )

        # Write IMU data
        total_imu = 0
        for imu_fn in files_imu:
            print(f"  Loading IMU: {imu_fn.name}")
            samples = load_imu_csv(str(imu_fn))
            for s in samples:
                if s.timestamp == 0:
                    continue
                msg = make_imu_msg(s)
                raw = serialize_cdr(msg, msg.__class__.__msgtype__)
                writer.write(imu_conn, sec_to_nsec(s.timestamp), raw)
                total_imu += 1

        print(f"  Wrote {total_imu} IMU messages")

        # Write point cloud data
        NUM_POINT = 19968
        total_pc = 0
        buffer: List[PointXYZIT] = []
        last_ts: Optional[float] = None

        for laz_fn in files_laz:
            print(f"  Loading LAZ: {laz_fn.name}")
            points = load_laz(str(laz_fn))
            for p in points:
                if p.timestamp == 0:
                    continue
                if last_ts is None:
                    last_ts = p.timestamp
                buffer.append(p)
                if len(buffer) > NUM_POINT and last_ts is not None and last_ts > 0:
                    pc_msg = make_pointcloud2_msg(buffer, last_ts)
                    raw = serialize_cdr(pc_msg, pc_msg.__class__.__msgtype__)
                    writer.write(pc_conn, sec_to_nsec(last_ts), raw)
                    buffer.clear()
                    last_ts = None
                    total_pc += 1

        if buffer and last_ts is not None:
            pc_msg = make_pointcloud2_msg(buffer, last_ts)
            raw = serialize_cdr(pc_msg, pc_msg.__class__.__msgtype__)
            writer.write(pc_conn, sec_to_nsec(last_ts), raw)
            total_pc += 1

        print(f"  Wrote {total_pc} PointCloud2 messages")

    print("Done!")


# ---------------------------------------------------------------------------
# Shared: bag -> hdmapping (unified ROS1 / ROS2 implementation)
# ---------------------------------------------------------------------------
def _bag_to_hdmapping(
    input_path: str,
    output_dir: str,
    is_ros1: bool,
    pc_topic: str = "/livox/lidar",
    imu_topic: str = "/livox/imu",
    chunk_len: float = 20.0,
    emulate_point_ts: bool = False,
    csv_delim: str = ",",
    imu_id: int = 0,
    serial: str = "XXXXXXXXXX",
    acc_unit: str = "",
    gyro_unit: str = "",
    bag_files_override: Optional[List[Path]] = None,
    start_index: int = 0,
) -> Dict[str, Any]:
    """Convert ROS bag(s) to MandEye folder (shared ROS1/ROS2 implementation).

    Supports multi-volume sequences (via *bag_files_override*), Livox
    ``CustomMsg`` detection (both ROS1 and ROS2), and automatic IMU unit
    detection / conversion.  Returns a report dict.
    """
    t0 = time.monotonic()
    mode_label = "ROS1" if is_ros1 else "ROS2"
    mode_name = f"{mode_label.lower()}-to-hdmapping"
    ReaderCls = Reader1 if is_ros1 else Reader2
    deser_fn = _deserialize_ros1_safe if is_ros1 else deserialize_cdr

    print(f"Converting {mode_label} bag -> MandEye")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_dir}")
    print(f"  PC topic:  {pc_topic}")
    print(f"  IMU topic: {imu_topic}")
    print(f"  Chunk len: {chunk_len}s")
    if start_index:
        print(f"  Start index: {start_index}")

    os.makedirs(output_dir, exist_ok=True)

    # Resolve bag file list
    if bag_files_override is not None:
        bag_files = bag_files_override
    elif is_ros1:
        p = Path(input_path)
        if p.suffix == ".bag":
            bag_files = [p]
        else:
            bag_files = sorted(p.glob("*.bag"))
    else:
        bag_files = [Path(input_path)]

    buffer_pc: List[PointXYZIT] = []
    buffer_imu: List[str] = []
    last_save_ts = 0.0
    count = start_index
    last_imu_ts = -1.0
    lidar_frame_rate = 0.0
    total_pts = 0
    total_imu = 0
    warnings: List[str] = []
    acc_factor = 1.0
    gyro_factor = 1.0
    units_detected = False
    frame_rate_detected = False

    for bag_path in bag_files:
        print(f"  Processing bag: {bag_path.name}")

        # --- Detect message types ---
        has_custom_msg = False
        with ReaderCls(bag_path) as reader:
            print("  Topics in bag:")
            for c in reader.connections:
                cnt = getattr(c, "msgcount", None)
                cnt_str = f"({cnt} msgs)" if cnt is not None else ""
                print(f"    {c.topic}  [{c.msgtype}]  {cnt_str}")
                if c.topic == pc_topic and "CustomMsg" in c.msgtype:
                    has_custom_msg = True

        # --- IMU unit detection (once, from the first bag) ---
        if not units_detected:
            if acc_unit and gyro_unit:
                _au, acc_s2mps2 = _guess_acc_unit_by_name(acc_unit)
                _gu, gyro_s2radps = _guess_gyro_unit_by_name(gyro_unit)
                acc_factor, gyro_factor = _compute_imu_factors(
                    _au, acc_s2mps2, _gu, gyro_s2radps,
                )
                print(f"  IMU units (from audit/CLI):")
                print(f"    Accel:  {acc_unit}  (x{acc_factor:.6f} -> g)")
                print(f"    Gyro:   {gyro_unit}  (x{gyro_factor:.6f} -> deg/s)")
            else:
                acc_mags: List[float] = []
                gyro_mags: List[float] = []
                with ReaderCls(bag_path) as reader:
                  try:
                    for conn, timestamp, rawdata in reader.messages():
                        if conn.topic == imu_topic and "Imu" in conn.msgtype:
                            msg = deser_fn(rawdata, conn.msgtype)
                            ax = msg.linear_acceleration.x
                            ay = msg.linear_acceleration.y
                            az = msg.linear_acceleration.z
                            acc_mags.append(math.sqrt(ax*ax + ay*ay + az*az))
                            gx = msg.angular_velocity.x
                            gy = msg.angular_velocity.y
                            gz = msg.angular_velocity.z
                            gyro_mags.append(math.sqrt(gx*gx + gy*gy + gz*gz))
                            if len(acc_mags) >= 500:
                                break
                  except Exception:
                    pass

                acc_unit, acc_s2mps2 = _guess_acc_unit(np.array(acc_mags))
                gyro_unit, gyro_s2radps = _guess_gyro_unit(np.array(gyro_mags))
                acc_factor, gyro_factor = _compute_imu_factors(
                    acc_unit, acc_s2mps2, gyro_unit, gyro_s2radps,
                )
                print(f"  IMU units detected (auto):")
                print(f"    Accel:  {acc_unit}  (x{acc_factor:.6f} -> g)")
                print(f"    Gyro:   {gyro_unit}  (x{gyro_factor:.6f} -> deg/s)")
            units_detected = True

        # --- Frame rate estimation (once, if --emulate_point_ts) ---
        if emulate_point_ts and not frame_rate_detected:
            print("  Emulating point timestamps, collecting framerate...")
            header_diffs: List[float] = []
            last_header_ts = 0.0
            fr_start_ts = 0.0
            with ReaderCls(bag_path) as reader:
              try:
                for conn, timestamp, rawdata in reader.messages():
                    if conn.topic == pc_topic:
                        msg = deser_fn(rawdata, conn.msgtype)
                        ts = rostime_to_sec(msg.header.stamp)
                        if last_header_ts != 0.0:
                            if fr_start_ts == 0.0:
                                fr_start_ts = last_header_ts
                            diff = ts - last_header_ts
                            if diff > 0:
                                header_diffs.append(diff)
                            if ts - fr_start_ts > 20.0:
                                break
                        last_header_ts = ts
              except Exception as exc:
                print(f"  WARNING: Bag read error (truncated/corrupt?): {exc}",
                      file=sys.stderr)
                print("           Continuing with data read so far ...",
                      file=sys.stderr)

            if header_diffs:
                lidar_frame_rate = sum(header_diffs) / len(header_diffs)
                print(f"  Estimated frame rate: {lidar_frame_rate:.6f}s")
            frame_rate_detected = True

        # --- Main data extraction pass ---
        with ReaderCls(bag_path) as reader:
          try:
            for conn, timestamp, rawdata in reader.messages():
                msg_time_sec = timestamp / 1e9

                if conn.topic == imu_topic and "Imu" in conn.msgtype:
                    msg = deser_fn(rawdata, conn.msgtype)
                    ts_ns = rostime_to_nsec(msg.header.stamp)
                    d = csv_delim
                    gx = msg.angular_velocity.x * gyro_factor
                    gy = msg.angular_velocity.y * gyro_factor
                    gz = msg.angular_velocity.z * gyro_factor
                    ax = msg.linear_acceleration.x * acc_factor
                    ay = msg.linear_acceleration.y * acc_factor
                    az = msg.linear_acceleration.z * acc_factor
                    line = (
                        f"{ts_ns}{d}"
                        f"{gx}{d}{gy}{d}{gz}{d}"
                        f"{ax}{d}{ay}{d}{az}"
                    )
                    buffer_imu.append(line)
                    last_imu_ts = rostime_to_sec(msg.header.stamp)
                    total_imu += 1
                    if last_save_ts == 0.0:
                        last_save_ts = last_imu_ts

                if conn.topic == pc_topic and last_imu_ts > 0:
                    msg = deser_fn(rawdata, conn.msgtype)
                    ts = rostime_to_sec(msg.header.stamp)
                    if abs(ts - last_imu_ts) < 0.05 * chunk_len:
                        if has_custom_msg:
                            pts = parse_custom_msg(msg)
                        else:
                            pts = parse_pointcloud2(
                                msg, emulate_point_ts, lidar_frame_rate,
                            )
                        buffer_pc.extend(pts)
                    else:
                        warnings.append(
                            f"Skipped PC at {ts:.3f}s "
                            f"(IMU drift: {abs(ts - last_imu_ts):.3f}s)")
                        print(
                            f"  Skipping pointcloud at {ts:.3f}s "
                            f"(IMU drift: {abs(ts - last_imu_ts):.3f}s)")

                if msg_time_sec - last_save_ts > chunk_len and last_save_ts > 0:
                    _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                                csv_delim=csv_delim, imu_id=imu_id,
                                serial=serial)
                    total_pts += len(buffer_pc)
                    buffer_pc.clear()
                    buffer_imu.clear()
                    last_save_ts = msg_time_sec
                    count += 1

          except Exception as exc:
            w = f"Bag read error on {bag_path.name}: {exc}"
            warnings.append(w)
            print(f"  WARNING: Bag read error (truncated/corrupt?): {exc}",
                  file=sys.stderr)
            print("           Saving data read so far ...", file=sys.stderr)

    if buffer_pc or buffer_imu:
        _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                    csv_delim=csv_delim, imu_id=imu_id, serial=serial)
        total_pts += len(buffer_pc)

    elapsed = time.monotonic() - t0
    n_chunks = count - start_index + (1 if buffer_pc or buffer_imu else 0)
    report: Dict[str, Any] = {
        "mode": mode_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_path,
        "output": output_dir,
        "bags_processed": [str(b) for b in bag_files],
        "pc_topic": pc_topic,
        "imu_topic": imu_topic,
        "chunk_len_s": chunk_len,
        "start_index": start_index,
        "chunks_written": n_chunks,
        "chunk_range": [start_index, count],
        "total_points": total_pts,
        "total_imu_messages": total_imu,
        "acc_unit": acc_unit,
        "gyro_unit": gyro_unit,
        "imu_id": imu_id,
        "serial": serial,
        "elapsed_s": round(elapsed, 2),
        "warnings": warnings,
    }
    print("Done!")
    return report


# ---------------------------------------------------------------------------
# Mode: ROS1 bag -> hdmapping
# ---------------------------------------------------------------------------
def ros1_to_hdmapping(
    input_bag: str,
    output_dir: str,
    pc_topic: str = "/livox/lidar",
    imu_topic: str = "/livox/imu",
    chunk_len: float = 20.0,
    emulate_point_ts: bool = False,
    csv_delim: str = ",",
    imu_id: int = 0,
    serial: str = "XXXXXXXXXX",
    acc_unit: str = "",
    gyro_unit: str = "",
    bag_files_override: Optional[List[Path]] = None,
    start_index: int = 0,
) -> Dict[str, Any]:
    """Convert ROS1 bag(s) to MandEye folder.  Returns a report dict."""
    return _bag_to_hdmapping(
        input_bag, output_dir, is_ros1=True,
        pc_topic=pc_topic, imu_topic=imu_topic,
        chunk_len=chunk_len, emulate_point_ts=emulate_point_ts,
        csv_delim=csv_delim, imu_id=imu_id, serial=serial,
        acc_unit=acc_unit, gyro_unit=gyro_unit,
        bag_files_override=bag_files_override, start_index=start_index,
    )


# ---------------------------------------------------------------------------
# Mode: ROS2 bag -> hdmapping
# ---------------------------------------------------------------------------
def ros2_to_hdmapping(
    input_bag: str,
    output_dir: str,
    pc_topic: str = "/livox/lidar",
    imu_topic: str = "/livox/imu",
    chunk_len: float = 20.0,
    emulate_point_ts: bool = False,
    csv_delim: str = ",",
    imu_id: int = 0,
    serial: str = "XXXXXXXXXX",
    acc_unit: str = "",
    gyro_unit: str = "",
    bag_files_override: Optional[List[Path]] = None,
    start_index: int = 0,
) -> Dict[str, Any]:
    """Convert ROS2 bag to MandEye folder.  Returns a report dict."""
    return _bag_to_hdmapping(
        input_bag, output_dir, is_ros1=False,
        pc_topic=pc_topic, imu_topic=imu_topic,
        chunk_len=chunk_len, emulate_point_ts=emulate_point_ts,
        csv_delim=csv_delim, imu_id=imu_id, serial=serial,
        acc_unit=acc_unit, gyro_unit=gyro_unit,
        bag_files_override=bag_files_override, start_index=start_index,
    )


# resolve_unique_output_path imported from mandeye_bag_common


# ---------------------------------------------------------------------------
# Save a chunk (pointcloud + IMU) to disk
# ---------------------------------------------------------------------------
def _save_chunk(
    output_dir: str,
    count: int,
    buffer_pc: List[PointXYZIT],
    buffer_imu: List[str],
    csv_delim: str = ",",
    imu_id: int = 0,
    serial: str = "XXXXXXXXXX",
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pc_path = os.path.join(output_dir, f"pointcloud_{count:04d}.laz")
    imu_path = os.path.join(output_dir, f"imu_{count:04d}.csv")
    sn_path = os.path.join(output_dir, f"lidar{count:04d}.sn")
    print(f"  Saving chunk {count} ({len(buffer_pc)} pts, {len(buffer_imu)} IMU)")
    save_laz(pc_path, buffer_pc)
    save_imu_csv(imu_path, buffer_imu, delim=csv_delim, imu_id=imu_id)
    # Write serial number file
    with open(sn_path, "w") as f:
        f.write(f"{imu_id} {serial}\n")
    print(f"  Saved serial info -> {sn_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _apply_audit_json(args) -> None:
    """Read audit JSON and override topics + IMU units from best pair."""
    json_path = args.audit_json
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"WARNING: Could not read audit JSON '{json_path}': {e}")
        return

    rec = data.get("recommended")
    if not rec:
        print(f"WARNING: Audit JSON has no 'recommended' entry (all pairs scored too low?)")
        return

    pc_topic = rec.get("pointcloud_topic", args.pointcloud_topic)
    imu_topic = rec.get("imu_topic", args.imu_topic)
    clock = rec.get("recommended_pc_clock", "")
    offset_ms = rec.get("imu_time_offset_sec", 0) * 1000

    # Override only if user didn't explicitly set them on the command line
    # (argparse default values indicate user didn't set them)
    if args.pointcloud_topic == "/livox/lidar":
        args.pointcloud_topic = pc_topic
    if args.imu_topic == "/livox/imu":
        args.imu_topic = imu_topic

    # Extract IMU unit info from analysed_topics (if not already set via CLI)
    if not args.acc_unit or not args.gyro_unit:
        for topic_info in data.get("analysed_topics", []):
            if topic_info.get("topic") == imu_topic:
                if not args.acc_unit and "acc_unit" in topic_info:
                    args.acc_unit = topic_info["acc_unit"]
                if not args.gyro_unit and "gyro_unit" in topic_info:
                    args.gyro_unit = topic_info["gyro_unit"]
                break

    # Print best pair info
    pairs = data.get("pairs", [])
    best = pairs[0] if pairs else {}
    grade = best.get("grade", "?")
    total = best.get("total", 0)

    print(f"  Audit JSON loaded: {json_path}")
    print(f"    Bag:    {data.get('bag', '?')}")
    print(f"    Best:   {pc_topic} + {imu_topic}  [{grade} {total:.0%}]")
    print(f"    Clock:  {clock}  (offset ≈ {offset_ms:.1f} ms)")
    if args.acc_unit:
        print(f"    Acc unit:  {args.acc_unit}")
    if args.gyro_unit:
        print(f"    Gyro unit: {args.gyro_unit}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert between MandEye datasets and ROS bag files (standalone, no ROS needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  hdmapping-to-ros1   MandEye folder  -> ROS1 .bag
  ros1-to-hdmapping   ROS1 .bag       -> MandEye folder
  hdmapping-to-ros2   MandEye folder  -> ROS2 bag folder
  ros2-to-hdmapping   ROS2 bag folder -> MandEye folder

IMU unit conversion (bag -> MandEye):
  Output units:  Acc -> g,  Gyro -> deg/s
  Detection priority:
    1. --acc_unit / --gyro_unit   (explicit CLI override)
    2. --audit-json               (from mandeye_bag_audit.py --json)
    3. Auto-detection             (samples first 500 IMU messages)

Examples:
  # MandEye → ROS1 bag:
  python mandeye_bag_convert.py ./my_dataset ./output.bag hdmapping-to-ros1
  python mandeye_bag_convert.py ./my_dataset ./output.bag hdmapping-to-ros1 --imu_topic /imu --pointcloud_topic /points

  # MandEye → ROS2 bag:
  python mandeye_bag_convert.py ./my_dataset ./output_ros2 hdmapping-to-ros2
  python mandeye_bag_convert.py ./my_dataset ./output_ros2 hdmapping-to-ros2 --imu_topic /imu --pointcloud_topic /points

  # ROS1 bag → MandEye:
  python mandeye_bag_convert.py ./output.bag ./extracted   ros1-to-hdmapping

  # ROS2 bag → MandEye:
  python mandeye_bag_convert.py ./output_ros2 ./extracted  ros2-to-hdmapping

  # With audit JSON (auto topics + units):
  python mandeye_bag_convert.py rec.bag out ros1-to-hdmapping --audit-json audit.json

  # Explicit unit override:
  python mandeye_bag_convert.py rec.bag out ros1-to-hdmapping --acc_unit g --gyro_unit deg/s

  # Auto-indexed output: export_git, export_git_000, export_git_001, …
  python mandeye_bag_convert.py deg-vis-1.bag export_git ros1-to-hdmapping

  # Multi-volume sequence (split bags):
  python mandeye_bag_convert.py recording_0.bag out ros1-to-hdmapping --sequence
  python mandeye_bag_convert.py ./bag_dir/ out ros1-to-hdmapping

  # Start chunk index at 5 (appending to existing dataset):
  python mandeye_bag_convert.py next.bag out ros1-to-hdmapping --start_index 5

  # Round-trip: MandEye → ROS1 → MandEye:
  python mandeye_bag_convert.py ./dataset ./recording.bag hdmapping-to-ros1
  python mandeye_bag_convert.py ./recording.bag ./extracted ros1-to-hdmapping

  # Round-trip: MandEye → ROS2 → MandEye:
  python mandeye_bag_convert.py ./dataset ./recording_ros2 hdmapping-to-ros2
  python mandeye_bag_convert.py ./recording_ros2 ./extracted ros2-to-hdmapping

Output report:
  For bag-to-MandEye modes, a convert_report.json is written to the
  output directory containing: mode, timestamp, input/output paths,
  topics, chunk_len, start_index, chunks_written, chunk_range,
  total_points, total_imu_messages, IMU units, elapsed time, warnings.
""",
    )
    parser.add_argument("input", help="Input path (directory or bag file)")
    parser.add_argument("output", help="Output base path (directory or bag file)")
    parser.add_argument(
        "mode",
        choices=[
            "hdmapping-to-ros1",
            "ros1-to-hdmapping",
            "hdmapping-to-ros2",
            "ros2-to-hdmapping",
        ],
        help="Conversion mode",
    )
    parser.add_argument("--lines", type=int, default=8, help="Number of scan lines (default: 8)")
    parser.add_argument(
        "--pointcloud_topic", default="/livox/lidar", help="Point cloud topic (default: /livox/lidar)"
    )
    parser.add_argument("--imu_topic", default="/livox/imu", help="IMU topic (default: /livox/imu)")
    parser.add_argument(
        "--chunk_len", type=float, default=20.0, help="Chunk length in seconds (default: 20)"
    )
    parser.add_argument(
        "--emulate_point_ts",
        action="store_true",
        help="Interpolate per-point timestamps from header timestamp",
    )
    parser.add_argument(
        "--list_topics",
        action="store_true",
        help="Only list topics in the bag, don't convert",
    )
    parser.add_argument(
        "--imu_id", type=int, default=0,
        help="IMU ID written to .sn files and appended to CSV lines (default: 0)",
    )
    parser.add_argument(
        "--serial", default="XXXXXXXXXX",
        help='Lidar serial number written to .sn files (default: "XXXXXXXXXX")',
    )
    parser.add_argument(
        "--csv_delim", default=" ",
        help=r'CSV delimiter for output IMU files (default: " "). Use "\t" for tab.',
    )
    parser.add_argument(
        "--audit-json", metavar="FILE",
        help="Read audit JSON (from mandeye_bag_audit.py --json) to auto-set "
             "--pointcloud_topic, --imu_topic, IMU units, and print clock info",
    )
    parser.add_argument(
        "--acc_unit", default="",
        help="Override accelerometer unit (m/s2, g, mg, mm/s2). "
             "If omitted, auto-detected from bag data or audit JSON.",
    )
    parser.add_argument(
        "--gyro_unit", default="",
        help="Override gyroscope unit (rad/s, deg/s). "
             "If omitted, auto-detected from bag data or audit JSON.",
    )
    add_sequence_args(parser)
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Starting chunk index for exported files (default: 0). "
             "Useful when appending to an existing dataset.",
    )

    args = parser.parse_args()

    # Apply audit JSON overrides (before any other processing)
    if args.audit_json:
        _apply_audit_json(args)

    # --list_topics mode: just print bag contents and exit
    if args.list_topics:
        p = Path(args.input)
        if p.suffix == ".bag":
            print(f"Topics in ROS1 bag: {args.input}")
            with Reader1(p) as reader:
                for c in reader.connections:
                    print(f"  {c.topic}  [{c.msgtype}]  ({c.msgcount} msgs)")
        else:
            print(f"Topics in ROS2 bag: {args.input}")
            with Reader2(args.input) as reader:
                for c in reader.connections:
                    print(f"  {c.topic}  [{c.msgtype}]")
        sys.exit(0)

    # Resolve a unique output path so we never overwrite existing data
    output = resolve_unique_output_path(args.output)
    if output != args.output:
        print(f"Output '{args.output}' exists, using '{output}' instead.")

    # Decode escape sequences in delimiter (e.g. "\t" -> tab)
    csv_delim = args.csv_delim.encode().decode("unicode_escape")

    # --- Sequence detection (multi-volume split bags) ---
    input_path = Path(args.input)
    bag_sequence: Optional[List[Path]] = None

    # Variant C: directory containing .bag files (for ros1 modes)
    dir_bags = detect_dir_bags(input_path)
    is_dir_of_bags = bool(dir_bags)

    if args.mode in ("ros1-to-hdmapping",) and not args.no_sequence:
        if input_path.suffix == ".bag":
            detected = detect_bag_sequence(input_path)
            if len(detected) > 1:
                print(f"\n  Sequence detected: {len(detected)} bags")
                _print_sequence_info(detected)
                if args.sequence:
                    bag_sequence = detected
                    print(f"  Processing all {len(detected)} bags in sequence.\n")
                else:
                    print(f"  INFO: Use --sequence to process them all.\n")
        elif is_dir_of_bags:
            bag_sequence = dir_bags
            print(f"\n  Directory mode: {len(dir_bags)} .bag files")
            _print_sequence_info(dir_bags)
            print()

    report: Optional[Dict[str, Any]] = None

    if args.mode == "hdmapping-to-ros1":
        hdmapping_to_ros1(
            args.input,
            output,
            num_lines=args.lines,
            imu_topic=args.imu_topic,
            pc_topic=args.pointcloud_topic,
        )
    elif args.mode == "ros1-to-hdmapping":
        report = ros1_to_hdmapping(
            args.input,
            output,
            pc_topic=args.pointcloud_topic,
            imu_topic=args.imu_topic,
            chunk_len=args.chunk_len,
            emulate_point_ts=args.emulate_point_ts,
            csv_delim=csv_delim,
            imu_id=args.imu_id,
            serial=args.serial,
            acc_unit=args.acc_unit,
            gyro_unit=args.gyro_unit,
            bag_files_override=bag_sequence,
            start_index=args.start_index,
        )
    elif args.mode == "hdmapping-to-ros2":
        hdmapping_to_ros2(
            args.input,
            output,
            num_lines=args.lines,
            imu_topic=args.imu_topic,
            pc_topic=args.pointcloud_topic,
        )
    elif args.mode == "ros2-to-hdmapping":
        report = ros2_to_hdmapping(
            args.input,
            output,
            pc_topic=args.pointcloud_topic,
            imu_topic=args.imu_topic,
            chunk_len=args.chunk_len,
            emulate_point_ts=args.emulate_point_ts,
            csv_delim=csv_delim,
            imu_id=args.imu_id,
            serial=args.serial,
            acc_unit=args.acc_unit,
            gyro_unit=args.gyro_unit,
            bag_files_override=bag_sequence,
            start_index=args.start_index,
        )

    # Write conversion report to output directory
    if report is not None:
        report_path = os.path.join(output, "convert_report.json")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"  Report written to: {report_path}")
        except Exception as exc:
            print(f"  WARNING: Could not write report: {exc}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
