#!/usr/bin/env python3
"""
Standalone Python equivalent of the mandeye_to_bag C++ tools.

Converts between MandEye datasets (LAZ point clouds + CSV IMU data)
and ROS1/ROS2 bag files — without needing a ROS installation.

Supported modes:
  hdmapping-to-ros1   : MandEye folder  -> ROS1 bag
  ros1-to-hdmapping   : ROS1 bag        -> MandEye folder
  hdmapping-to-ros2   : MandEye folder  -> ROS2 bag
  ros2-to-hdmapping   : ROS2 bag folder -> MandEye folder

Dependencies (pip install):
  laspy[lazrs]   – read/write LAZ/LAS files
  rosbags        – read/write ROS1 & ROS2 bags (no ROS needed)
  numpy

Usage:
  python mandeye_convert.py <input> <output> <mode> [options]

Options:
  --lines <N>             Number of lidar scan lines (default: 8)
  --pointcloud_topic <t>  Topic for point clouds   (default: /livox/lidar)
  --imu_topic <t>         Topic for IMU messages    (default: /livox/imu)
  --chunk_len <sec>       Chunk length in seconds for rosbag->mandeye (default: 20)
  --emulate_point_ts      Interpolate per-point timestamps from header ts
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import struct
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    sys.exit("ERROR: 'laspy' is required.  Install with:  pip install laspy[lazrs]")

try:
    from rosbags.rosbag1 import Reader as Reader1, Writer as Writer1
    from rosbags.rosbag2 import Reader as Reader2, Writer as Writer2
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit(
        "ERROR: 'rosbags' is required.  Install with:  pip install rosbags"
    )

# ---------------------------------------------------------------------------
# ROS type system setup
# ---------------------------------------------------------------------------
typestore = get_typestore(Stores.ROS2_HUMBLE)

# Bind serialization functions from the typestore
serialize_cdr = typestore.serialize_cdr
deserialize_cdr = typestore.deserialize_cdr
serialize_ros1 = typestore.serialize_ros1
deserialize_ros1 = typestore.deserialize_ros1
cdr_to_ros1 = typestore.cdr_to_ros1
ros1_to_cdr = typestore.ros1_to_cdr

# Register Livox custom message types (livox_ros_driver / livox_ros_driver2)
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

# Import message types from the typestore
Header = typestore.types["std_msgs/msg/Header"]
Time = typestore.types["builtin_interfaces/msg/Time"]
Vector3 = typestore.types["geometry_msgs/msg/Vector3"]
Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
Imu = typestore.types["sensor_msgs/msg/Imu"]
PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
PointField = typestore.types["sensor_msgs/msg/PointField"]


# ---------------------------------------------------------------------------
# Safe deserialization helpers (work around Python 3.14 bug in deserialize_ros1)
# ---------------------------------------------------------------------------
def _deserialize_ros1_safe(rawdata: bytes, msgtype: str):
    """Deserialize ROS1 raw data. Falls back to ros1->cdr->deserialize path."""
    try:
        return deserialize_ros1(rawdata, msgtype)
    except (UnicodeDecodeError, Exception):
        cdr = ros1_to_cdr(rawdata, msgtype)
        return deserialize_cdr(cdr, msgtype)


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


# ---------------------------------------------------------------------------
# Helpers: ROS time <-> seconds
# ---------------------------------------------------------------------------
def sec_to_rostime(t: float) -> Time:
    sec = int(t)
    nsec = int((t - sec) * 1e9)
    return Time(sec=sec, nanosec=nsec)


def rostime_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def rostime_to_nsec(stamp) -> int:
    return int(stamp.sec) * 10**9 + int(stamp.nanosec)


def sec_to_nsec(t: float) -> int:
    return int(t * 1e9)


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
        ts_sec = ts_ns / 1e9
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

    # We need ROS1 typestore for serialization
    typestore_ros1 = get_typestore(Stores.ROS2_HUMBLE)

    with Writer1(output_bag) as writer:
        imu_conn = writer.add_connection(imu_topic, typestore_ros1.types["sensor_msgs/msg/Imu"].__msgtype__)
        pc_conn = writer.add_connection(pc_topic, typestore_ros1.types["sensor_msgs/msg/PointCloud2"].__msgtype__)

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
) -> None:
    print(f"Converting ROS1 bag -> MandEye")
    print(f"  Input:  {input_bag}")
    print(f"  Output: {output_dir}")
    print(f"  PC topic:  {pc_topic}")
    print(f"  IMU topic: {imu_topic}")
    print(f"  Chunk len: {chunk_len}s")

    os.makedirs(output_dir, exist_ok=True)

    bag_files = []
    p = Path(input_bag)
    if p.suffix == ".bag":
        bag_files = [p]
    else:
        bag_files = sorted(p.glob("*.bag"))

    buffer_pc: List[PointXYZIT] = []
    buffer_imu: List[str] = []
    last_save_ts = 0.0
    count = 0
    last_imu_ts = -1.0
    lidar_frame_rate = 0.0

    for bag_path in bag_files:
        print(f"  Processing bag: {bag_path.name}")

        # Detect lidar message type
        is_custom_msg = False
        with Reader1(bag_path) as reader:
            print("  Topics in bag:")
            for c in reader.connections:
                print(f"    {c.topic}  [{c.msgtype}]  ({c.msgcount} msgs)")
                if c.topic == pc_topic and "CustomMsg" in c.msgtype:
                    is_custom_msg = True

        # Pass 1: estimate frame rate if needed
        if emulate_point_ts:
            print("  Emulating point timestamps, collecting framerate...")
            header_diffs = []
            last_header_ts = 0.0
            start_ts = 0.0
            with Reader1(bag_path) as reader:
                for conn, timestamp, rawdata in reader.messages():
                    if conn.topic == pc_topic:
                        msg = _deserialize_ros1_safe(rawdata, conn.msgtype)
                        ts = rostime_to_sec(msg.header.stamp)
                        if last_header_ts != 0.0:
                            if start_ts == 0.0:
                                start_ts = last_header_ts
                            diff = ts - last_header_ts
                            if diff > 0:
                                header_diffs.append(diff)
                            if ts - start_ts > 20.0:
                                break
                        last_header_ts = ts

            if header_diffs:
                lidar_frame_rate = sum(header_diffs) / len(header_diffs)
                print(f"  Estimated frame rate: {lidar_frame_rate:.6f}s")

        # Pass 2: extract data
        with Reader1(bag_path) as reader:
            for conn, timestamp, rawdata in reader.messages():
                msg_time_sec = timestamp / 1e9

                if conn.topic == imu_topic and "Imu" in conn.msgtype:
                    msg = _deserialize_ros1_safe(rawdata, conn.msgtype)
                    ts_ns = rostime_to_nsec(msg.header.stamp)
                    d = csv_delim
                    line = (
                        f"{ts_ns}{d}"
                        f"{msg.angular_velocity.x}{d}{msg.angular_velocity.y}{d}{msg.angular_velocity.z}{d}"
                        f"{msg.linear_acceleration.x}{d}{msg.linear_acceleration.y}{d}"
                        f"{msg.linear_acceleration.z}"
                    )
                    buffer_imu.append(line)
                    last_imu_ts = rostime_to_sec(msg.header.stamp)
                    if last_save_ts == 0.0:
                        last_save_ts = last_imu_ts

                if conn.topic == pc_topic and last_imu_ts > 0:
                    msg = _deserialize_ros1_safe(rawdata, conn.msgtype)
                    ts = rostime_to_sec(msg.header.stamp)
                    if abs(ts - last_imu_ts) < 0.05 * chunk_len:
                        if is_custom_msg:
                            pts = parse_custom_msg(msg)
                        else:
                            pts = parse_pointcloud2(msg, emulate_point_ts, lidar_frame_rate)
                        buffer_pc.extend(pts)
                    else:
                        print(f"  Skipping pointcloud at {ts:.3f}s (IMU drift: {abs(ts - last_imu_ts):.3f}s)")

                if msg_time_sec - last_save_ts > chunk_len and last_save_ts > 0:
                    _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                                csv_delim=csv_delim, imu_id=imu_id, serial=serial)
                    buffer_pc.clear()
                    buffer_imu.clear()
                    last_save_ts = msg_time_sec
                    count += 1

    if buffer_pc or buffer_imu:
        _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                    csv_delim=csv_delim, imu_id=imu_id, serial=serial)

    print("Done!")


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
) -> None:
    print(f"Converting ROS2 bag -> MandEye")
    print(f"  Input:  {input_bag}")
    print(f"  Output: {output_dir}")
    print(f"  PC topic:  {pc_topic}")
    print(f"  IMU topic: {imu_topic}")
    print(f"  Chunk len: {chunk_len}s")

    os.makedirs(output_dir, exist_ok=True)

    buffer_pc: List[PointXYZIT] = []
    buffer_imu: List[str] = []
    last_save_ts = 0.0
    count = 0
    last_imu_ts = -1.0
    lidar_frame_rate = 0.0

    # Pass 1: estimate frame rate if needed
    if emulate_point_ts:
        print("  Emulating point timestamps, collecting framerate...")
        header_diffs = []
        last_header_ts = 0.0
        start_ts = 0.0
        with Reader2(input_bag) as reader:
            for conn, timestamp, rawdata in reader.messages():
                if conn.topic == pc_topic:
                    msg = deserialize_cdr(rawdata, conn.msgtype)
                    ts = rostime_to_sec(msg.header.stamp)
                    if last_header_ts != 0.0:
                        if start_ts == 0.0:
                            start_ts = last_header_ts
                        diff = ts - last_header_ts
                        if diff > 0:
                            header_diffs.append(diff)
                        if ts - start_ts > 20.0:
                            break
                    last_header_ts = ts

        if header_diffs:
            lidar_frame_rate = sum(header_diffs) / len(header_diffs)
            print(f"  Estimated frame rate: {lidar_frame_rate:.6f}s")

    # Pass 2: extract data
    with Reader2(input_bag) as reader:
        for conn, timestamp, rawdata in reader.messages():
            msg_time_sec = timestamp / 1e9

            if conn.topic == imu_topic:
                msg = deserialize_cdr(rawdata, conn.msgtype)
                ts_ns = rostime_to_nsec(msg.header.stamp)
                d = csv_delim
                line = (
                    f"{ts_ns}{d}"
                    f"{msg.angular_velocity.x}{d}{msg.angular_velocity.y}{d}{msg.angular_velocity.z}{d}"
                    f"{msg.linear_acceleration.x}{d}{msg.linear_acceleration.y}{d}"
                    f"{msg.linear_acceleration.z}"
                )
                buffer_imu.append(line)
                last_imu_ts = rostime_to_sec(msg.header.stamp)
                if last_save_ts == 0.0:
                    last_save_ts = last_imu_ts

            if conn.topic == pc_topic and last_imu_ts > 0:
                msg = deserialize_cdr(rawdata, conn.msgtype)
                ts = rostime_to_sec(msg.header.stamp)
                if abs(ts - last_imu_ts) < 0.05 * chunk_len:
                    pts = parse_pointcloud2(msg, emulate_point_ts, lidar_frame_rate)
                    buffer_pc.extend(pts)
                else:
                    print(f"  Skipping pointcloud at {ts:.3f}s (IMU drift: {abs(ts - last_imu_ts):.3f}s)")

            if msg_time_sec - last_save_ts > chunk_len and last_save_ts > 0:
                _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                            csv_delim=csv_delim, imu_id=imu_id, serial=serial)
                buffer_pc.clear()
                buffer_imu.clear()
                last_save_ts = msg_time_sec
                count += 1

    if buffer_pc or buffer_imu:
        _save_chunk(output_dir, count, buffer_pc, buffer_imu,
                    csv_delim=csv_delim, imu_id=imu_id, serial=serial)

    print("Done!")


# ---------------------------------------------------------------------------
# Resolve unique output path:  base, base_000, base_001, ...
# ---------------------------------------------------------------------------
def resolve_unique_output_path(base: str) -> str:
    """Return *base* if it doesn't exist yet, otherwise base_000, base_001, …"""
    candidate = base
    if not os.path.exists(candidate):
        return candidate
    idx = 0
    while True:
        candidate = f"{base}_{idx:03d}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


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
    """Read audit JSON and override --pointcloud_topic / --imu_topic from best pair."""
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

    # Print best pair info
    pairs = data.get("pairs", [])
    best = pairs[0] if pairs else {}
    grade = best.get("grade", "?")
    total = best.get("total", 0)

    print(f"  Audit JSON loaded: {json_path}")
    print(f"    Bag:    {data.get('bag', '?')}")
    print(f"    Best:   {pc_topic} + {imu_topic}  [{grade} {total:.0%}]")
    print(f"    Clock:  {clock}  (offset ≈ {offset_ms:.1f} ms)")
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

Examples:
  python mandeye_convert.py ./my_dataset ./output.bag hdmapping-to-ros1
  python mandeye_convert.py ./output.bag ./extracted   ros1-to-hdmapping
  python mandeye_convert.py ./my_dataset ./output_ros2 hdmapping-to-ros2
  python mandeye_convert.py ./output_ros2 ./extracted  ros2-to-hdmapping

  # Auto-indexed output: export_git, export_git_000, export_git_001, …
  python mandeye_convert.py deg-vis-1.bag export_git ros1-to-hdmapping
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
             "--pointcloud_topic, --imu_topic, and print clock info",
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

    if args.mode == "hdmapping-to-ros1":
        hdmapping_to_ros1(
            args.input,
            output,
            num_lines=args.lines,
            imu_topic=args.imu_topic,
            pc_topic=args.pointcloud_topic,
        )
    elif args.mode == "ros1-to-hdmapping":
        ros1_to_hdmapping(
            args.input,
            output,
            pc_topic=args.pointcloud_topic,
            imu_topic=args.imu_topic,
            chunk_len=args.chunk_len,
            emulate_point_ts=args.emulate_point_ts,
            csv_delim=csv_delim,
            imu_id=args.imu_id,
            serial=args.serial,
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
        ros2_to_hdmapping(
            args.input,
            output,
            pc_topic=args.pointcloud_topic,
            imu_topic=args.imu_topic,
            chunk_len=args.chunk_len,
            emulate_point_ts=args.emulate_point_ts,
            csv_delim=csv_delim,
            imu_id=args.imu_id,
            serial=args.serial,
        )


if __name__ == "__main__":
    main()
