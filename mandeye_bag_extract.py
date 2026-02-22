#!/usr/bin/env python3
"""
mandeye_bag_extract.py  v0.1 — Selective topic extraction from ROS bag files.

Reads a ROS1 (.bag) or ROS2 bag and exports chosen topics to a filtered bag
file and/or CSV files, with automatic topic-type identification.

Supported output formats (--format):
  bag   — write a new ROS1 bag containing only the selected topics
  csv   — write per-topic data files:
           • IMU, NavSatFix, NMEA, Odometry, TF → CSV
           • PointCloud2, Livox CustomMsg → LAZ (compressed point cloud)
           • CompressedImage → original format (jpg/png/tif/bmp/webp)
           • Image → PNG or TIFF (requires Pillow)
  both  — produce both a filtered bag AND data files

Topic selection (--topics):
  Specify one or more topic names or glob patterns.
  Examples:
    --topics /livox/imu                  single topic
    --topics "/livox/*"                  all topics under /livox/
    --topics /imu /gps/fix               multiple topics
    --topics "*"                         all topics (bag subset)

Topic type identification:
  Each topic is automatically classified as one of:
    imu, pointcloud, image, compressed_image, navsatfix, nmea,
    odometry, tf, other

Multi-volume / split bag sequences:
  --sequence    process all bags in a detected sequence
  --no-sequence suppress sequence detection

Listing mode:
  --list        show all topics with type classification and exit

Dependencies:  pip install rosbags numpy laspy[lazrs] Pillow

Usage:
  python mandeye_bag_extract.py recording.bag --list
  python mandeye_bag_extract.py recording.bag -o out --topics /livox/imu
  python mandeye_bag_extract.py recording.bag -o out --topics "/livox/*" --format both
  python mandeye_bag_extract.py recording.bag -o out --topics /imu --format csv
  python mandeye_bag_extract.py ./bag_dir/ -o out --topics "*" --format bag --sequence
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import fnmatch
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import laspy
except ImportError:
    laspy = None  # LAZ export unavailable

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None  # raw Image export unavailable (compressed still works)

from mandeye_bag_common import (
    Reader1, Reader1Error, Reader2,
    Writer1,
    typestore,
    deserialize_ros1_safe,
    deserialize_cdr,
    rostime_to_sec, rostime_to_nsec,
    classify_topic,
    is_imu_type, is_pc_type, is_custom_msg, is_navsatfix_type, is_nmea_type,
    is_odometry_type, is_tf_type,
    is_image_type, is_compressed_image_type,
    rostime_to_nsec as _rostime_to_nsec,
    detect_bag_sequence,
    validate_bag_sequence,
    print_sequence_summary,
    print_sequence_info,
    add_sequence_args,
    detect_dir_bags,
    resolve_unique_output_path,
)


# ============================================================================
# Topic listing
# ============================================================================

def list_topics(bag_path: Path) -> List[Dict[str, Any]]:
    """Return a list of dicts with topic info (topic, msgtype, count, category)."""
    is_ros1 = bag_path.suffix == ".bag"
    topics: List[Dict[str, Any]] = []
    if is_ros1:
        with Reader1(bag_path) as reader:
            for c in reader.connections:
                topics.append({
                    "topic": c.topic,
                    "msgtype": c.msgtype,
                    "count": c.msgcount,
                    "category": classify_topic(c.msgtype),
                })
    else:
        with Reader2(bag_path) as reader:
            for c in reader.connections:
                topics.append({
                    "topic": c.topic,
                    "msgtype": c.msgtype,
                    "count": getattr(c, "msgcount", None),
                    "category": classify_topic(c.msgtype),
                })
    return topics


def print_topic_list(bag_path: Path) -> None:
    """Print a formatted topic listing with type classification."""
    topics = list_topics(bag_path)
    if not topics:
        print("  (no topics found)")
        return
    # Column widths
    tw = max(len(t["topic"]) for t in topics)
    mw = max(len(t["msgtype"]) for t in topics)
    print(f"\n  {'TOPIC':<{tw}}  {'TYPE':<{mw}}  {'COUNT':>8}  CATEGORY")
    print(f"  {'─'*tw}  {'─'*mw}  {'─'*8}  {'─'*18}")
    for t in sorted(topics, key=lambda x: x["topic"]):
        cnt = t["count"] if t["count"] is not None else "?"
        print(f"  {t['topic']:<{tw}}  {t['msgtype']:<{mw}}  {cnt:>8}  {t['category']}")
    print()


# ============================================================================
# Topic matching
# ============================================================================

def match_topics(
    available: List[Dict[str, Any]],
    patterns: List[str],
) -> List[Dict[str, Any]]:
    """Filter *available* topics by glob patterns applied to topic names."""
    matched: List[Dict[str, Any]] = []
    seen: set = set()
    for pat in patterns:
        for t in available:
            if t["topic"] not in seen and fnmatch.fnmatch(t["topic"], pat):
                matched.append(t)
                seen.add(t["topic"])
    return matched


# ============================================================================
# CSV exporters (per topic category)
# ============================================================================

def _csv_path(output_dir: str, topic: str, suffix: str = ".csv") -> str:
    """Build a safe CSV filename from a topic name."""
    safe = topic.strip("/").replace("/", "_")
    return os.path.join(output_dir, safe + suffix)


def _write_imu_csv(path: str, rows: List[List[Any]]) -> int:
    """Write IMU samples to CSV. Returns row count."""
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["timestamp_s", "gyr_x", "gyr_y", "gyr_z",
                     "acc_x", "acc_y", "acc_z"])
        w.writerows(rows)
    return len(rows)


def _write_navsatfix_csv(path: str, rows: List[List[Any]]) -> int:
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["timestamp_s", "latitude", "longitude", "altitude",
                     "status", "cov_type"])
        w.writerows(rows)
    return len(rows)


def _write_nmea_csv(path: str, rows: List[List[Any]]) -> int:
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["timestamp_s", "sentence"])
        w.writerows(rows)
    return len(rows)


def _write_odometry_csv(path: str, rows: List[List[Any]]) -> int:
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["timestamp_s",
                     "pos_x", "pos_y", "pos_z",
                     "ori_x", "ori_y", "ori_z", "ori_w",
                     "lin_x", "lin_y", "lin_z",
                     "ang_x", "ang_y", "ang_z"])
        w.writerows(rows)
    return len(rows)


def _write_tf_csv(path: str, rows: List[List[Any]]) -> int:
    with open(path, "w", newline="") as f:
        w = csv_mod.writer(f)
        w.writerow(["timestamp_s", "parent_frame", "child_frame",
                     "tx", "ty", "tz", "rx", "ry", "rz", "rw"])
        w.writerows(rows)
    return len(rows)


# ============================================================================
# Pointcloud → LAZ export
# ============================================================================

def _parse_pointcloud2_points(
    msg, is_ros1: bool,
) -> List[Tuple[float, float, float, float, float]]:
    """Parse a PointCloud2 message into (x, y, z, intensity, timestamp_ns) tuples."""
    field_map = {f.name: f for f in msg.fields}
    x_off = field_map["x"].offset
    y_off = field_map["y"].offset
    z_off = field_map["z"].offset
    i_field = field_map.get("intensity", field_map.get("i", None))
    i_off = i_field.offset if i_field else -1

    data = bytes(msg.data)
    ps = msg.point_step
    n = msg.width * msg.height
    header_ts_ns = _rostime_to_nsec(msg.header.stamp)

    points: List[Tuple[float, float, float, float, float]] = []
    for idx in range(n):
        base = idx * ps
        x = struct.unpack_from("<f", data, base + x_off)[0]
        y = struct.unpack_from("<f", data, base + y_off)[0]
        z = struct.unpack_from("<f", data, base + z_off)[0]
        intensity = (
            struct.unpack_from("<f", data, base + i_off)[0]
            if i_off >= 0 else 0.0
        )
        points.append((x, y, z, intensity, float(header_ts_ns)))
    return points


def _parse_custom_msg_points(
    msg,
) -> List[Tuple[float, float, float, float, float]]:
    """Parse a Livox CustomMsg into (x, y, z, intensity, timestamp_ns) tuples."""
    points: List[Tuple[float, float, float, float, float]] = []
    timebase = msg.timebase  # nanoseconds
    for p in msg.points:
        ts_ns = timebase + p.offset_time
        points.append((p.x, p.y, p.z, float(p.reflectivity), float(ts_ns)))
    return points


def _write_pointcloud_laz(
    path: str,
    points: List[Tuple[float, float, float, float, float]],
) -> int:
    """Write pointcloud data to a LAZ file.  Returns point count."""
    if laspy is None:
        print("  WARNING: laspy not installed, skipping LAZ export. "
              "Install with: pip install laspy[lazrs]")
        return 0
    if not points:
        return 0

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    zs = np.array([p[2] for p in points], dtype=np.float64)
    intensities = np.array([p[3] for p in points], dtype=np.uint16)
    timestamps = np.array([p[4] for p in points], dtype=np.float64)

    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = [float(np.min(xs)), float(np.min(ys)), float(np.min(zs))]
    header.scales = [0.0001, 0.0001, 0.0001]

    las = laspy.LasData(header)
    las.x = xs
    las.y = ys
    las.z = zs
    las.intensity = intensities
    las.gps_time = timestamps * 1e-9  # nanoseconds → seconds

    las.write(path)
    return len(points)


# ============================================================================
# Image export
# ============================================================================

# Map compressed_image format string → file extension
_COMPRESSED_FMT_EXT: Dict[str, str] = {
    "jpeg": ".jpg", "jpg": ".jpg",
    "png": ".png",
    "tiff": ".tif", "tif": ".tif",
    "bmp": ".bmp",
    "webp": ".webp",
}


def _image_dir(output_dir: str, topic: str) -> str:
    """Return (and create) a per-topic subfolder for image files."""
    safe = topic.strip("/").replace("/", "_")
    d = os.path.join(output_dir, safe)
    os.makedirs(d, exist_ok=True)
    return d


def _ts_filename(timestamp_ns: int) -> str:
    """Build a zero-padded filename from a nanosecond timestamp."""
    sec = timestamp_ns // 1_000_000_000
    nsec = timestamp_ns % 1_000_000_000
    return f"{sec:010d}_{nsec:09d}"


def _save_compressed_image(
    output_dir: str, topic: str, msg: Any, timestamp_ns: int,
) -> Optional[str]:
    """Save a CompressedImage message to disk.  Returns path or None."""
    fmt_raw = getattr(msg, "format", "jpeg")
    # format can be e.g. "jpeg", "png", or "jpeg; quality=90"
    fmt_key = fmt_raw.split(";")[0].strip().lower()
    ext = _COMPRESSED_FMT_EXT.get(fmt_key, "." + fmt_key)

    img_dir = _image_dir(output_dir, topic)
    fname = _ts_filename(timestamp_ns) + ext
    path = os.path.join(img_dir, fname)

    data = bytes(msg.data)
    if not data:
        return None
    with open(path, "wb") as f:
        f.write(data)
    return path


# Encoding → (PIL mode, bytes-per-channel, channels, needs_swap)
_RAW_ENC_MAP: Dict[str, Tuple[str, int, int, bool]] = {
    "mono8":   ("L",    1, 1, False),
    "8UC1":    ("L",    1, 1, False),
    "mono16":  ("I;16", 2, 1, False),
    "16UC1":   ("I;16", 2, 1, False),
    "rgb8":    ("RGB",  1, 3, False),
    "8UC3":    ("RGB",  1, 3, False),
    "bgr8":    ("RGB",  1, 3, True),
    "rgba8":   ("RGBA", 1, 4, False),
    "bgra8":   ("RGBA", 1, 4, True),
}


def _save_raw_image(
    output_dir: str, topic: str, msg: Any, timestamp_ns: int,
) -> Optional[str]:
    """Save a sensor_msgs/Image message to disk.  Returns path or None."""
    if PILImage is None:
        return None

    encoding = getattr(msg, "encoding", "").lower()
    height = msg.height
    width = msg.width
    data = bytes(msg.data)
    if not data or height == 0 or width == 0:
        return None

    enc_info = _RAW_ENC_MAP.get(encoding)
    if enc_info is None:
        # Unknown encoding — save raw bytes with .raw extension
        img_dir = _image_dir(output_dir, topic)
        fname = _ts_filename(timestamp_ns) + f".{encoding}.raw"
        path = os.path.join(img_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        return path

    pil_mode, bpc, channels, needs_swap = enc_info

    try:
        arr = np.frombuffer(data, dtype=np.uint8 if bpc == 1 else np.uint16)
        arr = arr.reshape((height, width, channels) if channels > 1
                          else (height, width))
        if needs_swap and channels >= 3:
            # BGR(A) → RGB(A): swap R and B channels
            arr = arr.copy()
            arr[..., 0], arr[..., 2] = arr[..., 2].copy(), arr[..., 0].copy()

        img = PILImage.fromarray(arr, mode=pil_mode)

        # Choose output format: 16-bit → TIFF, else PNG
        if bpc == 2:
            ext = ".tif"
        else:
            ext = ".png"

        img_dir = _image_dir(output_dir, topic)
        fname = _ts_filename(timestamp_ns) + ext
        path = os.path.join(img_dir, fname)
        img.save(path)
        return path
    except Exception:
        return None


# ============================================================================
# Extraction core
# ============================================================================

def _deserialize_msg(raw: bytes, msgtype: str, is_ros1: bool):
    """Deserialize a message, returns msg or None on failure."""
    try:
        if is_ros1:
            return deserialize_ros1_safe(raw, msgtype)
        else:
            return deserialize_cdr(raw, msgtype)
    except Exception:
        return None


def extract_from_bag(
    bag_path: Path,
    selected_topics: List[Dict[str, Any]],
    output_dir: str,
    write_bag: bool = True,
    write_csv: bool = False,
    bag_files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Extract selected topics from one or more bag files.

    Parameters
    ----------
    bag_path : Path
        Primary bag file (used for naming).
    selected_topics : list
        Topic info dicts (from ``match_topics``).
    output_dir : str
        Output directory.
    write_bag : bool
        Write a filtered .bag file.
    write_csv : bool
        Write per-topic CSV files (where supported).
    bag_files : list or None
        If given, iterate over these bags (multi-volume sequence).

    Returns
    -------
    dict
        Extraction report.
    """
    os.makedirs(output_dir, exist_ok=True)
    topic_names = {t["topic"] for t in selected_topics}
    topic_meta = {t["topic"]: t for t in selected_topics}

    files_to_process = bag_files if bag_files else [bag_path]
    is_ros1 = bag_path.suffix == ".bag"

    # accumulators for CSV / LAZ
    csv_data: Dict[str, List[List[Any]]] = {t: [] for t in topic_names}
    pc_points: Dict[str, List[Tuple[float, float, float, float, float]]] = {
        t: [] for t in topic_names
        if topic_meta[t]["category"] == "pointcloud"
    }
    img_counts: Dict[str, int] = {}
    img_dirs: Dict[str, str] = {}
    msg_counts: Dict[str, int] = {t: 0 for t in topic_names}
    warnings: List[str] = []

    t0 = time.time()

    # --- Filtered bag output ---
    bag_out_path = os.path.join(output_dir, bag_path.stem + "_filtered.bag")
    writer = None
    if write_bag and Writer1 is not None:
        writer = Writer1(bag_out_path)
        writer.open()

    for bag_file in files_to_process:
        cur_is_ros1 = bag_file.suffix == ".bag"
        try:
            ReaderCls = Reader1 if cur_is_ros1 else Reader2
            with ReaderCls(bag_file) as reader:
                # Find connections matching our topics
                conns = [c for c in reader.connections if c.topic in topic_names]
                if not conns:
                    continue

                # Create writer connections
                writer_conns: Dict[str, Any] = {}
                if writer is not None:
                    for c in conns:
                        if c.topic not in writer_conns:
                            wc = writer.add_connection(c.topic, c.msgtype)
                            writer_conns[c.topic] = wc

                try:
                    for conn, timestamp, rawdata in reader.messages(
                        connections=conns
                    ):
                        topic = conn.topic
                        msg_counts[topic] = msg_counts.get(topic, 0) + 1

                        # Write to filtered bag
                        if writer is not None and topic in writer_conns:
                            writer.write(writer_conns[topic], timestamp, rawdata)

                        # Collect CSV / LAZ / image data
                        if write_csv:
                            cat = topic_meta[topic]["category"]
                            if cat == "pointcloud" and topic in pc_points:
                                msg = _deserialize_msg(
                                    rawdata, conn.msgtype, cur_is_ros1,
                                )
                                if msg is not None:
                                    if is_custom_msg(conn.msgtype):
                                        pts = _parse_custom_msg_points(msg)
                                    else:
                                        pts = _parse_pointcloud2_points(
                                            msg, cur_is_ros1,
                                        )
                                    pc_points[topic].extend(pts)
                            elif cat == "compressed_image":
                                msg = _deserialize_msg(
                                    rawdata, conn.msgtype, cur_is_ros1,
                                )
                                if msg is not None:
                                    p = _save_compressed_image(
                                        output_dir, topic, msg, timestamp,
                                    )
                                    if p:
                                        img_counts[topic] = (
                                            img_counts.get(topic, 0) + 1
                                        )
                                        if topic not in img_dirs:
                                            img_dirs[topic] = os.path.dirname(p)
                            elif cat == "image":
                                if PILImage is not None:
                                    msg = _deserialize_msg(
                                        rawdata, conn.msgtype, cur_is_ros1,
                                    )
                                    if msg is not None:
                                        p = _save_raw_image(
                                            output_dir, topic, msg, timestamp,
                                        )
                                        if p:
                                            img_counts[topic] = (
                                                img_counts.get(topic, 0) + 1
                                            )
                                            if topic not in img_dirs:
                                                img_dirs[topic] = os.path.dirname(p)
                                else:
                                    # Warn once per topic
                                    _k = f"_pil_warn_{topic}"
                                    if _k not in img_dirs:
                                        img_dirs[_k] = ""
                                        w = ("WARNING: Pillow not installed, "
                                             f"cannot export raw Image topic {topic}. "
                                             "Install with: pip install Pillow")
                                        print(f"  {w}")
                                        warnings.append(w)
                            elif cat in ("imu", "navsatfix", "nmea",
                                         "odometry", "tf"):
                                msg = _deserialize_msg(
                                    rawdata, conn.msgtype, cur_is_ros1,
                                )
                                if msg is not None:
                                    ts = timestamp / 1e9
                                    _collect_csv_row(
                                        csv_data, topic, cat, msg, ts,
                                    )

                except (Reader1Error, Exception) as exc:
                    w = (f"WARNING: {bag_file.name} read error "
                         f"(truncated/corrupt?): {exc}")
                    print(f"  {w}")
                    warnings.append(w)

        except Exception as exc:
            w = f"WARNING: Cannot open {bag_file.name}: {exc}"
            print(f"  {w}")
            warnings.append(w)

    if writer is not None:
        writer.close()
        total_msgs = sum(msg_counts.values())
        print(f"  Filtered bag: {bag_out_path}  ({total_msgs} messages)")

    # --- Write LAZ files for pointcloud topics ---
    laz_files: Dict[str, str] = {}
    if write_csv:
        for topic in sorted(pc_points):
            pts = pc_points[topic]
            if not pts:
                continue
            path = _csv_path(output_dir, topic, suffix=".laz")
            n = _write_pointcloud_laz(path, pts)
            if n > 0:
                laz_files[topic] = path
                print(f"  LAZ: {path}  ({n:,} points)")

    # --- Image summary ---
    if write_csv:
        for topic in sorted(img_counts):
            n = img_counts[topic]
            d = img_dirs.get(topic, "")
            if n > 0:
                print(f"  Images: {d}  ({n:,} files)")

    # --- Write CSV files ---
    csv_files: Dict[str, str] = {}
    if write_csv:
        for topic in sorted(topic_names):
            cat = topic_meta[topic]["category"]
            if cat == "pointcloud":
                continue  # already exported as LAZ above
            rows = csv_data.get(topic, [])
            if not rows:
                continue
            path = _csv_path(output_dir, topic)
            n = _write_csv_for_category(path, cat, rows)
            if n > 0:
                csv_files[topic] = path
                print(f"  CSV: {path}  ({n} rows)")

    elapsed = time.time() - t0

    # --- Build report ---
    report: Dict[str, Any] = {
        "tool": "mandeye_bag_extract",
        "version": "0.1",
        "source": str(bag_path),
        "output_dir": output_dir,
        "elapsed_sec": round(elapsed, 2),
        "topics_extracted": [
            {
                "topic": t["topic"],
                "msgtype": t["msgtype"],
                "category": t["category"],
                "messages": msg_counts.get(t["topic"], 0),
                "csv_file": csv_files.get(t["topic"]),
                "laz_file": laz_files.get(t["topic"]),
                "image_dir": img_dirs.get(t["topic"]),
                "image_count": img_counts.get(t["topic"], 0),
            }
            for t in selected_topics
        ],
    }
    if write_bag and writer is not None:
        report["filtered_bag"] = bag_out_path
    if bag_files and len(bag_files) > 1:
        report["source_bags"] = [str(b) for b in bag_files]
    if warnings:
        report["warnings"] = warnings

    # Write report JSON
    report_path = os.path.join(output_dir, "extract_report.json")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report: {report_path}")
    except Exception as exc:
        print(f"  WARNING: Could not write report: {exc}", file=sys.stderr)

    return report


def _collect_csv_row(
    csv_data: Dict[str, List],
    topic: str,
    category: str,
    msg: Any,
    timestamp_sec: float,
) -> None:
    """Parse a deserialized message and append a CSV row."""
    try:
        if category == "imu":
            a = msg.linear_acceleration
            g = msg.angular_velocity
            ts = rostime_to_sec(msg.header.stamp)
            csv_data[topic].append([
                ts, g.x, g.y, g.z, a.x, a.y, a.z,
            ])
        elif category == "navsatfix":
            ts = rostime_to_sec(msg.header.stamp)
            csv_data[topic].append([
                ts, msg.latitude, msg.longitude, msg.altitude,
                int(msg.status.status), int(msg.position_covariance_type),
            ])
        elif category == "nmea":
            csv_data[topic].append([timestamp_sec, msg.sentence])
        elif category == "odometry":
            ts = rostime_to_sec(msg.header.stamp)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            lv = msg.twist.twist.linear
            av = msg.twist.twist.angular
            csv_data[topic].append([
                ts, p.x, p.y, p.z, o.x, o.y, o.z, o.w,
                lv.x, lv.y, lv.z, av.x, av.y, av.z,
            ])
        elif category == "tf":
            for tf in msg.transforms:
                ts = rostime_to_sec(tf.header.stamp)
                t = tf.transform.translation
                r = tf.transform.rotation
                csv_data[topic].append([
                    ts, tf.header.frame_id, tf.child_frame_id,
                    t.x, t.y, t.z, r.x, r.y, r.z, r.w,
                ])
    except Exception:
        pass  # skip malformed messages silently


def _write_csv_for_category(
    path: str, category: str, rows: List[List[Any]],
) -> int:
    """Dispatch CSV writing to the appropriate per-category writer."""
    if category == "imu":
        return _write_imu_csv(path, rows)
    elif category == "navsatfix":
        return _write_navsatfix_csv(path, rows)
    elif category == "nmea":
        return _write_nmea_csv(path, rows)
    elif category == "odometry":
        return _write_odometry_csv(path, rows)
    elif category == "tf":
        return _write_tf_csv(path, rows)
    return 0


# ============================================================================
# CLI
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Extract selected topics from a ROS bag file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Topic type categories (auto-detected):
  imu              sensor_msgs/Imu
  pointcloud       sensor_msgs/PointCloud2, livox CustomMsg
  image            sensor_msgs/Image
  compressed_image sensor_msgs/CompressedImage
  navsatfix        sensor_msgs/NavSatFix
  nmea             nmea_msgs/Sentence
  odometry         nav_msgs/Odometry
  tf               tf2_msgs/TFMessage
  other            everything else

Data export (--format csv / both):
  imu, navsatfix, nmea, odometry, tf  ->  CSV files
  pointcloud (PointCloud2, CustomMsg)  ->  LAZ (compressed point cloud)
  compressed_image (CompressedImage)   ->  jpg/png/tif/bmp/webp (original)
  image (Image)                        ->  PNG or TIFF (requires Pillow)

Examples:
  python mandeye_bag_extract.py recording.bag --list
  python mandeye_bag_extract.py recording.bag -o out --topics /livox/imu
  python mandeye_bag_extract.py recording.bag -o out --topics "/livox/*" --format both
  python mandeye_bag_extract.py recording.bag -o out --topics /imu --format csv
  python mandeye_bag_extract.py recording.bag -o out --topics /livox/lidar --format csv  # LAZ
  python mandeye_bag_extract.py recording.bag -o out --topics /camera/image --format csv  # images
  python mandeye_bag_extract.py ./bag_dir/ -o out --topics "*" --format bag --sequence
""",
    )
    p.add_argument("bag", help="Path to ROS1 .bag file, ROS2 bag folder, "
                   "or directory of .bag files")
    p.add_argument("-o", "--output", default="extracted",
                   help="Output directory (default: extracted)")
    p.add_argument("--topics", nargs="+", default=[],
                   help="Topic names or glob patterns to extract")
    p.add_argument("--format", choices=["bag", "csv", "both"], default="bag",
                   help="Output format: bag, csv, or both (default: bag)")
    p.add_argument("--list", action="store_true",
                   help="List all topics with type classification and exit")
    add_sequence_args(p)

    args = p.parse_args()

    bag_path = Path(args.bag)

    # --- Directory of bags ---
    dir_bags = detect_dir_bags(bag_path)
    is_dir_of_bags = bool(dir_bags)
    if is_dir_of_bags:
        bag_path = dir_bags[0]
        print(f"Directory contains {len(dir_bags)} .bag file(s)")

    is_ros1 = bag_path.suffix == ".bag"

    if is_ros1 and not bag_path.exists():
        sys.exit(f"ERROR: File not found: {bag_path}")
    if not is_ros1 and not bag_path.is_dir():
        sys.exit(f"ERROR: ROS2 bag folder not found: {bag_path}")

    # --- List mode ---
    if args.list:
        print(f"Topics in: {args.bag}")
        print_topic_list(bag_path)
        sys.exit(0)

    # --- Must have --topics for extraction ---
    if not args.topics:
        sys.exit("ERROR: No topics specified.  Use --topics or --list.")

    # --- Discover available topics ---
    available = list_topics(bag_path)
    selected = match_topics(available, args.topics)
    if not selected:
        print("No topics matched the given patterns.")
        print("Available topics:")
        for t in available:
            print(f"  {t['topic']}  [{t['msgtype']}]  {t['category']}")
        sys.exit(1)

    print(f"Selected {len(selected)} topic(s):")
    for t in selected:
        print(f"  {t['topic']}  [{t['msgtype']}]  {t['category']}")

    # --- Sequence detection ---
    bag_sequence: Optional[List[Path]] = None
    if is_ros1 and not args.no_sequence:
        if is_dir_of_bags:
            detected = dir_bags
        else:
            detected = detect_bag_sequence(bag_path)
        if len(detected) > 1:
            seq_infos = validate_bag_sequence(detected)
            print_sequence_summary(seq_infos)
            if args.sequence or is_dir_of_bags:
                bag_sequence = detected
                print(f"  Processing all {len(detected)} bags in sequence.\n")
            else:
                print(f"  INFO: Detected {len(detected)} bags in sequence.")
                print(f"         Use --sequence to process them all.\n")

    # --- Output ---
    output = resolve_unique_output_path(args.output)
    if output != args.output:
        print(f"Output '{args.output}' exists, using '{output}' instead.")

    write_bag = args.format in ("bag", "both")
    write_csv = args.format in ("csv", "both")

    report = extract_from_bag(
        bag_path=bag_path,
        selected_topics=selected,
        output_dir=output,
        write_bag=write_bag,
        write_csv=write_csv,
        bag_files=bag_sequence,
    )

    print(f"\nDone.  Extracted {sum(t['messages'] for t in report['topics_extracted'])} "
          f"messages in {report['elapsed_sec']:.1f}s.")


if __name__ == "__main__":
    main()
