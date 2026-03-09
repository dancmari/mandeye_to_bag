# # hdmapping-to-ros1 | ros1-to-hdmapping | hdmapping-to-ros2| ros2-to-hdmapping

# Simlified instruction

## Step 1 (prepare code)
```shell
mkdir -p ~/hdmapping-benchmark
cd ~/hdmapping-benchmark
git clone https://github.com/MapsHD/mandeye_to_bag.git --recursive
```

## Step 2 (build docker)
```shell
cd ~/hdmapping-benchmark/mandeye_to_bag
docker build -t mandeye-ws_noetic --target ros1 .
docker build -t mandeye-ws_humble --target ros2 .
```

## Step 3 (run docker)
```shell
cd ~/hdmapping-benchmark/mandeye_to_bag
chmod +x mandeye-convert.sh 
./mandeye-convert.sh <input_hdmapping_folder> <output_folder> hdmapping-to-ros1 //remark output_folder should not exists, it will be created
./mandeye-convert.sh <input_hdmapping_folder> <output_folder> hdmapping-to-ros2
./mandeye-convert.sh <input_ros1_bag> <output_folder> ros1-to-hdmapping
./mandeye-convert.sh <input_ros2_folder> <output_folder> ros2-to-hdmapping
```

## Dependencies

```shell
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
```
## Workspace

```shell
mkdir -p ~/hdmapping-benchmark
cd ~/hdmapping-benchmark
git clone https://github.com/MapsHD/mandeye_to_bag.git --recursive
```
## Docker build
```shell
cd ~/hdmapping-benchmark/mandeye_to_bag
docker build -t mandeye-ws_noetic --target ros1 .
docker build -t mandeye-ws_humble --target ros2 .
```

## Docker run
```shell
cd ~/hdmapping-benchmark/mandeye_to_bag
chmod +x mandeye-convert.sh 
./mandeye-convert.sh <input_hdmapping_folder> <output_folder> hdmapping-to-ros1
./mandeye-convert.sh <input_hdmapping_folder> <output_folder> hdmapping-to-ros2
./mandeye-convert.sh <input_ros1_bag> <output_folder> ros1-to-hdmapping
./mandeye-convert.sh <input_ros2_folder> <output_folder> ros2-to-hdmapping
```

---

## Python tools (standalone, no ROS needed)

Three standalone Python scripts provide the same conversion capabilities
**without Docker or a ROS installation** — only pure-Python packages are
required.

### Install dependencies

```shell
pip install rosbags numpy "laspy[lazrs]"
# optional – for raw Image export:
pip install Pillow
```

### Scripts

| Script | Purpose |
|--------|---------|| `mandeye_check_deps.py` | Validate Python environment and installed dependencies || `mandeye_bag_audit.py` | Audit a ROS bag: score every (PointCloud, IMU) pair, detect units, export JSON |
| `mandeye_bag_convert.py` | Convert between MandEye datasets and ROS1/ROS2 bag files |
| `mandeye_bag_extract.py` | Extract selected topics to a filtered bag and/or CSV/LAZ/image files |
| `mandeye_imu_rescale.py` | Post-extraction IMU unit fix — rescale acc / gyro / timestamp in CSV files |
| `mandeye_bag_common.py` | Shared library used by the scripts above |

### MandEye file format

A MandEye / HDMapping dataset folder contains chunks of equal duration:

| File | Content | Units |
|------|---------|-------|
| `pointcloud_NNNN.laz` | LiDAR points: x, y, z, intensity, GPS time | m (xyz), s (time) |
| `imu_NNNN.csv` | IMU rows: `timestamp gyroX gyroY gyroZ accX accY accZ imuId` | ns (timestamp), **deg/s** (gyro), **g** (accel) |
| `lidarNNNN.sn` | Serial number info: `imuId serialNumber` | — |

> **Note:** When converting from a ROS bag the tool targets **g** for accelerometer
> and **deg/s** for gyroscope.  Use `--acc_unit` / `--gyro_unit` to tell the
> converter what unit is stored in the bag so it applies the right factor.
> If no unit flags are given, auto-detection runs but **no conversion** is
> applied — the informational message will tell you what was detected.

### Quick examples

```shell
# --- Check environment first (recommended before first use) ---------------

python mandeye_check_deps.py
# auto-install missing required packages:
python mandeye_check_deps.py --install-missing

# --- Audit ----------------------------------------------------------------

# Audit a bag (human report + machine-readable JSON):
python mandeye_bag_audit.py recording.bag --json audit.json

# --- Convert bag → MandEye ------------------------------------------------

# Explicit units (recommended):
python mandeye_bag_convert.py recording.bag output ros1-to-hdmapping \
    --acc_unit m/s2 --gyro_unit rad/s

# Auto topics + units suggested by audit JSON:
python mandeye_bag_convert.py recording.bag output ros1-to-hdmapping \
    --audit-json audit.json

# ROS2 bag → MandEye:
python mandeye_bag_convert.py recording_ros2/ output ros2-to-hdmapping \
    --acc_unit m/s2 --gyro_unit rad/s

# --- Convert MandEye → bag ------------------------------------------------

python mandeye_bag_convert.py ./my_dataset ./output.bag    hdmapping-to-ros1
python mandeye_bag_convert.py ./my_dataset ./output_ros2   hdmapping-to-ros2

# --- Multi-volume sequence ------------------------------------------------

python mandeye_bag_convert.py recording_0.bag output ros1-to-hdmapping --sequence

# --- Extract topics -------------------------------------------------------

python mandeye_bag_extract.py recording.bag -o out --topics "/livox/*" --format csv
python mandeye_bag_extract.py recording.bag --list

# --- Fix IMU units in already-extracted CSV files -------------------------

# Preview only (nothing written):
python mandeye_imu_rescale.py --dir ./output --acc-conv ms2g --gyro-conv rad2deg --dry-run

# Apply in-place with .bak backup:
python mandeye_imu_rescale.py --dir ./output --acc-conv ms2g --gyro-conv rad2deg --backup

# Convert timestamps from nanoseconds to seconds:
python mandeye_imu_rescale.py --dir ./output --time-conv ns2sec --backup

# Custom timestamp factor:
python mandeye_imu_rescale.py --dir ./output --time-factor 1e-9 --backup
```

Run any script with `--help` for the full list of options.

### IMU unit handling (bag → MandEye)

```
Priority order for unit selection:
  1. --acc_unit / --gyro_unit   explicit CLI arguments
  2. --audit-json               units detected by mandeye_bag_audit.py
  3. Auto-detection             samples first 500 IMU messages;
                                uses stationary-period analysis (|acc| ≈ 1 g)
                                when available.  Detection is informational only
                                — no conversion is applied automatically.
```

If the wrong units slipped through into the extracted CSVs, use
`mandeye_imu_rescale.py` to fix them without re-running the full extraction.
