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
|--------|---------|
| `mandeye_bag_audit.py` | Audit a ROS bag: score every (PointCloud, IMU) pair, detect units, export JSON |
| `mandeye_bag_convert.py` | Convert between MandEye datasets and ROS1/ROS2 bag files |
| `mandeye_bag_extract.py` | Extract selected topics to a filtered bag and/or CSV/LAZ/image files |
| `mandeye_bag_common.py` | Shared library used by the three scripts above |

### Quick examples

```shell
# Audit a bag (human report + machine-readable JSON):
python mandeye_bag_audit.py recording.bag --json audit.json

# Convert ROS1 bag → MandEye (auto topics + units from audit):
python mandeye_bag_convert.py recording.bag output ros1-to-hdmapping --audit-json audit.json

# Convert MandEye → ROS1 bag:
python mandeye_bag_convert.py ./my_dataset ./output.bag hdmapping-to-ros1

# Convert MandEye → ROS2 bag:
python mandeye_bag_convert.py ./my_dataset ./output_ros2 hdmapping-to-ros2

# Extract specific topics to LAZ / CSV / images:
python mandeye_bag_extract.py recording.bag -o out --topics "/livox/*" --format csv

# List topics in a bag:
python mandeye_bag_extract.py recording.bag --list

# Multi-volume sequence:
python mandeye_bag_convert.py recording_0.bag output ros1-to-hdmapping --sequence
```

Run any script with `--help` for the full list of options.
