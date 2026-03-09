// mandeye_convert.cpp — Standalone MandEye → ROS 1 bag converter for Windows
// No ROS installation required.
//
// Reads .laz / .las point clouds + .csv IMU files from <input_dir> and
// writes a ROS 1 bag file with:
//   sensor_msgs/PointCloud2  on  /livox/lidar   (one message per LAZ file)
//   sensor_msgs/Imu          on  /livox/imu
//
// Build using the adjacent CMakeLists.txt (see build_win_standalone.bat).
//
// Usage:
//   mandeye_convert <input_dir> <output.bag> [options]
//
// Options:
//   --pc_topic  <topic>     point-cloud topic  (default: /livox/lidar)
//   --imu_topic <topic>     IMU topic          (default: /livox/imu)
//   --imu_id    <N>         IMU id column      (default: 0)
//   --frame_id  <str>       TF frame id        (default: livox_frame)

#include "ros1bag_writer.h"
#include "common/LasLoader.h"
#include "common/ImuLoader.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ROS 1 message definitions (canonical text embedded in the bag metadata)
// ---------------------------------------------------------------------------

static const char IMU_MD5[]    = "b2b33aae8c0754b0e3db5c6bdb21bd6e";
static const char PC2_MD5[]    = "1158d486dd51d683ce2f1be655c3c181";

// Full sensor_msgs/Imu definition including sub-message definitions
static const char IMU_DEF[] =
"# This is a message to hold data from an IMU (Inertial Measurement Unit)\n"
"#\n"
"# Acceleration should be in m/s^2 (not in g's), and rotational velocity should be in rad/sec\n"
"#\n"
"Header header\n"
"\n"
"geometry_msgs/Quaternion orientation\n"
"float64[9] orientation_covariance\n"
"\n"
"geometry_msgs/Vector3 angular_velocity\n"
"float64[9] angular_velocity_covariance\n"
"\n"
"geometry_msgs/Vector3 linear_acceleration\n"
"float64[9] linear_acceleration_covariance\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"uint32 seq\n"
"time stamp\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"float64 x\n"
"float64 y\n"
"float64 z\n";

// Full sensor_msgs/PointCloud2 definition including sub-message definitions
static const char PC2_DEF[] =
"# This message holds a collection of N-dimensional points.\n"
"Header header\n"
"uint32 height\n"
"uint32 width\n"
"sensor_msgs/PointField[] fields\n"
"bool    is_bigendian\n"
"uint32  point_step\n"
"uint32  row_step\n"
"uint8[] data\n"
"bool is_dense\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"uint32 seq\n"
"time stamp\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/PointField\n"
"uint8 INT8    = 1\n"
"uint8 UINT8   = 2\n"
"uint8 INT16   = 3\n"
"uint8 UINT16  = 4\n"
"uint8 INT32   = 5\n"
"uint8 UINT32  = 6\n"
"uint8 FLOAT32 = 7\n"
"uint8 FLOAT64 = 8\n"
"string name\n"
"uint32 offset\n"
"uint8  datatype\n"
"uint32 count\n";

// ---------------------------------------------------------------------------
// Minimal binary serialisation helpers
// ---------------------------------------------------------------------------

struct BufWriter {
    std::vector<uint8_t> buf;

    void u8 (uint8_t  v) { buf.push_back(v); }
    void u32(uint32_t v) { auto s = buf.size(); buf.resize(s+4); std::memcpy(buf.data()+s, &v, 4); }
    void u64(uint64_t v) { auto s = buf.size(); buf.resize(s+8); std::memcpy(buf.data()+s, &v, 8); }
    void f32(float    v) { auto s = buf.size(); buf.resize(s+4); std::memcpy(buf.data()+s, &v, 4); }
    void f64(double   v) { auto s = buf.size(); buf.resize(s+8); std::memcpy(buf.data()+s, &v, 8); }
    // ROS1 string: 4-byte length + bytes (no null terminator)
    void str(const std::string& s) { u32(static_cast<uint32_t>(s.size())); buf.insert(buf.end(), s.begin(), s.end()); }
};

// Serialise sensor_msgs/Imu
// - angular_velocity: gyro  (rad/s as stored in the CSV)
// - linear_acceleration: accel (m/s^2 as stored in the CSV — no unit conversion)
static std::vector<uint8_t> make_imu(
    uint32_t seq,
    uint32_t sec, uint32_t nsec,
    const std::string& frame_id,
    float gx, float gy, float gz,       // angular velocity
    float ax, float ay, float az)       // linear acceleration
{
    BufWriter w;
    // std_msgs/Header
    w.u32(seq);
    w.u32(sec);
    w.u32(nsec);
    w.str(frame_id);

    // geometry_msgs/Quaternion orientation — identity (orientation unknown)
    w.f64(0.0); w.f64(0.0); w.f64(0.0); w.f64(1.0);
    // orientation_covariance — -1 on diagonal signals "unknown"
    for (int i = 0; i < 9; ++i) w.f64((i == 0 || i == 4 || i == 8) ? -1.0 : 0.0);

    // geometry_msgs/Vector3 angular_velocity
    w.f64(static_cast<double>(gx));
    w.f64(static_cast<double>(gy));
    w.f64(static_cast<double>(gz));
    // angular_velocity_covariance — all zeros (unknown)
    for (int i = 0; i < 9; ++i) w.f64(0.0);

    // geometry_msgs/Vector3 linear_acceleration
    w.f64(static_cast<double>(ax));
    w.f64(static_cast<double>(ay));
    w.f64(static_cast<double>(az));
    // linear_acceleration_covariance — all zeros (unknown)
    for (int i = 0; i < 9; ++i) w.f64(0.0);

    return w.buf;
}

// Serialise sensor_msgs/PointCloud2
// Point layout (24 bytes/point):  x(f32), y(f32), z(f32), intensity(f32), timestamp(f64)
static std::vector<uint8_t> make_pc2(
    uint32_t seq,
    uint32_t sec, uint32_t nsec,
    const std::string& frame_id,
    const std::vector<mandeye::Point>& pts)
{
    static const uint32_t POINT_STEP = 24; // 4+4+4+4+8
    uint32_t N = static_cast<uint32_t>(pts.size());

    BufWriter w;
    // std_msgs/Header
    w.u32(seq);
    w.u32(sec);
    w.u32(nsec);
    w.str(frame_id);

    w.u32(1);   // height (unorganised cloud)
    w.u32(N);   // width

    // sensor_msgs/PointField[] fields  (dynamic array: 4-byte count + elements)
    w.u32(5);   // 5 fields: x, y, z, intensity, timestamp
    // x — FLOAT32 = 7
    w.str("x");         w.u32(0);  w.u8(7); w.u32(1);
    // y — FLOAT32 = 7
    w.str("y");         w.u32(4);  w.u8(7); w.u32(1);
    // z — FLOAT32 = 7
    w.str("z");         w.u32(8);  w.u8(7); w.u32(1);
    // intensity — FLOAT32 = 7
    w.str("intensity"); w.u32(12); w.u8(7); w.u32(1);
    // timestamp — FLOAT64 = 8
    w.str("timestamp"); w.u32(16); w.u8(8); w.u32(1);

    w.u8(0);              // is_bigendian = false
    w.u32(POINT_STEP);    // point_step
    w.u32(POINT_STEP * N);// row_step

    // uint8[] data
    w.u32(POINT_STEP * N);
    for (const auto& p : pts) {
        w.f32(static_cast<float>(p.point.x()));
        w.f32(static_cast<float>(p.point.y()));
        w.f32(static_cast<float>(p.point.z()));
        w.f32(p.intensity);
        w.f64(p.timestamp);
    }

    w.u8(1);   // is_dense = true

    return w.buf;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " <input_dir> <output.bag> [options]\n"
        << "\n"
        << "Converts a MandEye/HDMapping dataset (LAZ + CSV) to a ROS 1 bag.\n"
        << "No ROS installation required.\n"
        << "\n"
        << "Options:\n"
        << "  --pc_topic  <topic>   point-cloud topic  (default: /livox/lidar)\n"
        << "  --imu_topic <topic>   IMU topic          (default: /livox/imu)\n"
        << "  --imu_id    <N>       IMU id to extract  (default: 0)\n"
        << "  --frame_id  <str>     TF frame id        (default: livox_frame)\n"
        << "\n"
        << "Output:\n"
        << "  sensor_msgs/PointCloud2 on <pc_topic>  (one msg per .laz file)\n"
        << "  sensor_msgs/Imu         on <imu_topic> (one msg per CSV row)\n";
}

static double ts_from_double(double ts, uint32_t& sec, uint32_t& nsec) {
    sec  = static_cast<uint32_t>(ts);
    nsec = static_cast<uint32_t>((ts - static_cast<double>(sec)) * 1e9);
    return ts;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string input_dir  = argv[1];
    std::string output_bag = argv[2];
    std::string pc_topic   = "/livox/lidar";
    std::string imu_topic  = "/livox/imu";
    std::string frame_id   = "livox_frame";
    int         imu_id     = 0;

    static const auto has_value_opt = [](const std::string& a) {
        return a == "--pc_topic" || a == "--imu_topic" ||
               a == "--frame_id" || a == "--imu_id";
    };

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (has_value_opt(arg)) {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: option '" << arg << "' requires a value\n";
                print_usage(argv[0]);
                return 1;
            }
            std::string val = argv[++i];
            if (val.empty()) {
                std::cerr << "ERROR: option '" << arg << "' value cannot be empty\n";
                return 1;
            }
            if (arg == "--pc_topic")       pc_topic  = val;
            else if (arg == "--imu_topic") imu_topic = val;
            else if (arg == "--frame_id")  frame_id  = val;
            else if (arg == "--imu_id") {
                try {
                    imu_id = std::stoi(val);
                    if (imu_id < 0) {
                        std::cerr << "ERROR: --imu_id must be >= 0, got: " << val << "\n";
                        return 1;
                    }
                } catch (const std::exception&) {
                    std::cerr << "ERROR: --imu_id expects an integer, got: '" << val << "'\n";
                    return 1;
                }
            }
        } else {
            std::cerr << "ERROR: unknown option: '" << arg << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate topic names
    if (pc_topic.empty() || pc_topic[0] != '/') {
        std::cerr << "ERROR: --pc_topic must start with '/', got: '" << pc_topic << "'\n";
        return 1;
    }
    if (imu_topic.empty() || imu_topic[0] != '/') {
        std::cerr << "ERROR: --imu_topic must start with '/', got: '" << imu_topic << "'\n";
        return 1;
    }

    // Scan input directory for .laz/.las and .csv files
    std::vector<std::string> laz_files, csv_files;
    try {
        for (const auto& e : std::filesystem::directory_iterator(input_dir)) {
            auto ext = e.path().extension().string();
            // case-insensitive compare for extension
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".laz" || ext == ".las")
                laz_files.push_back(e.path().string());
            else if (ext == ".csv")
                csv_files.push_back(e.path().string());
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR reading directory '" << input_dir << "': " << ex.what() << "\n";
        return 1;
    }

    std::sort(laz_files.begin(), laz_files.end());
    std::sort(csv_files.begin(), csv_files.end());

    if (laz_files.empty() && csv_files.empty()) {
        std::cerr << "ERROR: no .laz or .csv files found in: " << input_dir << "\n";
        return 1;
    }

    if (laz_files.empty())
        std::cerr << "WARNING: no .laz/.las files found — bag will contain IMU data only\n";
    if (csv_files.empty())
        std::cerr << "WARNING: no .csv files found — bag will contain point clouds only\n";

    std::cout << "Input  : " << input_dir  << "\n"
              << "Output : " << output_bag << "\n"
              << "  LAZ/LAS files : " << laz_files.size() << "\n"
              << "  CSV files     : " << csv_files.size() << "\n"
              << "  PC topic      : " << pc_topic  << "\n"
              << "  IMU topic     : " << imu_topic << "\n"
              << "  IMU id        : " << imu_id    << "\n"
              << "  Frame id      : " << frame_id  << "\n";

    // Open bag and register connections
    Ros1BagWriter bag = [&]() -> Ros1BagWriter {
        try {
            return Ros1BagWriter(output_bag);
        } catch (const std::exception& ex) {
            std::cerr << "ERROR: cannot create output bag '" << output_bag << "': " << ex.what() << "\n";
            std::exit(1);
        }
    }();
    int imu_conn = bag.addConnection(imu_topic, "sensor_msgs/Imu",
                                     IMU_MD5, IMU_DEF);
    int pc_conn  = bag.addConnection(pc_topic,  "sensor_msgs/PointCloud2",
                                     PC2_MD5, PC2_DEF);

    // ---- Write IMU ----
    uint32_t imu_seq = 0;
    for (const auto& fn : csv_files) {
        std::cout << "  [IMU] " << fn << "\n";
        std::vector<std::tuple<double, mandeye::ImuAngularVelocity,
                               mandeye::ImuAcceleration>> data;
        try {
            data = mandeye::load_imu(fn, imu_id);
        } catch (const std::exception& ex) {
            std::cerr << "    WARNING: " << ex.what() << " — skipping\n";
            continue;
        }

        for (const auto& [ts, ang, acc] : data) {
            if (ts == 0.0) continue;
            uint32_t sec, nsec;
            ts_from_double(ts, sec, nsec);
            uint64_t ts_ns = static_cast<uint64_t>(ts * 1e9);

            auto msg = make_imu(imu_seq++, sec, nsec, "livox",
                                ang[0], ang[1], ang[2],
                                acc[0], acc[1], acc[2]);
            bag.write(imu_conn, ts_ns, msg);
        }
    }

    // ---- Write point clouds ----
    uint32_t pc_seq  = 0;
    uint64_t total_pts = 0;
    for (const auto& fn : laz_files) {
        std::cout << "  [LAZ] " << fn << "\n";
        std::vector<mandeye::Point> pts;
        try {
            pts = mandeye::load(fn);
        } catch (const std::exception& ex) {
            std::cerr << "    WARNING: " << ex.what() << " — skipping\n";
            continue;
        }
        if (pts.empty()) continue;

        // Use timestamp of first valid point as the message header stamp
        double ts = 0.0;
        for (const auto& p : pts) { if (p.timestamp != 0.0) { ts = p.timestamp; break; } }

        uint32_t sec, nsec;
        ts_from_double(ts, sec, nsec);
        uint64_t ts_ns = static_cast<uint64_t>(ts * 1e9);

        auto msg = make_pc2(pc_seq++, sec, nsec, frame_id, pts);
        bag.write(pc_conn, ts_ns, msg);
        total_pts += pts.size();
    }

    bag.close();

    std::cout << "\nDone.\n"
              << "  IMU messages : " << imu_seq  << "\n"
              << "  PC2 messages : " << pc_seq   << "\n"
              << "  Total points : " << total_pts << "\n"
              << "  Bag written  : " << output_bag << "\n";
    return 0;
}
