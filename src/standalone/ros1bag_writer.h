#pragma once
// ros1bag_writer.h — Minimal standalone ROS 1 bag 2.0 writer
// No ROS installation required.
// Writes a valid, uncompressed single-chunk bag readable by rosbag play/info.
//
// Usage:
//   Ros1BagWriter w("out.bag");
//   int c0 = w.addConnection("/imu", "sensor_msgs/Imu", MD5, MSG_DEF);
//   w.write(c0, time_nsec, payload.data(), payload.size());
//   w.close();  // also called by destructor

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

class Ros1BagWriter {
public:
    // Open bag file and write the version line + placeholder bag header.
    explicit Ros1BagWriter(const std::string& path)
        : file_(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc)
    {
        if (!file_)
            throw std::runtime_error("Cannot open bag file: " + path);

        // Version line — exactly 13 bytes
        static const char VER[] = "#ROSBAG V2.0\n";
        file_.write(VER, 13);

        // Placeholder BAG_HEADER (padded to 4096 bytes, rewritten at close)
        bag_hdr_pos_ = static_cast<int64_t>(file_.tellp());
        write_bag_header(0, 0, 0);

        // Placeholder CHUNK record header (rewritten at close)
        chunk_pos_ = static_cast<int64_t>(file_.tellp());
        write_chunk_header(0);
        chunk_data_pos_ = static_cast<int64_t>(file_.tellp());
    }

    ~Ros1BagWriter() { if (!closed_) close(); }

    // Register a topic/type. Returns a connection id for use with write().
    int addConnection(const std::string& topic,
                      const std::string& type,
                      const std::string& md5,
                      const std::string& msg_def)
    {
        int cid = static_cast<int>(conns_.size());
        conns_.push_back({static_cast<uint32_t>(cid), topic, type, md5, msg_def});
        // Write CONNECTION record *inside* the chunk
        write_conn_record(conns_.back());
        return cid;
    }

    // Write a serialised message. time_ns = nanoseconds since Unix epoch.
    void write(int cid, uint64_t time_ns, const void* data, uint32_t data_len)
    {
        assert(cid >= 0 && cid < static_cast<int>(conns_.size()));
        assert(!closed_);

        if (time_ns < chunk_start_ns_) chunk_start_ns_ = time_ns;
        if (time_ns > chunk_end_ns_)   chunk_end_ns_   = time_ns;

        uint32_t sec  = static_cast<uint32_t>(time_ns / 1'000'000'000ULL);
        uint32_t nsec = static_cast<uint32_t>(time_ns % 1'000'000'000ULL);

        // Offset of this MSGDATA record within chunk DATA (after chunk header)
        uint64_t off = static_cast<uint64_t>(file_.tellp())
                     - static_cast<uint64_t>(chunk_data_pos_);
        idx_[static_cast<uint32_t>(cid)].push_back({sec, nsec,
                                                     static_cast<uint32_t>(off)});

        RHdr hdr;
        hdr.u8 ("op",   2);
        hdr.u32("conn", static_cast<uint32_t>(cid));
        hdr.tim("time", sec, nsec);
        write_record(hdr, data, data_len);
        ++n_msgs_;
    }

    void write(int cid, uint64_t time_ns, const std::vector<uint8_t>& v)
    {
        write(cid, time_ns, v.data(), static_cast<uint32_t>(v.size()));
    }

    // Finalise the bag: fix chunk header, write index, rewrite bag header.
    void close()
    {
        if (closed_) return;
        closed_ = true;

        if (chunk_start_ns_ == UINT64_MAX) chunk_start_ns_ = chunk_end_ns_ = 0;

        // --- Finalize chunk ---
        int64_t  chunk_end  = static_cast<int64_t>(file_.tellp());
        uint32_t chunk_data = static_cast<uint32_t>(chunk_end - chunk_data_pos_);
        file_.seekp(chunk_pos_);
        write_chunk_header(chunk_data);                        // overwrite with real size
        assert(static_cast<int64_t>(file_.tellp()) == chunk_data_pos_);
        file_.seekp(chunk_end);

        // --- Index section ---
        uint64_t index_pos = static_cast<uint64_t>(chunk_end);

        // INDEXDATA (op=4) — one per connection
        for (const auto& c : conns_) {
            const auto& entries = idx_[c.id];
            RHdr hdr;
            hdr.u8 ("op",    4);
            hdr.u32("ver",   1);
            hdr.u32("conn",  c.id);
            hdr.u32("count", static_cast<uint32_t>(entries.size()));

            std::vector<uint8_t> data;
            for (const auto& e : entries) {
                push32(data, e.sec);
                push32(data, e.nsec);
                push32(data, e.offset);   // uint32 offset within chunk data
            }
            write_record(hdr, data.data(), static_cast<uint32_t>(data.size()));
        }

        // CHUNK_HEADER (op=6) — one per chunk
        {
            uint32_t ss  = static_cast<uint32_t>(chunk_start_ns_ / 1'000'000'000ULL);
            uint32_t sns = static_cast<uint32_t>(chunk_start_ns_ % 1'000'000'000ULL);
            uint32_t es  = static_cast<uint32_t>(chunk_end_ns_   / 1'000'000'000ULL);
            uint32_t ens = static_cast<uint32_t>(chunk_end_ns_   % 1'000'000'000ULL);

            RHdr hdr;
            hdr.u8 ("op",         6);
            hdr.u32("ver",        1);
            hdr.u64("chunk_pos",  static_cast<uint64_t>(chunk_pos_));
            hdr.tim("start_time", ss,  sns);
            hdr.tim("end_time",   es,  ens);
            hdr.u32("count",      static_cast<uint32_t>(conns_.size()));

            std::vector<uint8_t> data;
            for (const auto& c : conns_) {
                push32(data, c.id);
                push32(data, static_cast<uint32_t>(idx_[c.id].size()));
            }
            write_record(hdr, data.data(), static_cast<uint32_t>(data.size()));
        }

        // CONNECTION records outside chunks — required by spec
        for (const auto& c : conns_) write_conn_record(c);

        // --- Rewrite BAG_HEADER with correct values ---
        file_.seekp(bag_hdr_pos_);
        write_bag_header(index_pos,
                         static_cast<uint32_t>(conns_.size()),
                         1);   // chunk_count = 1
        file_.flush();
        file_.close();
    }

private:
    // ---- Internal types ----

    struct ConnInfo {
        uint32_t    id;
        std::string topic, type, md5, msg_def;
    };

    struct IdxEntry {
        uint32_t sec, nsec, offset;   // offset within chunk DATA
    };

    // Lightweight header builder — encodes ROS bag record header fields.
    struct RHdr {
        std::vector<std::pair<std::string, std::vector<uint8_t>>> fields;

        void u8(const std::string& n, uint8_t v)  { fields.push_back({n, {v}}); }

        void u32(const std::string& n, uint32_t v) {
            uint8_t b[4]; std::memcpy(b, &v, 4);
            fields.push_back({n, {b[0],b[1],b[2],b[3]}});
        }
        void u64(const std::string& n, uint64_t v) {
            uint8_t b[8]; std::memcpy(b, &v, 8);
            fields.push_back({n, {b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]}});
        }
        void tim(const std::string& n, uint32_t s, uint32_t ns) {
            uint8_t b[8];
            std::memcpy(b,   &s,  4);
            std::memcpy(b+4, &ns, 4);
            fields.push_back({n, {b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]}});
        }
        void str(const std::string& n, const std::string& v) {
            fields.push_back({n, {v.begin(), v.end()}});
        }

        uint32_t ser_size() const {
            uint32_t sz = 0;
            for (const auto& [name, val] : fields)
                sz += 4u + static_cast<uint32_t>(name.size()) + 1u
                        + static_cast<uint32_t>(val.size());
            return sz;
        }

        std::vector<uint8_t> serialize() const {
            std::vector<uint8_t> out;
            out.reserve(ser_size());
            for (const auto& [name, val] : fields) {
                uint32_t flen = static_cast<uint32_t>(name.size()) + 1u
                              + static_cast<uint32_t>(val.size());
                uint8_t lb[4]; std::memcpy(lb, &flen, 4);
                out.insert(out.end(), lb, lb + 4);
                out.insert(out.end(), name.begin(), name.end());
                out.push_back('=');
                out.insert(out.end(), val.begin(), val.end());
            }
            return out;
        }
    };

    // ---- Member variables ----
    std::fstream file_;
    int64_t      bag_hdr_pos_{0};
    int64_t      chunk_pos_{0};
    int64_t      chunk_data_pos_{0};

    std::vector<ConnInfo>                     conns_;
    std::map<uint32_t, std::vector<IdxEntry>> idx_;

    uint64_t chunk_start_ns_{UINT64_MAX};
    uint64_t chunk_end_ns_{0};
    uint32_t n_msgs_{0};
    bool     closed_{false};

    // ---- Helpers ----

    void write_raw(const void* d, size_t n) {
        file_.write(static_cast<const char*>(d), static_cast<std::streamsize>(n));
    }

    void write_record(const RHdr& hdr, const void* data, uint32_t dlen) {
        auto   hb   = hdr.serialize();
        uint32_t hl = static_cast<uint32_t>(hb.size());
        write_raw(&hl,         4);
        write_raw(hb.data(),   hl);
        write_raw(&dlen,       4);
        if (dlen) write_raw(data, dlen);
    }

    // BAG_HEADER record padded to exactly 4096 bytes total.
    // h_len is deterministic (same field names/sizes), so padding is always the same.
    void write_bag_header(uint64_t index_pos, uint32_t nc, uint32_t nchunk) {
        RHdr hdr;
        hdr.u8 ("op",          3);
        hdr.u64("index_pos",   index_pos);
        hdr.u32("conn_count",  nc);
        hdr.u32("chunk_count", nchunk);

        auto     hb   = hdr.serialize();
        uint32_t hl   = static_cast<uint32_t>(hb.size());
        int64_t  pad  = 4096 - 4 - static_cast<int64_t>(hl) - 4;
        if (pad < 0) pad = 0;
        uint32_t dlen = static_cast<uint32_t>(pad);

        write_raw(&hl,  4);
        write_raw(hb.data(), hl);
        write_raw(&dlen, 4);
        std::vector<uint8_t> zeros(dlen, 0);
        if (dlen) write_raw(zeros.data(), dlen);
    }

    // CHUNK record header (op=5).  Fixed header size ensures placeholder and
    // final writes are byte-identical in length.
    void write_chunk_header(uint32_t data_size) {
        RHdr hdr;
        hdr.u8 ("op",          5);
        hdr.str("compression", "none");
        hdr.u32("size",        data_size);   // uncompressed == compressed for "none"

        auto     hb   = hdr.serialize();
        uint32_t hl   = static_cast<uint32_t>(hb.size());
        write_raw(&hl,         4);
        write_raw(hb.data(),   hl);
        write_raw(&data_size,  4);           // data_len == data_size
    }

    void write_conn_record(const ConnInfo& c) {
        RHdr hdr;
        hdr.u8 ("op",    7);
        hdr.u32("conn",  c.id);
        hdr.str("topic", c.topic);

        // Connection data = a second serialised header with type metadata
        RHdr meta;
        meta.str("type",               c.type);
        meta.str("md5sum",             c.md5);
        meta.str("message_definition", c.msg_def);
        meta.str("topic",              c.topic);
        meta.str("callerid",           "/mandeye_convert");
        meta.str("latching",           "0");

        auto data = meta.serialize();
        write_record(hdr, data.data(), static_cast<uint32_t>(data.size()));
    }

    static void push32(std::vector<uint8_t>& v, uint32_t x) {
        uint8_t b[4]; std::memcpy(b, &x, 4);
        v.insert(v.end(), b, b + 4);
    }
};
