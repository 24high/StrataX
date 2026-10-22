// src/core/routing_stats.cpp - routing statistics for the expert whitelist generator (routing_stats.hpp).
#include "strata/core/routing_stats.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <set>
#include <unordered_set>

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace fs = std::filesystem;

namespace {

constexpr char kLayerMagic[8] = {'S', 'X', 'R', 'S', 'T', 'A', 'T', '1'};
constexpr char kCalibMagic[8] = {'S', 'X', 'C', 'A', 'L', 'I', 'B', '1'};
constexpr char kCkptMagic[8] = {'S', 'X', 'R', 'S', 'C', 'K', 'P', '1'};
constexpr size_t kLayerHeader = 64;
constexpr const char* kCheckpoint = "checkpoint.bin";
constexpr const char* kMeta = "meta.json";

bool little_endian() {
    const uint16_t one = 1;
    uint8_t b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i)));
}
uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (8 * i);
    return v;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (8 * i);
    return v;
}

/// u64 arrays to and from the files' little-endian bytes.
bool write_u64s(AtomicFile& f, const uint64_t* v, size_t n) {
    if (little_endian()) return f.write(v, n * 8);
    std::vector<uint8_t> b;
    b.reserve(n * 8);
    for (size_t i = 0; i < n; ++i) put_u64(b, v[i]);
    return f.write(b.data(), b.size());
}
bool read_exact(std::FILE* f, void* dst, size_t n) { return n == 0 || std::fread(dst, 1, n, f) == n; }
bool read_u64s(std::FILE* f, uint64_t* v, size_t n) {
    if (!read_exact(f, v, n * 8)) return false;
    if (!little_endian())
        for (size_t i = 0; i < n; ++i) v[i] = get_u64((const uint8_t*) &v[i]);
    return true;
}

std::string crc_hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "%08x", (unsigned) v);
    return b;
}

std::string hex64(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "fnv1a64:%016llx", (unsigned long long) v);
    return b;
}

struct FileCloser {
    std::FILE* f;
    ~FileCloser() { if (f) std::fclose(f); }
};

}  // namespace

// ================================ SMALL HELPERS ================================

std::string json_escape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    o += b;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
}

std::string utc_date_today() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char b[16];
    std::strftime(b, sizeof b, "%Y-%m-%d", &tm);
    return b;
}

bool AtomicFile::open(const std::string& path, std::string& err) {
    abandon();
    path_ = path;
    tmp_ = path + ".tmp";
    f_ = std::fopen(tmp_.c_str(), "wb");
    failed_ = false;
    written_ = 0;
    if (f_ == nullptr) {
        err = "cannot write " + tmp_ + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

bool AtomicFile::write(const void* p, size_t n) {
    if (f_ == nullptr || failed_) return false;
    if (n > 0 && std::fwrite(p, 1, n, f_) != n) failed_ = true;
    written_ += n;
    return !failed_;
}

bool AtomicFile::commit(std::string& err) {
    if (f_ == nullptr) {
        err = "internal: commit without open";
        return false;
    }
    bool ok = !failed_ && std::fflush(f_) == 0;
#if defined(_WIN32)
    ok = ok && _commit(_fileno(f_)) == 0;
#else
    ok = ok && fsync(fileno(f_)) == 0;
#endif
    ok = std::fclose(f_) == 0 && ok;
    f_ = nullptr;
    if (!ok) {
        err = "writing " + tmp_ + " failed (" + std::strerror(errno) + "): is the drive full?";
        std::error_code ec;
        fs::remove(tmp_, ec);
        return false;
    }
    std::error_code ec;
    fs::rename(tmp_, path_, ec);
    if (ec) {
        err = "cannot move " + tmp_ + " to " + path_ + ": " + ec.message();
        fs::remove(tmp_, ec);
        return false;
    }
#if !defined(_WIN32)
    // the rename itself reaches the drive only with its folder: a power loss must not undo it after the next write
    const std::string folder = fs::path(path_).has_parent_path() ? fs::path(path_).parent_path().string() : ".";
    if (const int d = ::open(folder.c_str(), O_RDONLY); d >= 0) {
        (void) ::fsync(d);
        ::close(d);
    }
#endif
    return true;
}

void AtomicFile::abandon() {
    if (f_ != nullptr) {
        std::fclose(f_);
        f_ = nullptr;
        std::error_code ec;
        fs::remove(tmp_, ec);
    }
}

bool atomic_write_file(const std::string& path, const void* bytes, size_t n, std::string& err) {
    AtomicFile f;
    return f.open(path, err) && f.write(bytes, n) && f.commit(err);
}

uint32_t crc32_update(uint32_t crc, const void* data, size_t n) {
    static const std::vector<uint32_t> table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = c & 1u ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    const auto* p = static_cast<const uint8_t*>(data);
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ p[i]) & 0xffu] ^ (crc >> 8);
    return ~crc;
}

bool file_crc32(const std::string& path, uint32_t& crc, uint64_t& bytes, std::string& err) {
    crc = 0;
    bytes = 0;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    FileCloser fc{f};
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        crc = crc32_update(crc, buf.data(), n);
        bytes += n;
    }
    return true;
}

bool MemoryMeter::take(uint64_t bytes, const char* what, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    if (budget_ > 0 && used_ + bytes > budget_) {
        char b[512];
        std::snprintf(b, sizeof b, "the %s budget of %.1f MiB does not hold %s (%.1f MiB more on top of %.1f MiB in use)",
                      name_, (double) budget_ / 1048576.0, what, (double) bytes / 1048576.0, (double) used_ / 1048576.0);
        err = b;
        return false;
    }
    used_ += bytes;
    peak_ = std::max(peak_, used_);
    return true;
}

void MemoryMeter::give(uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    used_ = bytes > used_ ? 0 : used_ - bytes;
}

uint64_t MemoryMeter::used() const {
    std::lock_guard<std::mutex> lk(mu_);
    return used_;
}

uint64_t MemoryMeter::peak() const {
    std::lock_guard<std::mutex> lk(mu_);
    return peak_;
}

// ================================ CATEGORIES AND SAMPLES ================================

bool RoutingCategory::from_key(const std::string& key, RoutingCategory& out, std::string& err) {
    std::vector<std::string> f(1);
    for (char c : key) {
        if (c == '|') f.emplace_back();
        else f.back() += c;
    }
    if (f.size() != 5) {
        err = "category \"" + key + "\" is not dataset|language|task|region|origin";
        return false;
    }
    out.key = key;
    out.dataset = f[0];
    out.language = f[1];
    out.task = f[2];
    out.region = f[3];
    out.origin = f[4];
    return true;
}

int64_t CalibSet::tokens() const {
    int64_t n = 0;
    for (const CalibSample& s : samples) n += (int64_t) s.tokens.size();
    return n;
}

int64_t CalibSet::longest() const {
    int64_t n = 0;
    for (const CalibSample& s : samples) n = std::max(n, (int64_t) s.tokens.size());
    return n;
}

bool read_calib_samples(const std::string& path, CalibSet& out, std::string& err) {
    out = CalibSet{};
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read the prepared samples " + path + " (run `whitelist_gen.py prepare` first)";
        return false;
    }
    FileCloser fc{f};
    std::vector<uint8_t> all;
    {
        std::vector<uint8_t> buf(1 << 20);
        size_t n;
        while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) all.insert(all.end(), buf.begin(), buf.begin() + (long) n);
    }
    out.fnv = whitelist_fnv1a64(all.data(), all.size());
    size_t p = 0;
    auto need = [&](size_t n) {
        if (p + n > all.size()) {
            err = path + ": the file ends early (truncated or not a prepared sample file)";
            return false;
        }
        return true;
    };
    if (!need(24) || std::memcmp(all.data(), kCalibMagic, 8) != 0) {
        err = path + ": not a prepared sample file (whitelist_gen.py prepare writes samples.bin)";
        return false;
    }
    const uint32_t ver = get_u32(&all[8]), n_cat = get_u32(&all[12]), n_s = get_u32(&all[16]);
    p = 24;
    if (ver != kCalibSamplesVersion) {
        err = path + ": sample file version " + std::to_string(ver) + " (this engine reads " +
              std::to_string(kCalibSamplesVersion) + ")";
        return false;
    }
    if (n_cat == 0 || n_cat > 65535) {
        err = path + ": " + std::to_string(n_cat) + " categories (1..65535 allowed)";
        return false;
    }
    std::set<std::string> keys;
    for (uint32_t c = 0; c < n_cat; ++c) {
        if (!need(4)) return false;
        const uint32_t len = get_u32(&all[p]);
        p += 4;
        if (!need(len)) return false;
        RoutingCategory rc;
        if (!RoutingCategory::from_key(std::string((const char*) &all[p], len), rc, err)) {
            err = path + ": " + err;
            return false;
        }
        p += len;
        if (!keys.insert(rc.key).second) {
            err = path + ": category \"" + rc.key + "\" is listed twice";
            return false;
        }
        out.categories.push_back(std::move(rc));
    }
    std::unordered_set<std::string> ids;
    out.samples.reserve(n_s);
    for (uint32_t i = 0; i < n_s; ++i) {
        CalibSample s;
        if (!need(4)) return false;
        const uint32_t len = get_u32(&all[p]);
        p += 4;
        if (!need(len)) return false;
        s.id.assign((const char*) &all[p], len);
        p += len;
        if (!need(4)) return false;
        const uint32_t n = get_u32(&all[p]);
        p += 4;
        if (n == 0) {
            err = path + ": sample \"" + s.id + "\" has no tokens";
            return false;
        }
        if (!need((size_t) n * 6)) return false;
        s.tokens.resize(n);
        s.cat.resize(n);
        for (uint32_t t = 0; t < n; ++t) s.tokens[t] = (int32_t) get_u32(&all[p + (size_t) t * 4]);
        p += (size_t) n * 4;
        for (uint32_t t = 0; t < n; ++t) {
            s.cat[t] = (uint16_t) (all[p + (size_t) t * 2] | (all[p + (size_t) t * 2 + 1] << 8));
            if (s.cat[t] >= n_cat) {
                err = path + ": sample \"" + s.id + "\" names category " + std::to_string(s.cat[t]) + " of " +
                      std::to_string(n_cat);
                return false;
            }
        }
        p += (size_t) n * 2;
        if (!ids.insert(s.id).second) {
            err = path + ": sample id \"" + s.id + "\" is used twice";
            return false;
        }
        out.samples.push_back(std::move(s));
    }
    if (p != all.size()) {
        err = path + ": " + std::to_string(all.size() - p) + " bytes after the last sample";
        return false;
    }
    return true;
}

bool write_calib_samples(const std::string& path, const CalibSet& set, std::string& err) {
    std::vector<uint8_t> b(kCalibMagic, kCalibMagic + 8);
    put_u32(b, kCalibSamplesVersion);
    put_u32(b, (uint32_t) set.categories.size());
    put_u32(b, (uint32_t) set.samples.size());
    put_u32(b, 0);
    for (const RoutingCategory& c : set.categories) {
        put_u32(b, (uint32_t) c.key.size());
        b.insert(b.end(), c.key.begin(), c.key.end());
    }
    for (const CalibSample& s : set.samples) {
        put_u32(b, (uint32_t) s.id.size());
        b.insert(b.end(), s.id.begin(), s.id.end());
        put_u32(b, (uint32_t) s.tokens.size());
        for (int32_t t : s.tokens) put_u32(b, (uint32_t) t);
        for (uint16_t c : s.cat) {
            b.push_back((uint8_t) c);
            b.push_back((uint8_t) (c >> 8));
        }
    }
    return atomic_write_file(path, b.data(), b.size(), err);
}

// ================================ THE SCHEMA ================================

uint64_t RoutingStatsSchema::schema_fnv() const {
    std::string t;
    for (const RoutingCategory& c : categories) t += c.key + "\n";
    return whitelist_fnv1a64(t.data(), t.size());
}

uint64_t RoutingStatsSchema::fingerprint_fnv() const { return whitelist_fnv1a64(fingerprint.data(), fingerprint.size()); }

std::vector<int64_t> RoutingStatsSchema::moe_layers() const {
    std::vector<int64_t> out;
    for (int64_t l = 0; l < model.n_layers; ++l)
        if (model.experts_per_layer[(size_t) l] > 0) out.push_back(l);
    return out;
}

// ================================ ACCUMULATORS ================================

uint64_t routing_fixed(double v, int bits) {
    if (!(v > 0.0) || !std::isfinite(v)) return 0;
    const double s = std::floor(std::ldexp(v, bits) + 0.5);
    if (s >= 18446744073709551615.0) return ~0ull;
    return (uint64_t) s;
}

float routing_contrib(float w, const float* y, int64_t n) {
    double ss = 0.0;
    for (int64_t i = 0; i < n; ++i) ss += (double) y[i] * (double) y[i];
    return (float) ((double) w * std::sqrt(ss));
}

void LayerStats::reset(int32_t n_expert, int32_t top_k, int32_t n_cat) {
    n_expert_ = n_expert;
    top_k_ = top_k;
    n_cat_ = n_cat;
    tokens_.assign((size_t) n_cat, 0);
    v_.assign((size_t) n_cat * (size_t) n_expert * (size_t) (4 + top_k), 0);
    blocks = 0;
}

void LayerStats::add_pick(int32_t c, int32_t e, int32_t rank, float w, float contrib) {
    if (c < 0 || c >= n_cat_ || e < 0 || e >= n_expert_ || rank < 0 || rank >= top_k_) return;
    uint64_t* r = &v_[((size_t) c * n_expert_ + (size_t) e) * (size_t) width()];
    const double g = (double) w;
    r[kCount] += 1;
    r[kGate] += routing_fixed(g, kGateFracBits);
    r[kGateSq] += routing_fixed(g * g, kGateFracBits);
    r[kContrib] += routing_fixed((double) contrib, kContribFracBits);
    r[kRank + rank] += 1;
}

void LayerStats::add_token(int32_t c, const int32_t* ids, const float* w, const float* contrib) {
    if (c < 0 || c >= n_cat_) return;
    tokens_[(size_t) c] += 1;
    for (int32_t j = 0; j < top_k_; ++j)
        if (ids[j] >= 0) add_pick(c, ids[j], j, w[j], contrib != nullptr ? contrib[j] : 0.0f);
}

void LayerStats::merge(const LayerStats& o) {
    for (size_t i = 0; i < tokens_.size() && i < o.tokens_.size(); ++i) tokens_[i] += o.tokens_[i];
    for (size_t i = 0; i < v_.size() && i < o.v_.size(); ++i) v_[i] += o.v_[i];
}

// ================================ LAYER FILES ================================

std::string layer_stats_name(int64_t layer) {
    char b[32];
    std::snprintf(b, sizeof b, "layer_%03lld.bin", (long long) layer);
    return b;
}

bool write_layer_stats(const std::string& dir, int64_t layer, const LayerStats& s, const RoutingStatsSchema& schema,
                       std::string& err) {
    std::vector<uint8_t> h(kLayerMagic, kLayerMagic + 8);
    put_u32(h, kRoutingStatsVersion);
    put_u32(h, (uint32_t) layer);
    put_u32(h, (uint32_t) s.n_expert());
    put_u32(h, (uint32_t) s.top_k());
    put_u32(h, (uint32_t) s.n_cat());
    put_u32(h, schema.has_contrib ? 1u : 0u);
    put_u32(h, (uint32_t) kGateFracBits);
    put_u32(h, (uint32_t) kContribFracBits);
    put_u64(h, s.blocks);
    put_u64(h, schema.fingerprint_fnv());
    put_u64(h, schema.schema_fnv());
    AtomicFile f;
    if (!f.open((fs::path(dir) / layer_stats_name(layer)).string(), err)) return false;
    if (!f.write(h.data(), h.size()) || !write_u64s(f, s.token_array().data(), s.token_array().size()) ||
        !write_u64s(f, s.value_array().data(), s.value_array().size())) {
        err = "writing the statistics of layer " + std::to_string(layer) + " failed: is the drive full?";
        return false;
    }
    return f.commit(err);
}

bool read_layer_stats(const std::string& dir, int64_t layer, const RoutingStatsSchema& schema, LayerStats& out,
                      std::string& err) {
    const std::string path = (fs::path(dir) / layer_stats_name(layer)).string();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    FileCloser fc{f};
    uint8_t h[kLayerHeader];
    if (!read_exact(f, h, sizeof h) || std::memcmp(h, kLayerMagic, 8) != 0) {
        err = path + ": not a routing statistics layer file";
        return false;
    }
    const int32_t ne = layer >= 0 && layer < schema.model.n_layers ? schema.model.experts_per_layer[(size_t) layer] : -1;
    if (get_u32(h + 8) != kRoutingStatsVersion) {
        err = path + ": format version " + std::to_string(get_u32(h + 8)) + " (this engine reads " +
              std::to_string(kRoutingStatsVersion) + ")";
        return false;
    }
    if ((int64_t) get_u32(h + 12) != layer || (int32_t) get_u32(h + 16) != ne || (int64_t) get_u32(h + 20) != schema.model.top_k ||
        get_u32(h + 24) != (uint32_t) schema.categories.size()) {
        err = path + ": the layer, expert count, top-k or category count does not match this run";
        return false;
    }
    if (get_u32(h + 32) != (uint32_t) kGateFracBits || get_u32(h + 36) != (uint32_t) kContribFracBits) {
        err = path + ": another fixed-point scale";
        return false;
    }
    if (get_u64(h + 48) != schema.fingerprint_fnv()) {
        err = path + ": the statistics were collected on another model or pack (the model fingerprint differs)";
        return false;
    }
    if (get_u64(h + 56) != schema.schema_fnv()) {
        err = path + ": the statistics have other categories (another prepared sample file)";
        return false;
    }
    out.reset(ne, (int32_t) schema.model.top_k, (int32_t) schema.categories.size());
    out.blocks = get_u64(h + 40);
    if (!read_u64s(f, out.token_array().data(), out.token_array().size()) ||
        !read_u64s(f, out.value_array().data(), out.value_array().size())) {
        err = path + ": the file ends early";
        return false;
    }
    uint8_t extra;
    if (std::fread(&extra, 1, 1, f) != 0) {
        err = path + ": bytes after the statistics";
        return false;
    }
    return true;
}

// ================================ META.JSON ================================

namespace {

template <class T>
std::string json_list(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

}  // namespace

bool write_stats_meta(const std::string& dir, const RoutingStatsSchema& schema, const RoutingStatsRun& run,
                      const std::vector<std::string>& samples_done, uint64_t tokens, std::string& err) {
    const std::vector<int64_t> moe = schema.moe_layers();
    std::vector<uint32_t> crc(moe.size());
    std::vector<uint64_t> size(moe.size());
    for (size_t i = 0; i < moe.size(); ++i)
        if (!file_crc32((fs::path(dir) / layer_stats_name(moe[i])).string(), crc[i], size[i], err)) return false;
    const std::string created = run.created.empty() ? utc_date_today() : run.created;
    // the run's id: what it counted (the model, the categories, the samples, the parameters) - two runs over the same
    // samples with the same parameters have the same id, and merge refuses to add them twice
    std::string idt = schema.fingerprint + "\n" + hex64(schema.schema_fnv()) + "\n" + run.mode + "\n" + run.collect_json +
                      "\n" + run.run_params_json + "\n";
    if (schema.similarity.on()) idt += "similarity " + hex64(schema.similarity.fnv()) + "\n";
    for (const std::string& s : samples_done) idt += s + "\n";
    // the similarity files (when collected): their checksums, and the sampled tokens (every layer counts the same ones)
    std::string sim_json;
    if (schema.similarity.on()) {
        uint64_t sim_tokens = 0;
        std::string files;
        for (size_t i = 0; i < moe.size(); ++i) {
            LayerSimilarity s;
            if (!read_layer_similarity(dir, moe[i], schema, s, err)) return false;
            uint32_t c = 0;
            uint64_t b = 0;
            if (!file_crc32((fs::path(dir) / similarity_file_name(moe[i])).string(), c, b, err)) return false;
            if (i == 0) sim_tokens = s.tokens;
            files += std::string(i ? ",\n      " : "\n      ") + "{\"layer\": " + std::to_string(moe[i]) + ", \"file\": \"" +
                     similarity_file_name(moe[i]) + "\", \"bytes\": " + std::to_string(b) + ", \"crc32\": \"" + crc_hex(c) +
                     "\"}";
        }
        char frac[40];
        std::snprintf(frac, sizeof frac, "%.17g", schema.similarity.fraction);
        sim_json = "  \"similarity\": {\"fraction\": " + std::string(frac) + ", \"seed\": " +
                   std::to_string(schema.similarity.seed) + ", \"config_fnv1a64\": \"" + hex64(schema.similarity.fnv()) +
                   "\", \"frac_bits\": " + std::to_string(kSimilarityFracBits) + ", \"rel_cap\": " +
                   std::to_string((int) kSimilarityRelCap) + ",\n    \"weighting\": \"the final router weight of the picked "
                   "expert\", \"tokens\": " + std::to_string(sim_tokens) + ",\n    \"files\": [" + files + "\n    ]},\n";
    }
    const std::string run_id = hex64(whitelist_fnv1a64(idt.data(), idt.size()));
    std::string j = "{\n";
    j += "  \"format_version\": 1,\n  \"kind\": \"strata-routing-stats\",\n";
    j += "  \"model_fingerprint\": \"" + json_escape(schema.fingerprint) + "\",\n";
    j += "  \"model\": {\"num_layers\": " + std::to_string(schema.model.n_layers) +
         ", \"experts_per_layer\": " + json_list(schema.model.experts_per_layer) +
         ", \"top_k\": " + std::to_string(schema.model.top_k) + ", \"hidden\": " + std::to_string(schema.model.hidden) +
         ",\n            \"layer_offset\": " + json_list(schema.model.layer_offset) +
         ",\n            \"blob_bytes\": " + json_list(schema.model.blob_bytes) +
         ",\n            \"moe_layers\": " + json_list(moe) + "},\n";
    j += "  \"dense_bytes\": " + std::to_string(schema.dense_bytes) + ",\n";
    j += "  \"quantization\": \"" + json_escape(schema.quantization) + "\",\n";
    j += "  \"fixed_point\": {\"gate_frac_bits\": " + std::to_string(kGateFracBits) +
         ", \"contrib_frac_bits\": " + std::to_string(kContribFracBits) + "},\n";
    j += std::string("  \"has_contrib\": ") + (schema.has_contrib ? "true" : "false") + ",\n";
    j += "  \"categories\": [";
    for (size_t c = 0; c < schema.categories.size(); ++c) {
        const RoutingCategory& k = schema.categories[c];
        j += (c ? ",\n    " : "\n    ");
        j += "{\"key\": \"" + json_escape(k.key) + "\", \"dataset\": \"" + json_escape(k.dataset) + "\", \"language\": \"" +
             json_escape(k.language) + "\", \"task\": \"" + json_escape(k.task) + "\", \"region\": \"" +
             json_escape(k.region) + "\", \"origin\": \"" + json_escape(k.origin) + "\"}";
    }
    j += "\n  ],\n";
    j += "  \"schema_fnv1a64\": \"" + hex64(schema.schema_fnv()) + "\",\n";
    j += "  \"layer_files\": [";
    for (size_t i = 0; i < moe.size(); ++i)
        j += std::string(i ? ",\n    " : "\n    ") + "{\"layer\": " + std::to_string(moe[i]) + ", \"file\": \"" +
             layer_stats_name(moe[i]) + "\", \"bytes\": " + std::to_string(size[i]) + ", \"crc32\": \"" + crc_hex(crc[i]) + "\"}";
    j += "\n  ],\n";
    j += sim_json;
    j += "  \"tokens\": " + std::to_string(tokens) + ",\n";
    j += "  \"samples\": " + std::to_string(samples_done.size()) + ",\n";
    j += "  \"run_id\": \"" + run_id + "\",\n";
    j += "  \"runs\": [{\"run_id\": \"" + run_id + "\", \"mode\": \"" + json_escape(run.mode) + "\", \"created\": \"" +
         json_escape(created) + "\", \"samples\": " + std::to_string(samples_done.size()) + ", \"tokens\": " +
         std::to_string(tokens) + "}],\n";
    j += "  \"mode\": \"" + json_escape(run.mode) + "\",\n";
    j += "  \"created\": \"" + json_escape(created) + "\",\n";
    j += "  \"generator\": \"" + json_escape(run.generator) + "\",\n";
    j += "  \"run\": " + (run.run_params_json.empty() ? std::string("{}") : run.run_params_json) + ",\n";
    j += "  \"collect\": " + (run.collect_json.empty() ? std::string("{}") : run.collect_json) + ",\n";
    j += "  \"samples_done\": [";
    for (size_t i = 0; i < samples_done.size(); ++i)
        j += std::string(i ? ", " : "") + "\"" + json_escape(samples_done[i]) + "\"";
    j += "]\n}\n";
    return atomic_write_file((fs::path(dir) / kMeta).string(), j.data(), j.size(), err);
}

// ================================ THE SAMPLE SINK ================================

SampleRoutingSink::SampleRoutingSink(const RoutingStatsSchema& schema, bool has_contrib)
    : schema_(schema), moe_(schema.moe_layers()), k_((int32_t) schema.model.top_k), contrib_(has_contrib) {
    ids_.resize((size_t) schema.model.n_layers);
    w_.resize((size_t) schema.model.n_layers);
    c_.resize((size_t) schema.model.n_layers);
    seen_.resize((size_t) schema.model.n_layers);
    sim_seen_.resize((size_t) schema.model.n_layers);
    sim_.resize((size_t) schema.model.n_layers);
}

uint64_t SampleRoutingSink::bytes_for(int64_t n) const {
    return (uint64_t) moe_.size() * (uint64_t) n * ((uint64_t) k_ * (contrib_ ? 12 : 8) + 1);
}

uint64_t SampleRoutingSink::bytes_for(const CalibSample& s) const {
    const int64_t n = (int64_t) s.tokens.size();
    uint64_t b = bytes_for(n);
    if (schema_.similarity.on()) {
        const uint64_t sampled = (uint64_t) similarity_sampled_count(schema_.similarity, s.id, n);
        b += (uint64_t) n;   // which positions are sampled
        for (int64_t l : moe_)
            b += (uint64_t) n + sampled * SimilarityToken::bytes_for(schema_.model.experts_per_layer[(size_t) l], k_);
    }
    return b;
}

void SampleRoutingSink::begin(const CalibSample& s) {
    s_ = &s;
    err_.clear();
    const size_t n = s.tokens.size();
    for (int64_t l : moe_) {
        ids_[(size_t) l].assign(n * (size_t) k_, -1);
        w_[(size_t) l].assign(n * (size_t) k_, 0.0f);
        if (contrib_) c_[(size_t) l].assign(n * (size_t) k_, 0.0f);
        seen_[(size_t) l].assign(n, 0);
    }
    n_sampled_ = 0;
    if (schema_.similarity.on()) {
        sampled_.assign(n, 0);
        for (size_t p = 0; p < n; ++p) n_sampled_ += (sampled_[p] = similarity_sampled(schema_.similarity, s.id, (int64_t) p) ? 1 : 0);
        for (int64_t l : moe_) {
            sim_seen_[(size_t) l].assign(n, 0);
            sim_[(size_t) l].clear();
        }
    }
}

bool SampleRoutingSink::wants_expert_outputs(int64_t layer, int64_t pos) const {
    return s_ != nullptr && schema_.similarity.on() && layer >= 0 && layer < schema_.model.n_layers &&
           schema_.model.experts_per_layer[(size_t) layer] > 0 && pos >= 0 && pos < (int64_t) sampled_.size() &&
           sampled_[(size_t) pos] != 0;
}

void SampleRoutingSink::expert_outputs(int64_t layer, int64_t pos, const float* y, int64_t n, const int32_t* ids,
                                       const float* w) {
    if (s_ == nullptr || !err_.empty()) return;
    if (!wants_expert_outputs(layer, pos)) {
        err_ = "expert outputs for layer " + std::to_string(layer) + ", position " + std::to_string(pos) +
               ", which is not a sampled position of an MoE layer";
        return;
    }
    if (sim_seen_[(size_t) layer][(size_t) pos]++) {
        err_ = "layer " + std::to_string(layer) + " reported the expert outputs of position " + std::to_string(pos) + " twice";
        return;
    }
    SimilarityToken t;
    similarity_token(y, schema_.model.experts_per_layer[(size_t) layer], n, ids, w, k_, t);
    sim_[(size_t) layer].push_back(std::move(t));
}

void SampleRoutingSink::routed(int64_t layer, int64_t pos0, int64_t rows, const int32_t* ids, const float* w,
                               const float* contrib) {
    if (s_ == nullptr || !err_.empty()) return;
    const int64_t n = (int64_t) s_->tokens.size();
    if (layer < 0 || layer >= schema_.model.n_layers || schema_.model.experts_per_layer[(size_t) layer] <= 0) {
        err_ = "the model reported routing for layer " + std::to_string(layer) + ", which has no routed experts";
        return;
    }
    if (pos0 < 0 || rows < 0 || pos0 + rows > n) {
        err_ = "routing for positions " + std::to_string(pos0) + ".." + std::to_string(pos0 + rows) + " of a " +
               std::to_string(n) + "-token sample";
        return;
    }
    const int32_t ne = schema_.model.experts_per_layer[(size_t) layer];
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = pos0 + r;
        if (seen_[(size_t) layer][(size_t) p]++) {
            err_ = "layer " + std::to_string(layer) + " reported position " + std::to_string(p) + " twice";
            return;
        }
        for (int32_t j = 0; j < k_; ++j) {
            const int32_t e = ids[r * k_ + j];
            if (e < -1 || e >= ne) {
                err_ = "layer " + std::to_string(layer) + " routed expert id " + std::to_string(e) + " (the layer has " +
                       std::to_string(ne) + ")";
                return;
            }
            ids_[(size_t) layer][(size_t) (p * k_ + j)] = e;
            w_[(size_t) layer][(size_t) (p * k_ + j)] = w[r * k_ + j];
            if (contrib_ && contrib != nullptr) c_[(size_t) layer][(size_t) (p * k_ + j)] = contrib[r * k_ + j];
        }
    }
}

bool SampleRoutingSink::commit(std::vector<LayerStats>& layers, std::vector<LayerSimilarity>* sims, std::string& err) {
    if (!err_.empty()) {
        err = "sample \"" + (s_ ? s_->id : std::string()) + "\": " + err_;
        return false;
    }
    const int64_t n = (int64_t) s_->tokens.size();
    for (int64_t l : moe_)
        for (int64_t p = 0; p < n; ++p)
            if (seen_[(size_t) l][(size_t) p] != 1) {
                err = "sample \"" + s_->id + "\": layer " + std::to_string(l) + " gave no routing for position " +
                      std::to_string(p) + " (the path skipped it)";
                return false;
            }
    if (schema_.similarity.on()) {
        if (sims == nullptr) {
            err = "the similarity collection is on, but there are no similarity accumulators";
            return false;
        }
        for (int64_t l : moe_)
            for (int64_t p = 0; p < n; ++p)
                if (sampled_[(size_t) p] && sim_seen_[(size_t) l][(size_t) p] != 1) {
                    err = "sample \"" + s_->id + "\": layer " + std::to_string(l) + " gave no expert outputs for the sampled "
                          "position " + std::to_string(p) + ": this path does not collect similarity data";
                    return false;
                }
        for (int64_t l : moe_)
            for (const SimilarityToken& t : sim_[(size_t) l]) (*sims)[(size_t) l].add(t);
    }
    for (int64_t l : moe_) {
        LayerStats& st = layers[(size_t) l];
        for (int64_t p = 0; p < n; ++p)
            st.add_token((int32_t) s_->cat[(size_t) p], &ids_[(size_t) l][(size_t) (p * k_)], &w_[(size_t) l][(size_t) (p * k_)],
                         contrib_ ? &c_[(size_t) l][(size_t) (p * k_)] : nullptr);
    }
    return true;
}

// ================================ THE PROMPT PATH'S EXPERT OUTPUTS ================================

void ChunkOutputFeeder::routed(int64_t layer, int64_t pos0, int64_t T, const int32_t* ids, const float* w) {
    sink_.routed(layer, pos0, T, ids, w, nullptr);
    layer_ = layer;
    pos0_ = pos0;
    T_ = T;
    ids_.assign(ids, ids + T * k_);
    w_.assign(w, w + T * k_);
}

void ChunkOutputFeeder::wanted(int64_t layer, int64_t pos0, int64_t T, std::vector<int32_t>& rows) const {
    rows.clear();
    for (int64_t r = 0; r < T; ++r)
        if (sink_.wants_expert_outputs(layer, pos0 + r)) rows.push_back((int32_t) r);
}

void ChunkOutputFeeder::moe_rows(int64_t layer, int64_t pos0, const int32_t* rows, int64_t n, const float* x) {
    if (!err_.empty() || n <= 0) return;
    if (layer != layer_ || pos0 != pos0_) {
        err_ = "the expert input of layer " + std::to_string(layer) + ", position " + std::to_string(pos0) +
               " came without its routing";
        return;
    }
    for (int64_t i = 0; i < n; ++i)
        if (rows[i] < 0 || rows[i] >= T_) {
            err_ = "an expert input row outside its chunk";
            return;
        }
    y_.resize((size_t) n * (size_t) e_ * (size_t) width_);
    std::string e;
    if (!outputs_(layer, x, n, y_.data(), e)) {
        err_ = "every expert's output for the similarity data (layer " + std::to_string(layer) + "): " + e;
        return;
    }
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = rows[i];
        sink_.expert_outputs(layer, pos0 + r, y_.data() + (size_t) i * (size_t) e_ * (size_t) width_, width_,
                             ids_.data() + r * k_, w_.data() + r * k_);
    }
    rows_done_ += n;
}

bool ChunkOutputFeeder::ok(std::string& err) const {
    if (err_.empty()) return true;
    err = err_;
    return false;
}

// ================================ TOKEN MODE ================================

namespace {

bool same_categories(const CalibSet& set, const RoutingStatsSchema& schema, std::string& err) {
    if (set.categories.size() != schema.categories.size()) {
        err = "the sample file has " + std::to_string(set.categories.size()) + " categories, the run " +
              std::to_string(schema.categories.size());
        return false;
    }
    for (size_t c = 0; c < set.categories.size(); ++c)
        if (set.categories[c].key != schema.categories[c].key) {
            err = "category " + std::to_string(c) + " is \"" + set.categories[c].key + "\" in the sample file, \"" +
                  schema.categories[c].key + "\" in the run";
            return false;
        }
    return true;
}

/// The checkpoint of token mode: every MoE layer's accumulators and the ids of the samples they hold, one file.  With
/// the similarity collection on (flag 1) the configuration's hash follows the header and every layer's similarity
/// accumulators (sampled tokens, then the arrays) follow its statistics.
bool write_checkpoint(const std::string& dir, const RoutingStatsSchema& schema, uint64_t calib_fnv,
                      const std::vector<std::string>& done, uint64_t tokens, const std::vector<LayerStats>& layers,
                      const std::vector<LayerSimilarity>& sims, std::string& err) {
    const bool sim = schema.similarity.on();
    std::vector<uint8_t> h(kCkptMagic, kCkptMagic + 8);
    put_u32(h, kRoutingStatsVersion);
    put_u32(h, sim ? 1u : 0u);
    put_u64(h, schema.fingerprint_fnv());
    put_u64(h, schema.schema_fnv());
    put_u64(h, calib_fnv);
    put_u64(h, tokens);
    put_u64(h, (uint64_t) done.size());
    if (sim) put_u64(h, schema.similarity.fnv());
    for (const std::string& s : done) {
        put_u32(h, (uint32_t) s.size());
        h.insert(h.end(), s.begin(), s.end());
    }
    AtomicFile f;
    if (!f.open((fs::path(dir) / kCheckpoint).string(), err)) return false;
    bool ok = f.write(h.data(), h.size());
    for (int64_t l : schema.moe_layers()) {
        const LayerStats& s = layers[(size_t) l];
        ok = ok && write_u64s(f, s.token_array().data(), s.token_array().size()) &&
             write_u64s(f, s.value_array().data(), s.value_array().size());
        if (sim) {
            const LayerSimilarity& m = sims[(size_t) l];
            ok = ok && write_u64s(f, &m.tokens, 1) && write_u64s(f, m.data().data(), m.data().size());
        }
    }
    if (!ok) {
        err = "writing the checkpoint failed: is the drive full?";
        return false;
    }
    return f.commit(err);
}

bool read_checkpoint(const std::string& dir, const RoutingStatsSchema& schema, uint64_t calib_fnv,
                     std::vector<std::string>& done, uint64_t& tokens, std::vector<LayerStats>& layers,
                     std::vector<LayerSimilarity>& sims, std::string& err) {
    const std::string path = (fs::path(dir) / kCheckpoint).string();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    FileCloser fc{f};
    uint8_t h[56];
    if (!read_exact(f, h, sizeof h) || std::memcmp(h, kCkptMagic, 8) != 0 || get_u32(h + 8) != kRoutingStatsVersion) {
        err = path + ": not a checkpoint of this version";
        return false;
    }
    if (get_u64(h + 16) != schema.fingerprint_fnv()) {
        err = path + ": the interrupted run was on another model or pack: it cannot be resumed with this one";
        return false;
    }
    if (get_u64(h + 24) != schema.schema_fnv() || get_u64(h + 32) != calib_fnv) {
        err = path + ": the interrupted run read another prepared sample file: resume it with the same --prep";
        return false;
    }
    const bool sim = schema.similarity.on();
    if ((get_u32(h + 12) & 1u) != (sim ? 1u : 0u)) {
        err = path + (sim ? ": the interrupted run collected no similarity data: resume it without the similarity share"
                          : ": the interrupted run collected similarity data: resume it with the same share and seed");
        return false;
    }
    if (sim) {
        uint8_t c[8];
        if (!read_exact(f, c, 8)) { err = path + ": the file ends early"; return false; }
        if (get_u64(c) != schema.similarity.fnv()) {
            err = path + ": the interrupted run sampled the similarity data with another share or seed";
            return false;
        }
    }
    tokens = get_u64(h + 40);
    const uint64_t n = get_u64(h + 48);
    done.clear();
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t lb[4];
        if (!read_exact(f, lb, 4)) { err = path + ": the file ends early"; return false; }
        std::string s(get_u32(lb), '\0');
        if (!read_exact(f, s.data(), s.size())) { err = path + ": the file ends early"; return false; }
        done.push_back(std::move(s));
    }
    for (int64_t l : schema.moe_layers()) {
        LayerStats& s = layers[(size_t) l];
        if (!read_u64s(f, s.token_array().data(), s.token_array().size()) ||
            !read_u64s(f, s.value_array().data(), s.value_array().size())) {
            err = path + ": the file ends early";
            return false;
        }
        if (sim) {
            LayerSimilarity& m = sims[(size_t) l];
            if (!read_u64s(f, &m.tokens, 1) || !read_u64s(f, m.data().data(), m.data().size())) {
                err = path + ": the file ends early";
                return false;
            }
        }
    }
    uint8_t extra;
    if (std::fread(&extra, 1, 1, f) != 0) {
        err = path + ": bytes after the accumulators (written by another configuration?)";
        return false;
    }
    return true;
}

}  // namespace

bool collect_token_mode(const CalibSet& set, const RoutingStatsSchema& schema, const SampleRunner& run,
                        const CollectOptions& o, CollectReport& rep, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    rep = CollectReport{};
    auto log = [&](const std::string& s) { if (o.log) o.log(s); };
    if (!same_categories(set, schema, err)) return false;
    const std::vector<int64_t> moe = schema.moe_layers();
    if (moe.empty()) {
        err = "the model has no MoE layer: there are no experts to collect statistics for";
        return false;
    }
    std::error_code ec;
    fs::create_directories(o.out_dir, ec);
    if (ec) {
        err = "cannot create " + o.out_dir + ": " + ec.message();
        return false;
    }
    const fs::path dir(o.out_dir);
    if (fs::exists(dir / kMeta)) {
        err = o.out_dir + " already holds complete statistics (meta.json); give another --out";
        return false;
    }
    const bool have_ckpt = fs::exists(dir / kCheckpoint);
    if (have_ckpt && !o.resume) {
        err = o.out_dir + " holds an interrupted run (checkpoint.bin): --resume continues it, or give another --out";
        return false;
    }
    // the memory frame: every layer's accumulators (and similarity accumulators) and one sample's routing
    MemoryMeter ram(o.ram_budget, "RAM");
    SampleRoutingSink sink(schema, schema.has_contrib);
    const bool sim = schema.similarity.on();
    uint64_t acc_bytes = 0, sample_bytes = 0;
    for (int64_t l : moe) {
        acc_bytes += LayerStats::bytes_for(schema.model.experts_per_layer[(size_t) l], (int32_t) schema.model.top_k,
                                           (int32_t) schema.categories.size());
        if (sim) acc_bytes += LayerSimilarity::bytes_for(schema.model.experts_per_layer[(size_t) l]);
    }
    for (const CalibSample& s : set.samples) sample_bytes = std::max(sample_bytes, sink.bytes_for(s));
    if (!ram.take(acc_bytes, sim ? "the statistics and similarity accumulators of every layer"
                                 : "the statistics accumulators of every layer", err) ||
        !ram.take(sample_bytes, "the routing of the longest sample", err))
        return false;
    std::vector<LayerStats> layers((size_t) schema.model.n_layers);
    std::vector<LayerSimilarity> sims((size_t) schema.model.n_layers);
    for (int64_t l : moe) {
        layers[(size_t) l].reset(schema.model.experts_per_layer[(size_t) l], (int32_t) schema.model.top_k,
                                 (int32_t) schema.categories.size());
        if (sim) sims[(size_t) l].reset(schema.model.experts_per_layer[(size_t) l]);
    }
    std::vector<std::string> done;
    uint64_t tokens = 0;
    if (have_ckpt) {
        if (!read_checkpoint(o.out_dir, schema, set.fnv, done, tokens, layers, sims, err)) return false;
        rep.resumed = true;
        log("resuming after " + std::to_string(done.size()) + " samples (" + std::to_string(tokens) + " tokens)");
    } else if (o.resume) {
        log("nothing to resume in " + o.out_dir + ": starting from the first sample");
    }
    std::unordered_set<std::string> done_set(done.begin(), done.end());
    int64_t since = 0;
    for (size_t i = 0; i < set.samples.size(); ++i) {
        const CalibSample& s = set.samples[i];
        if (done_set.count(s.id)) {
            ++rep.samples_skipped;
            continue;
        }
        sink.begin(s);
        if (!run(s, sink, err)) {
            err = "sample \"" + s.id + "\": " + err;
            return false;
        }
        if (!sink.commit(layers, sim ? &sims : nullptr, err)) return false;
        done.push_back(s.id);
        done_set.insert(s.id);
        tokens += s.tokens.size();
        ++rep.samples;
        rep.tokens += (int64_t) s.tokens.size();
        rep.similarity_tokens += sink.sampled();
        if (o.fault && o.fault("sample", (int64_t) i, -1)) {
            err = "aborted (simulated) after sample " + std::to_string(i);
            return false;
        }
        if (++since >= std::max<int64_t>(o.checkpoint_every, 1)) {
            if (!write_checkpoint(o.out_dir, schema, set.fnv, done, tokens, layers, sims, err)) return false;
            since = 0;
            log("checkpoint: " + std::to_string(done.size()) + " of " + std::to_string(set.samples.size()) + " samples");
        }
    }
    // the shard: the layer files (and similarity files), then meta.json (its presence marks the run complete), then the
    // checkpoint goes
    if (!write_checkpoint(o.out_dir, schema, set.fnv, done, tokens, layers, sims, err)) return false;
    for (int64_t l : moe) {
        layers[(size_t) l].blocks = 0;
        if (!write_layer_stats(o.out_dir, l, layers[(size_t) l], schema, err)) return false;
        if (sim) {
            sims[(size_t) l].blocks = 0;
            if (!write_layer_similarity(o.out_dir, l, sims[(size_t) l], schema, err)) return false;
        }
    }
    RoutingStatsRun r = o.run;
    if (r.mode.empty()) r.mode = "token";
    if (r.run_params_json.empty() || r.run_params_json == "{}")   // what shaped the result (the checkpoint interval does not)
        r.run_params_json = "{\"mode\": \"token\", \"ram_budget\": " + std::to_string(o.ram_budget) + "}";
    if (!write_stats_meta(o.out_dir, schema, r, done, tokens, err)) return false;
    fs::remove(dir / kCheckpoint, ec);
    rep.ram_peak = ram.peak();
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
