// src/core/expert_similarity.cpp - the experts' output similarity for the substitute mode (expert_similarity.hpp).
#include "strata/core/expert_similarity.hpp"

#include "strata/core/routing_stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

namespace strata::core {

namespace fs = std::filesystem;

namespace {

constexpr char kSimMagic[8] = {'S', 'X', 'E', 'S', 'I', 'M', '0', '1'};
constexpr size_t kSimHeader = 64;

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

}  // namespace

// ================================ SAMPLING ================================

uint64_t SimilarityConfig::fnv() const {
    char b[96];
    std::snprintf(b, sizeof b, "similarity-v1;fraction=%.17g;seed=%llu;cap=%.17g", fraction, (unsigned long long) seed,
                  kSimilarityRelCap);
    return whitelist_fnv1a64(b, std::strlen(b));
}

bool similarity_sampled(const SimilarityConfig& c, const std::string& id, int64_t pos) {
    if (!c.on()) return false;
    if (c.fraction >= 1.0) return true;
    const std::string key = std::to_string(c.seed) + ":" + id + ":" + std::to_string(pos);
    const uint64_t h = whitelist_fnv1a64(key.data(), key.size());
    // h / 2^64 < fraction, without rounding h to a double first
    return (double) (h >> 11) < c.fraction * 9007199254740992.0;   // 2^53
}

int64_t similarity_sampled_count(const SimilarityConfig& c, const std::string& id, int64_t n) {
    int64_t s = 0;
    for (int64_t p = 0; p < n; ++p) s += similarity_sampled(c, id, p);
    return s;
}

// ================================ ONE TOKEN ================================

void similarity_token(const float* y, int32_t n_expert, int64_t n, const int32_t* ids, const float* w, int32_t k,
                      SimilarityToken& out) {
    out.clear();
    out.n_expert = n_expert;
    std::vector<double> norm((size_t) n_expert, 0.0);
    for (int32_t e = 0; e < n_expert; ++e) {
        const float* ye = y + (size_t) e * (size_t) n;
        double ss = 0.0;
        for (int64_t h = 0; h < n; ++h) ss += (double) ye[h] * (double) ye[h];
        norm[(size_t) e] = std::sqrt(ss);
    }
    for (int32_t j = 0; j < k; ++j) {
        const int32_t i = ids[j];
        if (i < 0 || i >= n_expert || !(w[j] > 0.0f) || !(norm[(size_t) i] > 0.0)) continue;
        const double wi = (double) w[j];
        const float* yi = y + (size_t) i * (size_t) n;
        out.rows.push_back(i);
        out.weight.push_back(routing_fixed(wi, kSimilarityFracBits));
        for (int32_t e = 0; e < n_expert; ++e) {
            const float* ye = y + (size_t) e * (size_t) n;
            double dot = 0.0, d2 = 0.0;
            for (int64_t h = 0; h < n; ++h) {
                dot += (double) yi[h] * (double) ye[h];
                const double d = (double) yi[h] - (double) ye[h];
                d2 += d * d;
            }
            double cs = norm[(size_t) e] > 0.0 ? dot / (norm[(size_t) i] * norm[(size_t) e]) : 0.0;
            cs = std::min(1.0, std::max(-1.0, cs));
            const double rel = std::min(std::sqrt(d2) / norm[(size_t) i], kSimilarityRelCap);
            out.cos.push_back(routing_fixed(wi * (1.0 + cs), kSimilarityFracBits));
            out.rel.push_back(routing_fixed(wi * rel, kSimilarityFracBits));
        }
    }
}

// ================================ ACCUMULATORS ================================

void LayerSimilarity::reset(int32_t n_expert) {
    n_ = n_expert;
    v_.assign((size_t) (2 * (uint64_t) n_expert + 2 * (uint64_t) n_expert * n_expert), 0);
    tokens = 0;
    blocks = 0;
}

void LayerSimilarity::add(const SimilarityToken& t) {
    ++tokens;
    if (t.n_expert != n_) return;
    const size_t E = (size_t) n_;
    for (size_t r = 0; r < t.rows.size(); ++r) {
        const size_t i = (size_t) t.rows[r];
        v_[i] += t.weight[r];
        v_[E + i] += 1;
        uint64_t* cs = &v_[2 * E + i * E];
        uint64_t* rl = &v_[2 * E + E * E + i * E];
        for (size_t j = 0; j < E; ++j) {
            cs[j] += t.cos[r * E + j];
            rl[j] += t.rel[r * E + j];
        }
    }
}

void LayerSimilarity::merge(const LayerSimilarity& o) {
    tokens += o.tokens;
    for (size_t i = 0; i < v_.size() && i < o.v_.size(); ++i) v_[i] += o.v_[i];
}

double LayerSimilarity::mean_cos(int32_t i, int32_t j) const {
    const uint64_t w = row_weight(i);
    return w == 0 ? std::nan("") : (double) cos_sum(i, j) / (double) w - 1.0;
}

double LayerSimilarity::mean_rel(int32_t i, int32_t j) const {
    const uint64_t w = row_weight(i);
    return w == 0 ? std::nan("") : (double) rel_sum(i, j) / (double) w;
}

// ================================ FILES ================================

std::string similarity_file_name(int64_t layer) {
    char b[32];
    std::snprintf(b, sizeof b, "similarity_%03lld.bin", (long long) layer);
    return b;
}

bool write_layer_similarity(const std::string& dir, int64_t layer, const LayerSimilarity& s, const RoutingStatsSchema& schema,
                            std::string& err) {
    std::vector<uint8_t> b(kSimMagic, kSimMagic + 8);
    put_u32(b, kSimilarityVersion);
    put_u32(b, (uint32_t) layer);
    put_u32(b, (uint32_t) s.n_expert());
    put_u32(b, (uint32_t) kSimilarityFracBits);
    put_u32(b, (uint32_t) std::lround(kSimilarityRelCap * 1000.0));
    put_u32(b, 0);
    put_u64(b, s.tokens);
    put_u64(b, s.blocks);
    put_u64(b, schema.fingerprint_fnv());
    put_u64(b, schema.similarity.fnv());
    b.reserve(b.size() + s.data().size() * 8);
    for (uint64_t v : s.data()) put_u64(b, v);
    if (!atomic_write_file((fs::path(dir) / similarity_file_name(layer)).string(), b.data(), b.size(), err)) {
        err = "writing the similarity data of layer " + std::to_string(layer) + " failed: " + err;
        return false;
    }
    return true;
}

bool read_layer_similarity(const std::string& dir, int64_t layer, const RoutingStatsSchema& schema, LayerSimilarity& out,
                           std::string& err) {
    const std::string path = (fs::path(dir) / similarity_file_name(layer)).string();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    std::vector<uint8_t> data;
    uint8_t buf[1 << 16];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    const int32_t ne = layer >= 0 && layer < schema.model.n_layers ? schema.model.experts_per_layer[(size_t) layer] : -1;
    if (data.size() < kSimHeader || std::memcmp(data.data(), kSimMagic, 8) != 0) {
        err = path + ": not a similarity file";
        return false;
    }
    const uint8_t* h = data.data();
    if (get_u32(h + 8) != kSimilarityVersion || (int64_t) get_u32(h + 12) != layer || (int32_t) get_u32(h + 16) != ne ||
        get_u32(h + 20) != (uint32_t) kSimilarityFracBits || get_u32(h + 24) != (uint32_t) std::lround(kSimilarityRelCap * 1000.0)) {
        err = path + ": the version, layer, expert count or scale does not match this run";
        return false;
    }
    if (get_u64(h + 48) != schema.fingerprint_fnv()) {
        err = path + ": the similarity data was collected on another model or pack";
        return false;
    }
    if (get_u64(h + 56) != schema.similarity.fnv()) {
        err = path + ": the similarity data was collected with another sample share or seed";
        return false;
    }
    out.reset(ne);
    if (data.size() != kSimHeader + out.data().size() * 8) {
        err = path + ": wrong size";
        return false;
    }
    out.tokens = get_u64(h + 32);
    out.blocks = get_u64(h + 40);
    for (size_t i = 0; i < out.data().size(); ++i) out.data()[i] = get_u64(h + kSimHeader + 8 * i);
    return true;
}

}  // namespace strata::core
