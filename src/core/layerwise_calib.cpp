// src/core/layerwise_calib.cpp - layer-wise calibration (layerwise_calib.hpp).
#include "strata/core/layerwise_calib.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <future>
#include <limits>

namespace strata::core {

namespace fs = std::filesystem;

// ================================ HALF AND BFLOAT16 ================================

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t exp = (x >> 23) & 0xffu;
    uint32_t man = x & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t) (sign | 0x7c00u | (man ? 0x200u : 0u));   // inf, nan
    int32_t e = (int32_t) exp - 127 + 15;
    if (e >= 0x1f) return (uint16_t) (sign | 0x7c00u);                             // overflow: inf
    if (e <= 0) {                                                                    // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const int shift = 14 - e;
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) e << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // may carry into the exponent: still right (or inf)
    return (uint16_t) (sign | h);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (man == 0) {
            x = sign;
        } else {   // subnormal: normalise
            int e = -1;
            do { ++e; man <<= 1; } while ((man & 0x400u) == 0);
            x = sign | ((uint32_t) (127 - 15 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        x = sign | 0x7f800000u | (man << 13);
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

uint16_t float_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t) ((x >> 16) | 0x40u);   // nan stays nan
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t) (x >> 16);
}

float bf16_to_float(uint16_t h) {
    const uint32_t x = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

bool parse_spill_precision(const std::string& s, SpillPrecision& out) {
    if (s == "f32" || s == "fp32") out = SpillPrecision::F32;
    else if (s == "f16" || s == "fp16") out = SpillPrecision::F16;
    else if (s == "bf16") out = SpillPrecision::BF16;
    else return false;
    return true;
}

const char* spill_precision_name(SpillPrecision p) {
    return p == SpillPrecision::F16 ? "f16" : p == SpillPrecision::BF16 ? "bf16" : "f32";
}

int spill_bytes(SpillPrecision p) { return p == SpillPrecision::F32 ? 4 : 2; }

// ================================ THE PLAN ================================

namespace {

constexpr char kProgressMagic[8] = {'S', 'X', 'L', 'W', 'P', 'R', 'G', '1'};
constexpr const char* kProgress = "progress.bin";

std::string mib(uint64_t b) {
    char s[64];
    std::snprintf(s, sizeof s, "%.1f MiB", (double) b / 1048576.0);
    return s;
}

/// The scheduler's RAM per token of a block (split models): the residual after the mixer, the experts' input and
/// output, the ids and weights.
uint64_t block_bytes_per_token(const LayerwiseModel& m) {
    if (m.fused()) return 0;
    return (uint64_t) (m.row_floats() + 2 * m.moe_floats()) * 4 + (uint64_t) m.top_k() * 8;
}

/// The RAM a block needs beyond its per-token buffers: two layer accumulators (the committed one and the block's), one
/// sequence's input and output rows and the conversion buffer of the scratch files, the prefetched layer; with the
/// similarity collection two layer similarity accumulators and one token's contribution.
uint64_t fixed_ram(const LayerwiseModel& m, const CalibSet& set, const RoutingStatsSchema& schema, const LayerwiseConfig& c,
                   bool prefetch) {
    uint64_t acc = 0, tok = 0;
    for (int64_t l = 0; l < m.n_layers(); ++l)
        if (m.n_expert(l) > 0) {
            uint64_t a = LayerStats::bytes_for(m.n_expert(l), m.top_k(), (int32_t) schema.categories.size());
            if (schema.similarity.on()) {
                a += LayerSimilarity::bytes_for(m.n_expert(l));
                tok = std::max(tok, SimilarityToken::bytes_for(m.n_expert(l), m.top_k()));
            }
            acc = std::max(acc, a);
        }
    const uint64_t seq = (uint64_t) set.longest() * (uint64_t) m.row_floats() * (uint64_t) (8 + spill_bytes(c.spill));
    uint64_t pf = 0;
    if (prefetch)
        for (int64_t l = 0; l < m.n_layers(); ++l) pf = std::max(pf, m.layer_bytes(l));
    return 2 * acc + tok + seq + pf;
}

/// The largest expert count of a layer.
int32_t max_experts(const LayerwiseModel& m) {
    int32_t e = 0;
    for (int64_t l = 0; l < m.n_layers(); ++l) e = std::max(e, m.n_expert(l));
    return e;
}

/// The split form's RAM for the sampled tokens' outputs of every expert: per sampled row, E x moe_floats floats.
uint64_t similarity_row_bytes(const LayerwiseModel& m) {
    return m.fused() ? 0 : (uint64_t) max_experts(m) * (uint64_t) m.moe_floats() * 4;
}

}  // namespace

bool plan_layerwise(const LayerwiseModel& m, const CalibSet& set, const RoutingStatsSchema& schema,
                    const LayerwiseConfig& c, LayerwisePlan& plan) {
    plan = LayerwisePlan{};
    const int64_t L = m.n_layers(), R = m.row_floats(), X = m.moe_floats(), K = m.top_k();
    const int64_t total = set.tokens(), longest = set.longest();
    const uint64_t act = (uint64_t) spill_bytes(c.spill);
    if (set.samples.empty()) plan.problems.push_back("there are no samples");
    bool any_moe = false;
    for (int64_t l = 0; l < L; ++l) any_moe = any_moe || m.n_expert(l) > 0;
    if (!any_moe) plan.problems.push_back("the model has no MoE layer: there are no experts to collect statistics for");
    // RAM: what does not depend on the block size, then what each token of a block adds
    const uint64_t per_tok = block_bytes_per_token(m);
    bool prefetch = c.prefetch && !m.fused();
    uint64_t fixed = fixed_ram(m, set, schema, c, prefetch);
    if (prefetch && c.ram_budget > 0 && fixed + per_tok * (uint64_t) longest > c.ram_budget) {
        prefetch = false;   // the read-ahead is optional: without room for it the run goes without
        fixed = fixed_ram(m, set, schema, c, false);
    }
    // the similarity collection: what a sampled token's outputs of every expert take, on average per token of a block
    // (the first guess of the block size; the block that samples the most is checked below)
    const SimilarityConfig& sim = schema.similarity;
    const uint64_t sim_row = sim.on() ? similarity_row_bytes(m) : 0;
    const uint64_t per_tok_sized = per_tok + (uint64_t) std::ceil(std::min(sim.fraction, 1.0) * (double) sim_row);
    int64_t max_block = std::numeric_limits<int64_t>::max();
    if (c.ram_budget > 0) {
        if (fixed >= c.ram_budget) {
            plan.problems.push_back("the RAM budget of " + mib(c.ram_budget) + " is below the " + mib(fixed) +
                                    " every block needs (two layer accumulators, one sequence's rows of " +
                                    std::to_string(longest) + " tokens)");
            max_block = 0;
        } else if ((c.block_tokens > 0 ? per_tok : per_tok_sized) > 0) {
            // a block size given on the command line is checked without the guess: the exact check below covers it
            max_block = (int64_t) ((c.ram_budget - fixed) / (c.block_tokens > 0 ? per_tok : per_tok_sized));
        }
    }
    if (c.scratch_max > 0) {   // the two boundaries of a block
        const int64_t by_scratch = (int64_t) (c.scratch_max / (2 * (uint64_t) R * act));
        max_block = std::min(max_block, by_scratch);
    }
    plan.max_block_tokens = max_block == std::numeric_limits<int64_t>::max() ? 0 : max_block;
    // nothing limits the block: kDefaultBlockTokens, so the scratch files stay bounded (two boundaries of that many rows)
    const bool unbounded = max_block == std::numeric_limits<int64_t>::max();
    int64_t B = c.block_tokens > 0 ? c.block_tokens
              : unbounded           ? std::max(std::min<int64_t>(total, kDefaultBlockTokens), longest)
                                    : max_block;
    if (c.block_tokens > 0 && c.block_tokens > max_block)
        plan.problems.push_back("a block of " + std::to_string(c.block_tokens) + " tokens needs " +
                                mib(fixed + per_tok * (uint64_t) c.block_tokens) + " of RAM and " +
                                mib(2 * (uint64_t) c.block_tokens * (uint64_t) R * act) +
                                " of scratch; the budgets hold blocks of " + std::to_string(std::max<int64_t>(max_block, 0)) +
                                " tokens");
    if (longest > B)
        plan.problems.push_back("the longest sample has " + std::to_string(longest) + " tokens, more than a block (" +
                                std::to_string(std::max<int64_t>(B, 0)) + "); a sample is never split, so raise the "
                                "RAM budget or shorten the samples (prepare's max_tokens_per_sample)");
    B = std::max<int64_t>(B, longest);
    plan.block_tokens = B;
    // the blocks: samples in file order, packed up to B tokens.  The similarity collection's sampled tokens are a share
    // on average only: when the block that samples the most needs more than the RAM budget, the block size shrinks (by a
    // sixteenth at a time) until it fits, unless --block-tokens fixed it
    int64_t largest = 0;
    std::vector<int64_t> sampled;   // per block: the tokens the similarity collection samples
    auto pack = [&](int64_t size) {
        plan.blocks.clear();
        largest = 0;
        int64_t first = 0, cur = 0;
        for (int64_t i = 0; i < (int64_t) set.samples.size(); ++i) {
            const int64_t n = (int64_t) set.samples[(size_t) i].tokens.size();
            if (cur > 0 && cur + n > size) {
                plan.blocks.emplace_back(first, i);
                largest = std::max(largest, cur);
                first = i;
                cur = 0;
            }
            cur += n;
        }
        if (cur > 0) {
            plan.blocks.emplace_back(first, (int64_t) set.samples.size());
            largest = std::max(largest, cur);
        }
        sampled.assign(plan.blocks.size(), 0);
        plan.similarity_rows_max = 0;
        if (sim.on())
            for (size_t b = 0; b < plan.blocks.size(); ++b) {
                for (int64_t i = plan.blocks[b].first; i < plan.blocks[b].second; ++i)
                    sampled[b] += similarity_sampled_count(sim, set.samples[(size_t) i].id,
                                                           (int64_t) set.samples[(size_t) i].tokens.size());
                plan.similarity_rows_max = std::max(plan.similarity_rows_max, sampled[b]);
            }
        plan.similarity_bytes = (uint64_t) plan.similarity_rows_max * sim_row;
        plan.ram_need = fixed + per_tok * (uint64_t) largest + plan.similarity_bytes;
    };
    pack(B);
    while (sim.on() && c.block_tokens <= 0 && c.ram_budget > 0 && plan.ram_need > c.ram_budget && B > longest) {
        B = std::max(longest, B - std::max<int64_t>(1, B / 16));
        pack(B);
    }
    plan.block_tokens = B;
    if (sim.on() && c.ram_budget > 0 && plan.ram_need > c.ram_budget)
        plan.problems.push_back("a block samples up to " + std::to_string(plan.similarity_rows_max) + " tokens for the "
                                "similarity data, whose outputs of every expert take " + mib(plan.similarity_bytes) +
                                "; with them a block needs " + mib(plan.ram_need) + ", more than the RAM budget of " +
                                mib(c.ram_budget) + ": lower --block-tokens or the similarity share");
    plan.scratch_need = 2 * (uint64_t) largest * (uint64_t) R * act;
    if (c.scratch_max > 0 && plan.scratch_need > c.scratch_max)
        plan.problems.push_back("the scratch files need " + mib(plan.scratch_need) + ", more than the " +
                                mib(c.scratch_max) + " allowed");
    // VRAM (split models; a fused model places its own buffers): the layer, then the largest of one expert with a row
    // batch, one sequence through the mixer, one through the combine
    uint64_t vram = 0, layer_expert = 0;
    for (int64_t l = 0; l < L; ++l) {
        const uint64_t lb = m.layer_bytes(l);
        const uint64_t eb = m.n_expert(l) > 0 ? m.expert_bytes(l) : 0;
        const uint64_t seqb = (uint64_t) longest * (uint64_t) (R + 2 * X) * 4;
        const uint64_t expb = m.n_expert(l) > 0 ? eb + 2 * (uint64_t) X * 4 + (uint64_t) K * 8 : 0;
        vram = std::max(vram, lb + std::max(seqb, expb));
        layer_expert = std::max(layer_expert, lb + eb);
    }
    plan.vram_need = m.fused() ? 0 : vram;
    if (!m.fused() && c.vram_budget > 0 && vram > c.vram_budget)
        plan.problems.push_back("the VRAM budget of " + mib(c.vram_budget) + " is below the " + mib(vram) +
                                " the largest layer needs with one expert and one sequence of " + std::to_string(longest) +
                                " tokens");
    // a router batch holds rows and their ids and weights, an expert batch rows in and out: the larger per row
    plan.rows_per_batch = std::numeric_limits<int64_t>::max();
    const uint64_t per_row = std::max<uint64_t>(2 * (uint64_t) X * 4, (uint64_t) X * 4 + (uint64_t) K * 8);
    if (!m.fused() && c.vram_budget > 0 && c.vram_budget > layer_expert)
        plan.rows_per_batch = std::max<int64_t>(1, (int64_t) ((c.vram_budget - layer_expert) / per_row));
    plan.prefetch = prefetch;
    // what the run reads from the drive: every layer once per block, the experts a block routes to (every expert of a
    // block that samples a token for the similarity data)
    for (size_t bi = 0; bi < plan.blocks.size(); ++bi) {
        const auto [b0, b1] = plan.blocks[bi];
        int64_t toks = 0;
        for (int64_t i = b0; i < b1; ++i) toks += (int64_t) set.samples[(size_t) i].tokens.size();
        for (int64_t l = 0; l < L; ++l) {
            plan.dense_read_bytes += m.layer_bytes(l);
            const int32_t E = m.n_expert(l);
            if (E <= 0) continue;
            const uint64_t eb = m.expert_bytes(l);
            plan.expert_read_upper += (uint64_t) E * eb;
            const double p_hit = sampled[bi] > 0 ? 1.0 : 1.0 - std::pow(1.0 - (double) K / (double) E, (double) toks);
            plan.expert_read_expected += (uint64_t) ((double) E * p_hit * (double) eb);
        }
    }
    std::string id = "B=" + std::to_string(B) + ";R=" + std::to_string(R) + ";X=" + std::to_string(X) + ";spill=" +
                     spill_precision_name(c.spill) + ";fused=" + (m.fused() ? "1" : "0") + ";";
    if (sim.on()) id += "sim=" + std::to_string(sim.fnv()) + ";";
    for (const auto& [b0, b1] : plan.blocks) id += std::to_string(b0) + "-" + std::to_string(b1) + ",";
    plan.plan_fnv = whitelist_fnv1a64(id.data(), id.size()) ^ set.fnv;
    return plan.problems.empty();
}

// ================================ THE RUN ================================

namespace {

struct Progress {
    uint64_t plan_fnv = 0, calib_fnv = 0, schema_fnv = 0;
    int64_t block = 0, layer = 0;
};

bool write_progress(const std::string& dir, const Progress& p, std::string& err) {
    std::vector<uint8_t> b(kProgressMagic, kProgressMagic + 8);
    auto put = [&](uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((uint8_t) (v >> (8 * i))); };
    put(p.plan_fnv);
    put(p.calib_fnv);
    put(p.schema_fnv);
    put((uint64_t) p.block);
    put((uint64_t) p.layer);
    return atomic_write_file((fs::path(dir) / kProgress).string(), b.data(), b.size(), err);
}

bool read_progress(const std::string& dir, Progress& p, std::string& err) {
    const std::string path = (fs::path(dir) / kProgress).string();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot read " + path;
        return false;
    }
    uint8_t b[48];
    const bool ok = std::fread(b, 1, sizeof b, f) == sizeof b && std::memcmp(b, kProgressMagic, 8) == 0;
    std::fclose(f);
    if (!ok) {
        err = path + ": not a layer-wise progress record";
        return false;
    }
    auto get = [&](int at) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t) b[at + i] << (8 * i); return v; };
    p.plan_fnv = get(8);
    p.calib_fnv = get(16);
    p.schema_fnv = get(24);
    p.block = (int64_t) get(32);
    p.layer = (int64_t) get(40);
    return true;
}

/// A boundary file's name ("block_NNNNNN_layer_NNN.rows", or with ".tmp" while it is written): the only files the
/// scheduler ever removes.
bool is_boundary_file(const fs::path& p) {
    std::string n = p.filename().string();
    if (n.size() > 4 && n.compare(n.size() - 4, 4, ".tmp") == 0) n.resize(n.size() - 4);
    size_t i = 0;
    auto lit = [&](const char* w) {
        const size_t k = std::strlen(w);
        if (n.compare(i, k, w) != 0) return false;
        i += k;
        return true;
    };
    auto digits = [&] {
        const size_t i0 = i;
        while (i < n.size() && n[i] >= '0' && n[i] <= '9') ++i;
        return i > i0;
    };
    return lit("block_") && digits() && lit("_layer_") && digits() && lit(".rows") && i == n.size();
}

std::string boundary_name(const std::string& scratch, int64_t block, int64_t layer) {
    char b[64];
    std::snprintf(b, sizeof b, "block_%06lld_layer_%03lld.rows", (long long) block, (long long) layer);
    return (fs::path(scratch) / b).string();
}

/// One sequence's rows to and from a boundary file in the spill precision.
class Boundary {
public:
    Boundary(SpillPrecision p, int64_t row_floats) : p_(p), R_(row_floats) {}
    ~Boundary() { close_in(); }
    bool open_in(const std::string& path, std::string& err) {
        close_in();
        in_ = std::fopen(path.c_str(), "rb");
        if (in_ == nullptr) {
            err = "the scratch file " + path + " is missing: the run cannot continue from this layer";
            return false;
        }
        return true;
    }
    void close_in() {
        if (in_ != nullptr) std::fclose(in_);
        in_ = nullptr;
    }
    bool read(float* rows, int64_t n, std::string& err) {
        const size_t cnt = (size_t) (n * R_);
        bool ok;
        if (p_ == SpillPrecision::F32) {
            ok = std::fread(rows, 4, cnt, in_) == cnt;
        } else {
            buf_.resize(cnt);
            ok = std::fread(buf_.data(), 2, cnt, in_) == cnt;
            for (size_t i = 0; ok && i < cnt; ++i)
                rows[i] = p_ == SpillPrecision::F16 ? half_to_float(buf_[i]) : bf16_to_float(buf_[i]);
        }
        if (!ok) err = "a scratch file ends early (written by an interrupted run?)";
        return ok;
    }
    bool write(AtomicFile& f, const float* rows, int64_t n) {
        const size_t cnt = (size_t) (n * R_);
        if (p_ == SpillPrecision::F32) return f.write(rows, cnt * 4);
        buf_.resize(cnt);
        for (size_t i = 0; i < cnt; ++i) buf_[i] = p_ == SpillPrecision::F16 ? float_to_half(rows[i]) : float_to_bf16(rows[i]);
        return f.write(buf_.data(), cnt * 2);
    }

private:
    SpillPrecision p_;
    int64_t R_;
    std::FILE* in_ = nullptr;
    std::vector<uint16_t> buf_;
};

/// A fused model's routing of one sequence into the block's pending accumulators of the layer (and, with the similarity
/// collection, the sampled positions' expert outputs into the pending similarity accumulators).
class PendingSink : public RoutingSink {
public:
    PendingSink(LayerStats& st, const CalibSample& s, int64_t layer, int32_t k, LayerSimilarity* sim = nullptr,
                const SimilarityConfig* sc = nullptr)
        : st_(st), s_(s), layer_(layer), k_(k), sim_(sim) {
        seen_.assign(s.tokens.size(), 0);
        if (sim_ != nullptr && sc != nullptr && sc->on()) {
            sampled_.assign(s.tokens.size(), 0);
            for (size_t p = 0; p < s.tokens.size(); ++p) sampled_[p] = similarity_sampled(*sc, s.id, (int64_t) p) ? 1 : 0;
            sim_seen_.assign(s.tokens.size(), 0);
        } else {
            sim_ = nullptr;
        }
    }
    bool wants_expert_outputs(int64_t layer, int64_t pos) const override {
        return sim_ != nullptr && layer == layer_ && pos >= 0 && pos < (int64_t) sampled_.size() && sampled_[(size_t) pos];
    }
    void expert_outputs(int64_t layer, int64_t pos, const float* y, int64_t n, const int32_t* ids, const float* w) override {
        if (!err_.empty()) return;
        if (!wants_expert_outputs(layer, pos)) { err_ = "expert outputs for a position that is not sampled"; return; }
        if (sim_seen_[(size_t) pos]++) { err_ = "the expert outputs of a position came twice"; return; }
        similarity_token(y, sim_->n_expert(), n, ids, w, k_, tok_);
        sim_->add(tok_);
    }
    void routed(int64_t layer, int64_t pos0, int64_t rows, const int32_t* ids, const float* w,
                const float* contrib) override {
        if (!err_.empty()) return;
        if (layer != layer_ || pos0 < 0 || pos0 + rows > (int64_t) s_.tokens.size()) {
            err_ = "routing for layer " + std::to_string(layer) + " positions " + std::to_string(pos0) + ".." +
                   std::to_string(pos0 + rows) + " while running layer " + std::to_string(layer_);
            return;
        }
        for (int64_t r = 0; r < rows; ++r) {
            if (seen_[(size_t) (pos0 + r)]++) { err_ = "a position was routed twice"; return; }
            for (int32_t j = 0; j < k_; ++j)
                if (ids[r * k_ + j] >= st_.n_expert()) { err_ = "a routed id is out of range"; return; }
        }
        // after the checks: a rejected call adds nothing
        for (int64_t r = 0; r < rows; ++r)
            st_.add_token((int32_t) s_.cat[(size_t) (pos0 + r)], ids + r * k_, w + r * k_,
                          contrib != nullptr ? contrib + r * k_ : nullptr);
    }
    bool check(std::string& err) const {
        if (!err_.empty()) { err = err_; return false; }
        for (uint8_t v : seen_)
            if (v != 1) {
                err = "layer " + std::to_string(layer_) + " gave no routing for a position of sample \"" + s_.id + "\"";
                return false;
            }
        for (size_t p = 0; p < sim_seen_.size(); ++p)
            if (sampled_[p] && sim_seen_[p] != 1) {
                err = "layer " + std::to_string(layer_) + " gave no expert outputs for the sampled position " +
                      std::to_string(p) + " of sample \"" + s_.id + "\": this model does not collect similarity data";
                return false;
            }
        return true;
    }

private:
    LayerStats& st_;
    const CalibSample& s_;
    int64_t layer_;
    int32_t k_;
    LayerSimilarity* sim_;
    std::vector<uint8_t> seen_, sampled_, sim_seen_;
    SimilarityToken tok_;
    std::string err_;
};

/// RAM taken from the meter for the scope of a buffer.
struct Taken {
    MemoryMeter& m;
    uint64_t b = 0;
    ~Taken() { m.give(b); }
};

uint64_t dir_files_bytes(const std::string& dir, int64_t& count) {
    uint64_t b = 0;
    count = 0;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file(ec)) {
            b += e.file_size(ec);
            ++count;
        }
    return b;
}

}  // namespace

bool collect_layerwise(LayerwiseModel& m, const CalibSet& set, const RoutingStatsSchema& schema, const LayerwiseConfig& c,
                       const CollectOptions& o, CollectReport& rep, LayerwiseCounters* counters, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    rep = CollectReport{};
    auto log = [&](const std::string& s) { if (o.log) o.log(s); };
    const int64_t L = m.n_layers(), R = m.row_floats(), X = m.moe_floats();
    const int32_t K = m.top_k();
    // ---- the model must be the one the schema describes, the samples the ones it names
    if (L != schema.model.n_layers || K != schema.model.top_k) {
        err = "the model's layers or top-k differ from the statistics schema";
        return false;
    }
    for (int64_t l = 0; l < L; ++l)
        if (m.n_expert(l) != schema.model.experts_per_layer[(size_t) l]) {
            err = "layer " + std::to_string(l) + " has " + std::to_string(m.n_expert(l)) + " experts, the schema " +
                  std::to_string(schema.model.experts_per_layer[(size_t) l]);
            return false;
        }
    if (set.categories.size() != schema.categories.size()) {
        err = "the sample file's categories differ from the statistics schema";
        return false;
    }
    for (size_t i = 0; i < set.categories.size(); ++i)
        if (set.categories[i].key != schema.categories[i].key) {
            err = "the sample file's categories differ from the statistics schema";
            return false;
        }
    LayerwisePlan plan;
    if (!plan_layerwise(m, set, schema, c, plan)) {
        err = "the layer-wise run does not fit:";
        for (const std::string& p : plan.problems) err += "\n  " + p;
        return false;
    }
    // ---- the directories, and where a resumed run continues.  The boundary files live in a folder of their own inside
    // the scratch folder, named by the plan and the output folder, and only files with the boundaries' own names are
    // ever removed: a scratch folder the user names (a drive's root, /tmp) keeps everything else, and two runs never
    // share or read each other's files
    std::error_code ec;
    fs::create_directories(o.out_dir, ec);
    if (ec) {
        err = "cannot create " + o.out_dir + ": " + ec.message();
        return false;
    }
    const bool default_scratch = c.scratch_dir.empty();
    const fs::path scratch_root = default_scratch ? fs::path(o.out_dir) / "scratch" : fs::path(c.scratch_dir);
    const std::string out_canon = fs::weakly_canonical(fs::path(o.out_dir), ec).string();
    char sub[48];
    std::snprintf(sub, sizeof sub, "strata-layerwise-%016llx",
                  (unsigned long long) (plan.plan_fnv ^ whitelist_fnv1a64(out_canon.data(), out_canon.size())));
    const std::string scratch = (scratch_root / sub).string();
    fs::create_directories(scratch, ec);
    if (ec) {
        err = "cannot create the scratch folder " + scratch + ": " + ec.message();
        return false;
    }
    if (fs::exists(fs::path(o.out_dir) / "meta.json")) {
        err = o.out_dir + " already holds complete statistics (meta.json); give another --out";
        return false;
    }
    const std::vector<int64_t> moe = schema.moe_layers();
    bool started = fs::exists(fs::path(o.out_dir) / kProgress);
    for (int64_t l : moe) started = started || fs::exists(fs::path(o.out_dir) / layer_stats_name(l));
    Progress pr;
    pr.plan_fnv = plan.plan_fnv;
    pr.calib_fnv = set.fnv;
    pr.schema_fnv = schema.schema_fnv();
    if (started && !o.resume) {
        err = o.out_dir + " holds an interrupted run: --resume continues it, or give another --out";
        return false;
    }
    if (started) {
        if (!fs::exists(fs::path(o.out_dir) / kProgress)) {
            err = o.out_dir + " has statistics files but no progress.bin: it cannot be resumed";
            return false;
        }
        Progress have;
        if (!read_progress(o.out_dir, have, err)) return false;
        if (have.calib_fnv != pr.calib_fnv || have.schema_fnv != pr.schema_fnv) {
            err = "the interrupted run read another prepared sample file: resume it with the same --prep";
            return false;
        }
        if (have.plan_fnv != pr.plan_fnv) {
            err = "the interrupted run had other blocks (block size, scratch precision or budgets): resume it with the "
                  "same --block-tokens, --spill and budgets";
            return false;
        }
        pr.block = have.block;
        pr.layer = have.layer;
        rep.resumed = true;
        log("resuming at block " + std::to_string(pr.block) + " of " + std::to_string(plan.blocks.size()) + ", layer " +
            std::to_string(pr.layer));
    } else if (o.resume) {
        log("nothing to resume in " + o.out_dir + ": starting with the first block");
    }
    // the input of the layer the run continues with can be gone (scratch files on another drive whose last rename a power
    // loss undid, or a tmpfs after a reboot): the block then starts again at layer 0; the layers that already counted it
    // (their files say so) run again for their output rows only
    if (pr.layer > 0 && !fs::exists(boundary_name(scratch, pr.block, pr.layer))) {
        log("the scratch file of block " + std::to_string(pr.block) + ", layer " + std::to_string(pr.layer) +
            " is gone: the block starts again at layer 0");
        pr.layer = 0;
    }
    // the scratch folder holds only the input of the layer the run continues with
    for (const auto& e : fs::directory_iterator(scratch, ec)) {
        if (!is_boundary_file(e.path())) continue;
        if (pr.layer > 0 && e.path().string() == boundary_name(scratch, pr.block, pr.layer)) continue;
        fs::remove(e.path(), ec);
    }
    if (const fs::space_info sp = fs::space(scratch, ec); !ec && sp.available < plan.scratch_need) {
        err = "the scratch drive (" + scratch + ") has " + mib(sp.available) + " free, the run needs " + mib(plan.scratch_need);
        return false;
    }
    // ---- the memory frames: the scheduler's buffers, and the model's own where it reports them
    MemoryMeter ram(c.ram_budget, "RAM"), vram(c.vram_budget, "VRAM");
    m.bind_meters(&ram, &vram);
    struct Unbind { LayerwiseModel& m; ~Unbind() { m.bind_meters(nullptr, nullptr); } } unbind{m};
    int64_t largest = 0;
    for (const auto& [b0, b1] : plan.blocks) {
        int64_t t = 0;
        for (int64_t i = b0; i < b1; ++i) t += (int64_t) set.samples[(size_t) i].tokens.size();
        largest = std::max(largest, t);
    }
    const bool split = !m.fused();
    const SimilarityConfig& simc = schema.similarity;
    const bool sim = simc.on();
    uint64_t acc_max = 0, tok_max = 0;
    for (int64_t l : moe) {
        uint64_t a = LayerStats::bytes_for(m.n_expert(l), K, (int32_t) schema.categories.size());
        if (sim) {
            a += LayerSimilarity::bytes_for(m.n_expert(l));
            tok_max = std::max(tok_max, SimilarityToken::bytes_for(m.n_expert(l), K));
        }
        acc_max = std::max(acc_max, a);
    }
    const uint64_t seq_bytes = (uint64_t) set.longest() * (uint64_t) R * 8;
    Taken t_acc{ram}, t_seq{ram}, t_blk{ram}, t_sim{ram};
    if (!ram.take(2 * acc_max + tok_max, sim ? "two layer accumulators and similarity accumulators" : "two layer accumulators",
                  err))
        return false;
    t_acc.b = 2 * acc_max + tok_max;
    if (!ram.take(seq_bytes + (uint64_t) set.longest() * (uint64_t) R * (uint64_t) spill_bytes(c.spill),
                  "one sequence's input and output rows", err))
        return false;
    t_seq.b = seq_bytes + (uint64_t) set.longest() * (uint64_t) R * (uint64_t) spill_bytes(c.spill);
    const uint64_t blk = split ? block_bytes_per_token(m) * (uint64_t) largest : 0;
    if (!ram.take(blk, "the block's rows, expert inputs and outputs and routing", err)) return false;
    t_blk.b = blk;
    if (!ram.take(plan.similarity_bytes, "the sampled tokens' outputs of every expert (the similarity data)", err)) return false;
    t_sim.b = plan.similarity_bytes;
    const int32_t E_max = max_experts(m);
    std::vector<float> ysim(split && sim ? (size_t) (plan.similarity_rows_max * E_max * X) : 0);
    std::vector<int64_t> srows;      // the block's sampled rows (split form)
    SimilarityToken stok;
    std::vector<float> rows_mid(split ? (size_t) (largest * R) : 0), moe_in(split ? (size_t) (largest * X) : 0),
        moe_out(split ? (size_t) (largest * X) : 0), seq_in((size_t) (set.longest() * R)), seq_out((size_t) (set.longest() * R));
    std::vector<int32_t> ids(split ? (size_t) (largest * K) : 0);
    std::vector<float> w(split ? (size_t) (largest * K) : 0);
    if (counters != nullptr) {
        int64_t mx = 0;
        for (int64_t l = 0; l < L; ++l) mx = std::max<int64_t>(mx, m.n_expert(l));
        counters->n_layers = L;
        counters->max_expert = mx;
        counters->layer_loads.assign(plan.blocks.size() * (size_t) L, 0);
        counters->expert_loads.assign(plan.blocks.size() * (size_t) L * (size_t) std::max<int64_t>(mx, 1), 0);
    }
    const int64_t batch = plan.rows_per_batch;
    Boundary bnd(c.spill, R);
    LayerStats pending, committed;
    LayerSimilarity pending_sim, committed_sim;
    std::string pf_err;   // before the future: its destructor waits for the task that writes this string
    std::future<bool> pf;
    int64_t pf_layer = -1;
    auto wait_prefetch = [&]() -> bool {
        if (!pf.valid()) return true;
        const bool ok = pf.get();
        if (!ok) err = "reading layer " + std::to_string(pf_layer) + " ahead: " + pf_err;
        return ok;
    };
    auto fault = [&](const char* point, int64_t b, int64_t l) {
        if (o.fault && o.fault(point, b, l)) {
            (void) wait_prefetch();
            err = std::string("aborted (simulated) at ") + point + ", block " + std::to_string(b) + ", layer " +
                  std::to_string(l);
            return true;
        }
        return false;
    };
    const bool prefetch = plan.prefetch;   // plan_layerwise priced it in, or turned it off
    if (c.prefetch && !prefetch) log("the read-ahead of the next layer is off: the RAM budget does not hold it");
    // a fresh run records where it is before anything else is written, so a statistics file never exists without one
    if (!started && !write_progress(o.out_dir, pr, err)) return false;
    auto count_scratch = [&]() {
        int64_t files = 0;
        rep.scratch_peak = std::max(rep.scratch_peak, dir_files_bytes(scratch, files));
        if (counters != nullptr) counters->boundary_files_max = std::max(counters->boundary_files_max, files);
    };
    std::vector<int64_t> tok_off;   // the block's first row of each sample
    for (int64_t b = pr.block; b < (int64_t) plan.blocks.size(); ++b) {
        const auto [s0, s1] = plan.blocks[(size_t) b];
        tok_off.assign((size_t) (s1 - s0 + 1), 0);
        for (int64_t i = s0; i < s1; ++i)
            tok_off[(size_t) (i - s0 + 1)] = tok_off[(size_t) (i - s0)] + (int64_t) set.samples[(size_t) i].tokens.size();
        const int64_t T = tok_off.back();
        // the block's sampled rows for the similarity data (the same tokens in every layer and on every path)
        srows.clear();
        if (sim)
            for (int64_t i = s0; i < s1; ++i)
                for (int64_t p = 0; p < (int64_t) set.samples[(size_t) i].tokens.size(); ++p)
                    if (similarity_sampled(simc, set.samples[(size_t) i].id, p)) srows.push_back(tok_off[(size_t) (i - s0)] + p);
        for (int64_t l = (b == pr.block ? pr.layer : 0); l < L; ++l) {
            const int32_t E = m.n_expert(l);
            const bool last = l + 1 == L;
            // the block's routing of this layer: its own accumulators, added to the layer's file at the end; a layer
            // whose file already holds this block (a crash after that write) runs again only for its output rows.  The
            // similarity data likewise, by its own file (a crash can fall between the two writes)
            bool count = E > 0, count_sim = E > 0 && sim;
            if (E > 0) {
                pending.reset(E, K, (int32_t) schema.categories.size());
                if (fs::exists(fs::path(o.out_dir) / layer_stats_name(l))) {
                    if (!read_layer_stats(o.out_dir, l, schema, committed, err)) return false;
                    count = (int64_t) committed.blocks <= b;
                } else {
                    committed.reset(E, K, (int32_t) schema.categories.size());
                }
            }
            if (count_sim) {
                pending_sim.reset(E);
                if (fs::exists(fs::path(o.out_dir) / similarity_file_name(l))) {
                    if (!read_layer_similarity(o.out_dir, l, schema, committed_sim, err)) return false;
                    count_sim = (int64_t) committed_sim.blocks <= b;
                } else {
                    committed_sim.reset(E);
                }
            }
            if (counters != nullptr) ++counters->layer_loads[(size_t) (b * L + l)];
            if (split) {
                if (pf_layer == l && !wait_prefetch()) return false;
                if (!m.load_layer(l, err)) {
                    err = "layer " + std::to_string(l) + ": " + err;
                    return false;
                }
                if (prefetch && !last) {   // the next layer from the drive while this one computes
                    pf_layer = l + 1;
                    pf = std::async(std::launch::async, [&m, &pf_err, l] { return m.prefetch_layer(l + 1, pf_err); });
                    if (counters != nullptr) ++counters->prefetched;
                }
            }
            if (l > 0 && !bnd.open_in(boundary_name(scratch, b, l), err)) return false;
            AtomicFile out;
            if (!last && !out.open(boundary_name(scratch, b, l + 1), err)) return false;
            count_scratch();
            if (!split) {
                // ---- fused: the model runs the whole layer for one sequence at a time
                for (int64_t i = s0; i < s1; ++i) {
                    const CalibSample& s = set.samples[(size_t) i];
                    const int64_t n = (int64_t) s.tokens.size();
                    if (l > 0 && !bnd.read(seq_in.data(), n, err)) return false;
                    LayerStats scratch_stats;
                    LayerStats& into = count ? pending : scratch_stats;
                    if (E > 0 && !count) scratch_stats.reset(E, K, (int32_t) schema.categories.size());
                    LayerSimilarity scratch_sim;   // a block the file already holds: its outputs are checked, not counted
                    if (E > 0 && sim && !count_sim) scratch_sim.reset(E);
                    PendingSink sink(into, s, l, K, E > 0 && sim ? (count_sim ? &pending_sim : &scratch_sim) : nullptr, &simc);
                    if (!m.forward_layer(l, s, l > 0 ? seq_in.data() : nullptr, seq_out.data(), sink, err)) {
                        err = "layer " + std::to_string(l) + ", sample \"" + s.id + "\": " + err;
                        return false;
                    }
                    if (E > 0 && !sink.check(err)) return false;
                    if (!last && !bnd.write(out, seq_out.data(), n)) {
                        err = "writing the scratch rows failed: is the drive full?";
                        return false;
                    }
                    if (i == (s0 + s1) / 2 && fault("mid_experts", b, l)) return false;
                }
            } else {
                // ---- split: the mixer per sequence, the router over the block, each expert once, the combine
                for (int64_t i = s0; i < s1; ++i) {
                    const CalibSample& s = set.samples[(size_t) i];
                    const int64_t n = (int64_t) s.tokens.size(), r0 = tok_off[(size_t) (i - s0)];
                    float* rows = rows_mid.data() + r0 * R;
                    if (l == 0) {
                        if (!m.embed(s, rows, err)) return false;
                    } else if (!bnd.read(rows, n, err)) {
                        return false;
                    }
                    if (!m.mixer(l, s, rows, moe_in.data() + r0 * X, err)) {
                        err = "layer " + std::to_string(l) + ", sample \"" + s.id + "\": " + err;
                        return false;
                    }
                }
                std::fill(moe_out.begin(), moe_out.begin() + T * X, 0.0f);
                if (E > 0) {
                    for (int64_t r = 0; r < T; r += std::min<int64_t>(batch, T)) {
                        const int64_t nr = std::min<int64_t>(std::min<int64_t>(batch, T), T - r);
                        if (!m.route(l, moe_in.data() + r * X, nr, ids.data() + r * K, w.data() + r * K, err)) {
                            err = "layer " + std::to_string(l) + " router: " + err;
                            return false;
                        }
                    }
                    if (fault("after_route", b, l)) return false;
                    // the denominators, and every pick grouped by expert (a counting sort keeps the token order)
                    std::vector<int32_t> cat_of((size_t) T);
                    for (int64_t i = s0; i < s1; ++i)
                        for (int64_t p = 0; p < (int64_t) set.samples[(size_t) i].tokens.size(); ++p)
                            cat_of[(size_t) (tok_off[(size_t) (i - s0)] + p)] = set.samples[(size_t) i].cat[(size_t) p];
                    if (count)
                        for (int64_t t = 0; t < T; ++t) pending.add_denominator(cat_of[(size_t) t]);
                    std::vector<int64_t> start((size_t) E + 1, 0);
                    for (int64_t q = 0; q < T * K; ++q) {
                        const int32_t e = ids[(size_t) q];
                        if (e < -1 || e >= E) {
                            err = "layer " + std::to_string(l) + " routed expert id " + std::to_string(e);
                            return false;
                        }
                        if (e >= 0) ++start[(size_t) e + 1];
                    }
                    for (int32_t e = 0; e < E; ++e) start[(size_t) e + 1] += start[(size_t) e];
                    std::vector<int64_t> picks((size_t) start[(size_t) E]), fill(start.begin(), start.end() - 1);
                    for (int64_t q = 0; q < T * K; ++q)
                        if (ids[(size_t) q] >= 0) picks[(size_t) fill[(size_t) ids[(size_t) q]]++] = q;
                    // the similarity data: a block that samples a token runs every expert on its sampled rows too
                    const bool sim_rows = sim && !srows.empty();
                    const int64_t S = sim_rows ? (int64_t) srows.size() : 0;
                    int32_t used = 0;
                    for (int32_t e = 0; e < E; ++e) used += start[(size_t) e + 1] > start[(size_t) e] || sim_rows;
                    int32_t ran = 0;
                    std::vector<float> xb, yb;
                    for (int32_t e = 0; e < E; ++e) {
                        const int64_t a = start[(size_t) e], z = start[(size_t) e + 1];
                        if (a == z && !sim_rows) continue;   // no token of the block routed here: never loaded
                        if (counters != nullptr)
                            ++counters->expert_loads[(size_t) ((b * L + l) * counters->max_expert + e)];
                        if (!m.load_expert(l, e, err)) {
                            err = "layer " + std::to_string(l) + ", expert " + std::to_string(e) + ": " + err;
                            return false;
                        }
                        for (int64_t q0 = a; q0 < z; q0 += batch) {
                            const int64_t nq = std::min<int64_t>(batch, z - q0);
                            xb.resize((size_t) (nq * X));
                            yb.resize((size_t) (nq * X));
                            for (int64_t j = 0; j < nq; ++j) {
                                const int64_t t = picks[(size_t) (q0 + j)] / K;
                                std::memcpy(xb.data() + j * X, moe_in.data() + t * X, (size_t) X * 4);
                            }
                            if (!m.run_expert(l, e, xb.data(), nq, yb.data(), err)) {
                                m.release_expert(l, e);
                                err = "layer " + std::to_string(l) + ", expert " + std::to_string(e) + ": " + err;
                                return false;
                            }
                            for (int64_t j = 0; j < nq; ++j) {
                                const int64_t q = picks[(size_t) (q0 + j)], t = q / K;
                                const int32_t rank = (int32_t) (q % K);
                                const float wq = w[(size_t) q];
                                const float* y = yb.data() + j * X;
                                float* acc = moe_out.data() + t * X;
                                for (int64_t h = 0; h < X; ++h) acc[h] += wq * y[h];
                                if (count)
                                    pending.add_pick(cat_of[(size_t) t], e, rank, wq,
                                                     schema.has_contrib ? routing_contrib(wq, y, X) : 0.0f);
                            }
                        }
                        for (int64_t q0 = 0; q0 < S; q0 += batch) {   // the sampled rows: this expert's output on each
                            const int64_t nq = std::min<int64_t>(batch, S - q0);
                            xb.resize((size_t) (nq * X));
                            yb.resize((size_t) (nq * X));
                            for (int64_t j = 0; j < nq; ++j)
                                std::memcpy(xb.data() + j * X, moe_in.data() + srows[(size_t) (q0 + j)] * X, (size_t) X * 4);
                            if (!m.run_expert(l, e, xb.data(), nq, yb.data(), err)) {
                                m.release_expert(l, e);
                                err = "layer " + std::to_string(l) + ", expert " + std::to_string(e) + " (similarity): " + err;
                                return false;
                            }
                            for (int64_t j = 0; j < nq; ++j)
                                std::memcpy(ysim.data() + ((size_t) (q0 + j) * (size_t) E + (size_t) e) * (size_t) X,
                                            yb.data() + j * X, (size_t) X * 4);
                        }
                        m.release_expert(l, e);
                        if (++ran == (used + 1) / 2 && fault("mid_experts", b, l)) return false;
                    }
                    if (count_sim)   // every expert has run: each sampled token's row of the matrix
                        for (int64_t s = 0; s < S; ++s) {
                            const int64_t t = srows[(size_t) s];
                            similarity_token(ysim.data() + (size_t) s * (size_t) E * (size_t) X, E, X, ids.data() + t * K,
                                             w.data() + t * K, K, stok);
                            pending_sim.add(stok);
                        }
                }
                for (int64_t i = s0; i < s1; ++i) {
                    const CalibSample& s = set.samples[(size_t) i];
                    const int64_t n = (int64_t) s.tokens.size(), r0 = tok_off[(size_t) (i - s0)];
                    float* rows = rows_mid.data() + r0 * R;
                    if (!m.combine(l, s, rows, moe_in.data() + r0 * X, moe_out.data() + r0 * X, err)) {
                        err = "layer " + std::to_string(l) + ", sample \"" + s.id + "\": " + err;
                        return false;
                    }
                    if (!last && !bnd.write(out, rows, n)) {
                        err = "writing the scratch rows failed: is the drive full?";
                        return false;
                    }
                }
                m.release_layer(l);
            }
            bnd.close_in();
            // ---- the layer is done for this block: its output rows, its statistics, the progress, then its input goes
            if (!last && !out.commit(err)) return false;
            count_scratch();
            if (fault("after_boundary", b, l)) return false;
            if (E > 0 && count) {
                committed.merge(pending);
                committed.blocks = (uint64_t) b + 1;
                if (!write_layer_stats(o.out_dir, l, committed, schema, err)) return false;
            }
            if (fault("after_stats", b, l)) return false;
            if (E > 0 && count_sim) {
                committed_sim.merge(pending_sim);
                committed_sim.blocks = (uint64_t) b + 1;
                if (!write_layer_similarity(o.out_dir, l, committed_sim, schema, err)) return false;
            }
            if (sim && fault("after_similarity", b, l)) return false;
            pr.block = last ? b + 1 : b;
            pr.layer = last ? 0 : l + 1;
            if (!write_progress(o.out_dir, pr, err)) return false;
            if (fault("after_progress", b, l)) return false;
            if (l > 0) fs::remove(boundary_name(scratch, b, l), ec);
        }
        rep.samples += s1 - s0;
        rep.tokens += T;
        rep.similarity_tokens += (int64_t) srows.size();
        log("block " + std::to_string(b + 1) + " of " + std::to_string(plan.blocks.size()) + " done (" +
            std::to_string(T) + " tokens)");
    }
    if (!wait_prefetch()) return false;
    // ---- complete: the final layer files (progress field 0, as token mode writes them), meta.json, then the rest goes
    std::vector<std::string> done;
    uint64_t tokens = 0;
    for (const CalibSample& s : set.samples) {
        done.push_back(s.id);
        tokens += s.tokens.size();
    }
    for (int64_t l : moe) {
        LayerStats s;
        if (!read_layer_stats(o.out_dir, l, schema, s, err)) return false;
        s.blocks = 0;
        if (!write_layer_stats(o.out_dir, l, s, schema, err)) return false;
        if (sim) {
            LayerSimilarity ls;
            if (!read_layer_similarity(o.out_dir, l, schema, ls, err)) return false;
            ls.blocks = 0;
            if (!write_layer_similarity(o.out_dir, l, ls, schema, err)) return false;
        }
    }
    RoutingStatsRun run = o.run;
    if (run.mode.empty()) run.mode = "layerwise";
    if (run.run_params_json.empty() || run.run_params_json == "{}")
        run.run_params_json = "{\"mode\": \"layerwise\", \"form\": \"" + std::string(split ? "split" : "fused") +
                              "\", \"block_tokens\": " + std::to_string(plan.block_tokens) + ", \"blocks\": " +
                              std::to_string(plan.blocks.size()) + ", \"spill\": \"" + spill_precision_name(c.spill) +
                              "\", \"vram_budget\": " + std::to_string(c.vram_budget) + ", \"ram_budget\": " +
                              std::to_string(c.ram_budget) + ", \"scratch_max\": " + std::to_string(c.scratch_max) +
                              ", \"prefetch\": " + (prefetch ? "true" : "false") + "}";
    if (!write_stats_meta(o.out_dir, schema, run, done, tokens, err)) return false;
    fs::remove(fs::path(o.out_dir) / kProgress, ec);
    if (!c.keep_scratch) {   // the boundary files, their folder if that leaves it empty, the default scratch folder likewise
        for (const auto& e : fs::directory_iterator(scratch, ec))
            if (is_boundary_file(e.path())) fs::remove(e.path(), ec);
        fs::remove(scratch, ec);
        if (default_scratch) fs::remove(scratch_root, ec);
    }
    rep.ram_peak = ram.peak();
    rep.vram_peak = vram.peak();
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace strata::core
