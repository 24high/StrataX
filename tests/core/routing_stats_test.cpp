// tests/core/routing_stats_test.cpp - the whitelist generator's statistics collection on a synthetic mini-MoE (no model,
// no GPU, no drive beyond a temporary folder).
//
// The model: 4 layers (layer 2 dense, the others 16 experts, top-2), width 8, a real causal attention per layer with a
// KV cache, and a router that sends each "language" to its own experts: tokens 0-15 (rust) to experts 0-3, 16-31
// (python) to 4-7, 32-47 (prose) to 8-11; 12-15 are general.  Dims 0-2 of the hidden row carry the language and
// nothing writes them, so the routing stays language-specific in every layer.  The weights live in a "drive" image
// (per layer: the dense blob, then the experts) read through counters.
//
// What it checks (docs/EXPERT_WHITELIST.md section 7.8):
//   1. statistics: the accumulators merge exactly (commutative, associative, a split run = the whole run); token mode
//      resumed after an abort gives the same bytes; the RAM frame is kept and a too-small one is refused up front;
//   2. the known language experts carry the language's routing (the Python test builds the whitelist from the fixture);
//   3. the layer-wise mode: the same statistics as the token path (byte for byte), small VRAM and RAM budgets kept,
//      every layer loaded once per block, every expert at most once per block and layer (and only when routed), at
//      most two boundaries on the drive, resumable after an abort at any point (mid-layer included) without starting
//      over; the fused form, the read-ahead and the 16-bit scratch files too.
//
// `routing_stats_test --write-fixture DIR` writes the token-mode statistics and the sample file into DIR
// (tests/data/expert_whitelist/stats-mini, which tools/test_whitelist_gen.py reads); without it the committed fixture
// is compared byte for byte, so the C++ writer and the Python reader stay in step.
#include "strata/core/layerwise_calib.hpp"
#include "strata/core/routing_stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);            \
            ++g_fail;                                                                    \
        }                                                                                \
    } while (0)

// ================================ THE MINI MODEL ================================

constexpr int L = 4, E = 16, K = 2, H = 8, A = 5, FF = 4, V = 48, DENSE_LAYER = 2;
constexpr int LANG = 3;   // dims 0..2: the language channels; attention, experts and the dense FFN write dims 3..7
constexpr int DENSE_FLOATS = 4 * A * A + E * A + 2 * FF * H + A * FF;   // Wq Wk Wv Wo | router | FFN gate up down
constexpr int EXPERT_FLOATS = 2 * FF * H + A * FF;                       // gate up (FF x H) | down (A x FF)
constexpr uint64_t DENSE_BYTES = DENSE_FLOATS * 4, EXPERT_BYTES = EXPERT_FLOATS * 4;
const char* const kCreated = "2026-10-21";

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    float next() {   // [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float) ((double) (s >> 40) / (double) (1ull << 23) - 1.0);
    }
    uint32_t below(uint32_t n) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (uint32_t) ((s >> 33) % n);
    }
};

int32_t n_expert_of(int64_t l) { return l == DENSE_LAYER ? 0 : E; }

/// The similarity test's twins: in every MoE layer expert 4 (a python expert) is a copy of expert 0 (rust) and expert 5
/// is expert 1 with its down projection halved, so their outputs are the same and half as long.
constexpr int TWIN_OF_0 = 4, HALF_OF_1 = 5;

struct Mini {
    std::vector<uint8_t> drive;
    std::vector<uint64_t> dense_off, expert_off;
    std::vector<float> embed;   // V x H
    explicit Mini(uint64_t seed, bool twins = false) {
        Lcg r(seed);
        uint64_t at = 0;
        for (int l = 0; l < L; ++l) {
            dense_off.push_back(at);
            at += DENSE_BYTES;
            expert_off.push_back(at);
            at += (uint64_t) n_expert_of(l) * EXPERT_BYTES;
        }
        drive.resize(at);
        for (int l = 0; l < L; ++l) {
            auto* d = (float*) (drive.data() + dense_off[(size_t) l]);
            for (int i = 0; i < DENSE_FLOATS; ++i) d[i] = 0.6f * r.next();
            for (int i = 4 * A * A; i < 4 * A * A + E * A; ++i) d[i] = 0.25f * r.next();   // the router's small part
            auto* x = (float*) (drive.data() + expert_off[(size_t) l]);
            for (int i = 0; i < n_expert_of(l) * EXPERT_FLOATS; ++i) x[i] = 0.7f * r.next();
        }
        embed.resize((size_t) V * H);
        for (int t = 0; t < V; ++t)
            for (int h = 0; h < H; ++h) embed[(size_t) t * H + h] = h < LANG ? (h == t / 16 ? 1.0f : 0.0f) : r.next();
        if (twins)
            for (int l = 0; l < L; ++l) {
                if (n_expert_of(l) == 0) continue;
                auto* x = (float*) (drive.data() + expert_off[(size_t) l]);
                std::memcpy(x + TWIN_OF_0 * EXPERT_FLOATS, x, EXPERT_BYTES);
                std::memcpy(x + HALF_OF_1 * EXPERT_FLOATS, x + EXPERT_FLOATS, EXPERT_BYTES);
                for (int i = 2 * FF * H; i < EXPERT_FLOATS; ++i) x[HALF_OF_1 * EXPERT_FLOATS + i] *= 0.5f;
            }
    }
    const float* dense(int64_t l) const { return (const float*) (drive.data() + dense_off[(size_t) l]); }
    const float* expert(int64_t l, int32_t e) const {
        return (const float*) (drive.data() + expert_off[(size_t) l] + (uint64_t) e * EXPERT_BYTES);
    }
    WhitelistModelInfo info() const {
        WhitelistModelInfo m;
        m.n_layers = L;
        m.top_k = K;
        m.hidden = H;
        uint64_t at = 0;
        for (int l = 0; l < L; ++l) {   // the experts' own layout (experts.bin): the fingerprint's pack index
            m.experts_per_layer.push_back(n_expert_of(l));
            m.layer_offset.push_back(at);
            m.blob_bytes.push_back(n_expert_of(l) > 0 ? EXPERT_BYTES : 0);
            at += (uint64_t) n_expert_of(l) * EXPERT_BYTES;
        }
        return m;
    }
};

// ---- the row math every path runs (one function each, so every path computes the same bits)

void qkv_row(const float* d, const float* x, float* q, float* k, float* v) {
    for (int i = 0; i < A; ++i) {
        float a = 0.0f, b = 0.0f, c = 0.0f;
        for (int j = 0; j < A; ++j) {
            a += d[i * A + j] * x[LANG + j];
            b += d[A * A + i * A + j] * x[LANG + j];
            c += d[2 * A * A + i * A + j] * x[LANG + j];
        }
        q[i] = a;
        k[i] = b;
        v[i] = c;
    }
}

/// Causal attention of one query over keys 0..nk-1, added to x's written dims through Wo.
void attend_row(const float* d, const float* q, const float* Ks, const float* Vs, int64_t nk, float* x) {
    std::vector<float> s((size_t) nk);
    float mx = -INFINITY;
    const float inv = 1.0f / std::sqrt((float) A);
    for (int64_t j = 0; j < nk; ++j) {
        float a = 0.0f;
        for (int i = 0; i < A; ++i) a += q[i] * Ks[j * A + i];
        s[(size_t) j] = a * inv;
        mx = std::max(mx, s[(size_t) j]);
    }
    float z = 0.0f;
    for (int64_t j = 0; j < nk; ++j) {
        s[(size_t) j] = std::exp(s[(size_t) j] - mx);
        z += s[(size_t) j];
    }
    float o[A] = {};
    for (int64_t j = 0; j < nk; ++j)
        for (int i = 0; i < A; ++i) o[i] += s[(size_t) j] * Vs[j * A + i];
    for (int i = 0; i < A; ++i) o[i] /= z;
    const float* wo = d + 3 * A * A;
    for (int i = 0; i < A; ++i) {
        float a = 0.0f;
        for (int j = 0; j < A; ++j) a += wo[i * A + j] * o[j];
        x[LANG + i] += a;
    }
}

/// router_top10.cu's semantics (softmax with double sums, strict greater, ties to the lower id, w / max(sum, 2^-14)).
void route_row(const float* d, const float* x, int32_t* ids, float* w) {
    const float* wr = d + 4 * A * A;
    float lg[E];
    for (int e = 0; e < E; ++e) {
        float a = 0.0f;
        for (int i = 0; i < A; ++i) a += wr[e * A + i] * x[LANG + i];
        lg[e] = (e < 12 ? 4.0f * x[e / 4] : 1.5f) + a;
    }
    float mx = -INFINITY;
    for (float v : lg) mx = std::max(mx, v);
    double ex[E], z = 0.0;
    for (int e = 0; e < E; ++e) { ex[e] = std::exp((double) lg[e] - (double) mx); z += ex[e]; }
    float p[E];
    for (int e = 0; e < E; ++e) p[e] = (float) (ex[e] / z);
    bool taken[E] = {};
    double sum = 0.0;
    for (int j = 0; j < K; ++j) {
        int bi = -1;
        for (int e = 0; e < E; ++e)
            if (!taken[e] && (bi < 0 || p[e] > p[bi])) bi = e;
        taken[bi] = true;
        ids[j] = bi;
        sum += p[bi];
    }
    sum = std::max(sum, 6.103515625e-5);
    for (int j = 0; j < K; ++j) w[j] = (float) ((double) p[ids[j]] / sum);
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

/// An FFN blob (an expert, or the dense layer's FFN): down(silu(gate x) * up x) into dims 3..7; dims 0..2 stay 0.
void ffn_row(const float* blob, const float* x, float* y) {
    float hb[FF];
    for (int f = 0; f < FF; ++f) {
        float g = 0.0f, u = 0.0f;
        for (int h = 0; h < H; ++h) {
            g += blob[f * H + h] * x[h];
            u += blob[FF * H + f * H + h] * x[h];
        }
        hb[f] = silu(g) * u;
    }
    for (int h = 0; h < LANG; ++h) y[h] = 0.0f;
    for (int i = 0; i < A; ++i) {
        float a = 0.0f;
        for (int f = 0; f < FF; ++f) a += blob[2 * FF * H + i * FF + f] * hb[f];
        y[LANG + i] = a;
    }
}

void combine_row(const float* d, int64_t l, float* x, const float* moe_in, const float* moe_out) {
    float f[H] = {};
    if (l == DENSE_LAYER) ffn_row(d + 4 * A * A + E * A, moe_in, f);
    for (int i = 0; i < A; ++i) x[LANG + i] = std::tanh(x[LANG + i] + moe_out[LANG + i] + f[LANG + i]);
}

/// A test of the refusal: the paths below report no expert outputs, as a path that cannot collect similarity data.
bool g_no_outputs = false;

/// Every expert's output on one expert input (the similarity collection's sampled positions), [E][H].
void all_expert_outputs(const Mini& m, int64_t l, const float* moe_in, float* y) {
    for (int32_t e = 0; e < n_expert_of(l); ++e) ffn_row(m.expert(l, e), moe_in, y + (size_t) e * H);
}

/// THE NORMAL INFERENCE PATH: one position at a time through every layer, a KV cache per layer, the picks summed in
/// ascending expert id.  Reports each layer's routing of each position to `sink`, and every expert's output at the
/// positions the sink asks for (the similarity data).
bool token_forward(const Mini& m, const CalibSample& s, RoutingSink& sink, std::string& err) {
    std::vector<std::vector<float>> kc((size_t) L), vc((size_t) L);
    for (size_t t = 0; t < s.tokens.size(); ++t) {
        const int32_t tok = s.tokens[t];
        if (tok < 0 || tok >= V) { err = "token out of range"; return false; }
        float x[H], moe_in[H];
        std::memcpy(x, &m.embed[(size_t) tok * H], sizeof x);
        for (int64_t l = 0; l < L; ++l) {
            const float* d = m.dense(l);
            float q[A], k[A], v[A];
            qkv_row(d, x, q, k, v);
            kc[(size_t) l].insert(kc[(size_t) l].end(), k, k + A);
            vc[(size_t) l].insert(vc[(size_t) l].end(), v, v + A);
            attend_row(d, q, kc[(size_t) l].data(), vc[(size_t) l].data(), (int64_t) t + 1, x);
            std::memcpy(moe_in, x, sizeof x);
            float moe_out[H] = {};
            if (n_expert_of(l) > 0) {
                int32_t ids[K];
                float w[K], contrib[K], y[H];
                route_row(d, moe_in, ids, w);
                int order[K];
                for (int j = 0; j < K; ++j) order[j] = j;
                std::sort(order, order + K, [&](int a, int b) { return ids[a] < ids[b]; });
                for (int jj = 0; jj < K; ++jj) {
                    const int j = order[jj];
                    ffn_row(m.expert(l, ids[j]), moe_in, y);
                    for (int h = 0; h < H; ++h) moe_out[h] += w[j] * y[h];
                    contrib[j] = routing_contrib(w[j], y, H);
                }
                sink.routed(l, (int64_t) t, 1, ids, w, contrib);
                if (!g_no_outputs && sink.wants_expert_outputs(l, (int64_t) t)) {
                    float yall[E * H];
                    all_expert_outputs(m, l, moe_in, yall);
                    sink.expert_outputs(l, (int64_t) t, yall, H, ids, w);
                }
            }
            combine_row(d, l, x, moe_in, moe_out);
        }
    }
    return true;
}

// ================================ THE LAYER-WISE MODELS ================================

/// The split form: the scheduler groups the experts.  Every "device" buffer is taken from the VRAM meter, the
/// read-ahead's RAM copy from the RAM meter; every read of the drive is counted.
class RefSplit : public LayerwiseModel {
public:
    explicit RefSplit(const Mini& m) : m_(m) {
        dense_reads.assign(L, 0);
        expert_reads.assign((size_t) L * E, 0);
    }
    int64_t n_layers() const override { return L; }
    int64_t row_floats() const override { return H; }
    int64_t moe_floats() const override { return H; }
    int32_t n_expert(int64_t l) const override { return n_expert_of(l); }
    int32_t top_k() const override { return K; }
    uint64_t layer_bytes(int64_t) const override { return DENSE_BYTES; }
    uint64_t expert_bytes(int64_t) const override { return EXPERT_BYTES; }
    void bind_meters(MemoryMeter* ram, MemoryMeter* vram) override { ram_ = ram; vram_ = vram; }

    bool embed(const CalibSample& s, float* rows, std::string&) override {
        for (size_t t = 0; t < s.tokens.size(); ++t) std::memcpy(rows + t * H, &m_.embed[(size_t) s.tokens[t] * H], H * 4);
        return true;
    }
    bool load_layer(int64_t l, std::string& err) override {
        if (layer_ >= 0) { err = "two layers loaded at once"; return false; }
        if (!vram_take(DENSE_BYTES, "a layer", err)) return false;
        std::vector<float> pre;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = ahead_.find(l);
            if (it != ahead_.end()) {
                pre = std::move(it->second);
                ahead_.erase(it);
                if (ram_) ram_->give(DENSE_BYTES);
            }
        }
        if (pre.empty()) {
            pre.assign(m_.dense(l), m_.dense(l) + DENSE_FLOATS);
            std::lock_guard<std::mutex> lk(mu_);
            ++dense_reads[(size_t) l];
            drive_bytes += DENSE_BYTES;
        }
        dev_layer_ = std::move(pre);
        layer_ = l;
        loads.push_back(l);
        routed.emplace_back();
        return true;
    }
    void release_layer(int64_t l) override {
        if (layer_ == l) {
            layer_ = -1;
            dev_layer_.clear();
            vram_give(DENSE_BYTES);
        }
    }
    bool prefetch_layer(int64_t l, std::string& err) override {
        if (l == fail_prefetch) {
            err = "a simulated read error";
            return false;
        }
        if (ram_ && !ram_->take(DENSE_BYTES, "the next layer read ahead", err)) return false;
        std::vector<float> c(m_.dense(l), m_.dense(l) + DENSE_FLOATS);
        std::lock_guard<std::mutex> lk(mu_);
        ++dense_reads[(size_t) l];
        drive_bytes += DENSE_BYTES;
        ahead_[l] = std::move(c);
        return true;
    }
    bool mixer(int64_t l, const CalibSample& s, float* rows, float* moe_in, std::string& err) override {
        const int64_t n = (int64_t) s.tokens.size();
        if (layer_ != l) { err = "the layer is not loaded"; return false; }
        if (!vram_take((uint64_t) n * 2 * H * 4, "a sequence's rows", err)) return false;
        std::vector<float> q((size_t) (n * A)), k((size_t) (n * A)), v((size_t) (n * A));
        for (int64_t t = 0; t < n; ++t) qkv_row(dev_layer_.data(), rows + t * H, &q[(size_t) (t * A)], &k[(size_t) (t * A)], &v[(size_t) (t * A)]);
        for (int64_t t = 0; t < n; ++t) attend_row(dev_layer_.data(), &q[(size_t) (t * A)], k.data(), v.data(), t + 1, rows + t * H);
        std::memcpy(moe_in, rows, (size_t) (n * H) * 4);
        vram_give((uint64_t) n * 2 * H * 4);
        return true;
    }
    bool route(int64_t l, const float* moe_in, int64_t n, int32_t* ids, float* w, std::string& err) override {
        if (layer_ != l) { err = "the layer is not loaded"; return false; }
        if (!vram_take((uint64_t) n * (H * 4 + K * 8), "a router batch", err)) return false;
        for (int64_t t = 0; t < n; ++t) {
            route_row(dev_layer_.data(), moe_in + t * H, ids + t * K, w + t * K);
            for (int j = 0; j < K; ++j) routed.back().insert(ids[t * K + j]);
        }
        vram_give((uint64_t) n * (H * 4 + K * 8));
        return true;
    }
    bool load_expert(int64_t l, int32_t e, std::string& err) override {
        if (expert_ >= 0) { err = "two experts loaded at once"; return false; }
        if (!vram_take(EXPERT_BYTES, "an expert", err)) return false;
        dev_expert_.assign(m_.expert(l, e), m_.expert(l, e) + EXPERT_FLOATS);
        ++expert_reads[(size_t) (l * E + e)];
        drive_bytes += EXPERT_BYTES;
        expert_ = e;
        return true;
    }
    void release_expert(int64_t, int32_t e) override {
        if (expert_ == e) {
            expert_ = -1;
            vram_give(EXPERT_BYTES);
        }
    }
    bool run_expert(int64_t, int32_t e, const float* x, int64_t n, float* y, std::string& err) override {
        if (expert_ != e) { err = "the expert is not loaded"; return false; }
        if (!vram_take((uint64_t) n * 2 * H * 4, "an expert batch", err)) return false;
        for (int64_t t = 0; t < n; ++t) ffn_row(dev_expert_.data(), x + t * H, y + t * H);
        vram_give((uint64_t) n * 2 * H * 4);
        return true;
    }
    bool combine(int64_t l, const CalibSample& s, float* rows, const float* moe_in, const float* moe_out,
                 std::string& err) override {
        const int64_t n = (int64_t) s.tokens.size();
        if (layer_ != l) { err = "the layer is not loaded"; return false; }
        if (!vram_take((uint64_t) n * 3 * H * 4, "a sequence's combine", err)) return false;
        for (int64_t t = 0; t < n; ++t) combine_row(dev_layer_.data(), l, rows + t * H, moe_in + t * H, moe_out + t * H);
        vram_give((uint64_t) n * 3 * H * 4);
        return true;
    }

    std::vector<int64_t> dense_reads, expert_reads, loads;
    int64_t fail_prefetch = -1;   ///< a layer whose read-ahead fails (tests)
    std::vector<std::set<int32_t>> routed;   // per load_layer: the experts the router picked
    uint64_t drive_bytes = 0;

protected:
    bool vram_take(uint64_t b, const char* what, std::string& err) { return vram_ == nullptr || vram_->take(b, what, err); }
    void vram_give(uint64_t b) { if (vram_) vram_->give(b); }
    const Mini& m_;
    MemoryMeter *ram_ = nullptr, *vram_ = nullptr;
    std::vector<float> dev_layer_, dev_expert_;
    int64_t layer_ = -1;
    int32_t expert_ = -1;
    std::mutex mu_;
    std::map<int64_t, std::vector<float>> ahead_;
};

/// The fused form (the engine's shape): one layer for one sequence, the experts per sequence.
class RefFused : public RefSplit {
public:
    using RefSplit::RefSplit;
    bool fused() const override { return true; }
    bool forward_layer(int64_t l, const CalibSample& s, const float* in, float* out, RoutingSink& sink,
                       std::string& err) override {
        const int64_t n = (int64_t) s.tokens.size();
        ++forwards;
        std::vector<float> rows((size_t) (n * H)), moe_in((size_t) (n * H)), moe_out((size_t) (n * H), 0.0f);
        if (in == nullptr) { if (!embed(s, rows.data(), err)) return false; }
        else std::memcpy(rows.data(), in, rows.size() * 4);
        if (!load_layer(l, err) || !mixer(l, s, rows.data(), moe_in.data(), err)) return false;
        if (n_expert_of(l) > 0) {
            std::vector<int32_t> ids((size_t) (n * K));
            std::vector<float> w((size_t) (n * K)), contrib((size_t) (n * K));
            if (!route(l, moe_in.data(), n, ids.data(), w.data(), err)) return false;
            std::vector<float> y((size_t) H);
            for (int32_t e = 0; e < E; ++e) {   // ascending id, as the split scheduler sums
                bool any = false;
                for (int64_t q = 0; q < n * K; ++q) any = any || ids[(size_t) q] == e;
                if (!any) continue;
                if (!load_expert(l, e, err)) return false;
                for (int64_t q = 0; q < n * K; ++q) {
                    if (ids[(size_t) q] != e) continue;
                    const int64_t t = q / K;
                    if (!run_expert(l, e, &moe_in[(size_t) (t * H)], 1, y.data(), err)) return false;
                    for (int h = 0; h < H; ++h) moe_out[(size_t) (t * H + h)] += w[(size_t) q] * y[(size_t) h];
                    contrib[(size_t) q] = routing_contrib(w[(size_t) q], y.data(), H);
                }
                release_expert(l, e);
            }
            sink.routed(l, 0, n, ids.data(), w.data(), contrib.data());
            std::vector<float> yall((size_t) E * H);
            for (int64_t t = 0; t < n && !g_no_outputs; ++t)
                if (sink.wants_expert_outputs(l, t)) {
                    all_expert_outputs(m_, l, &moe_in[(size_t) (t * H)], yall.data());
                    sink.expert_outputs(l, t, yall.data(), H, &ids[(size_t) (t * K)], &w[(size_t) (t * K)]);
                }
        }
        if (!combine(l, s, rows.data(), moe_in.data(), moe_out.data(), err)) return false;
        release_layer(l);
        std::memcpy(out, rows.data(), rows.size() * 4);
        return true;
    }
    int64_t forwards = 0;
};

// ================================ THE SAMPLES ================================

/// rust (24 samples: codegen and compiler errors, prompt and response), python (16), a base requirement in prose (12),
/// a prose reference corpus (12): tokens drawn from each language's 16 ids.
CalibSet make_samples() {
    CalibSet set;
    const char* keys[] = {
        "target|rust|codegen|prompt|prefill",          "target|rust|codegen|response|prefill",
        "target|rust|compiler_errors|prompt|prefill",  "target|rust|compiler_errors|response|prefill",
        "target|python|codegen|prompt|prefill",        "target|python|codegen|response|prefill",
        "base|instructions|instructions|prompt|prefill", "base|instructions|instructions|response|prefill",
        "reference|prose|reference|prompt|prefill",
    };
    for (const char* k : keys) {
        RoutingCategory c;
        std::string err;
        RoutingCategory::from_key(k, c, err);
        set.categories.push_back(c);
    }
    Lcg r(99);
    auto add = [&](const char* id, int lang, uint16_t cat_prompt, int16_t cat_resp, int n_prompt, int n_resp) {
        CalibSample s;
        s.id = id;
        for (int t = 0; t < n_prompt + n_resp; ++t) {
            s.tokens.push_back(lang * 16 + (int32_t) r.below(16));
            s.cat.push_back(t < n_prompt || cat_resp < 0 ? cat_prompt : (uint16_t) cat_resp);
        }
        set.samples.push_back(std::move(s));
    };
    char id[32];
    for (int i = 0; i < 24; ++i) {
        std::snprintf(id, sizeof id, "rust-%04d", i + 1);
        add(id, 0, i % 2 ? 2 : 0, i % 2 ? 3 : 1, 9 + i % 5, 7 + i % 4);
    }
    for (int i = 0; i < 16; ++i) {
        std::snprintf(id, sizeof id, "python-%04d", i + 1);
        add(id, 1, 4, 5, 8 + i % 3, 8 + i % 5);
    }
    for (int i = 0; i < 12; ++i) {
        std::snprintf(id, sizeof id, "base-instructions-%04d", i + 1);
        add(id, 2, 6, 7, 10, 6 + i % 3);
    }
    for (int i = 0; i < 12; ++i) {
        std::snprintf(id, sizeof id, "reference-%04d", i + 1);
        add(id, 2, 8, -1, 14 + i % 6, 0);
    }
    return set;
}

RoutingStatsSchema schema_for(const Mini& m, const CalibSet& set) {
    RoutingStatsSchema s;
    s.model = m.info();
    s.fingerprint = whitelist_fingerprint(s.model);
    s.categories = set.categories;
    s.has_contrib = true;
    s.quantization = "f32 (synthetic)";
    s.dense_bytes = (uint64_t) L * DENSE_BYTES;
    return s;
}

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// The statistics files of a run (the layer files, and meta.json when `with_meta`), by name.
std::map<std::string, std::string> stats_files(const fs::path& dir, bool with_meta) {
    std::map<std::string, std::string> out;
    for (int l = 0; l < L; ++l)
        if (n_expert_of(l) > 0) out[layer_stats_name(l)] = slurp(dir / layer_stats_name(l));
    if (with_meta) out["meta.json"] = slurp(dir / "meta.json");
    return out;
}

CollectOptions options(const fs::path& dir, const char* mode = "token") {
    CollectOptions o;
    o.out_dir = dir.string();
    o.run.created = kCreated;
    o.run.mode = mode;
    o.run.collect_json = "{\"test\": \"routing_stats_test\"}";
    return o;
}

fs::path g_tmp;

/// The reference: token mode over every sample.
fs::path g_token_dir;

// ================================ 1. STATISTICS ================================

void test_merge() {
    auto random_stats = [&](int seed) {
        LayerStats s;
        s.reset(E, K, 3);
        Lcg q((uint64_t) seed);
        for (int t = 0; t < 200; ++t) {
            int32_t ids[K] = {(int32_t) q.below(E), (int32_t) q.below(E)};
            if (ids[1] == ids[0]) ids[1] = (ids[0] + 1) % E;
            const float w0 = 0.5f + 0.5f * std::fabs(q.next());
            const float w[K] = {w0, 1.0f - w0}, c[K] = {3.0f * std::fabs(q.next()), 2.0f * std::fabs(q.next())};
            s.add_token((int32_t) q.below(3), ids, w, c);
        }
        return s;
    };
    const LayerStats a = random_stats(1), b = random_stats(2), c = random_stats(3);
    LayerStats ab = a, ba = b, ab_c, a_bc, bc = b;
    ab.merge(b);
    ba.merge(a);
    CHECK(ab == ba);   // commutative
    ab_c = ab;
    ab_c.merge(c);
    bc.merge(c);
    a_bc = a;
    a_bc.merge(bc);
    CHECK(ab_c == a_bc);   // associative
    CHECK(!(ab == a));
    // fixed point: the sum of two weights is the sum of their fixed points, whatever the order
    CHECK(routing_fixed(0.5, kGateFracBits) == (1ull << 31) && routing_fixed(-1.0, 32) == 0 && routing_fixed(NAN, 32) == 0);
    std::puts("routing_stats_test: merge is exact (commutative, associative)");
}

void test_token_mode(const Mini& m, const CalibSet& set) {
    const RoutingStatsSchema schema = schema_for(m, set);
    auto runner = [&](const CalibSample& s, RoutingSink& sink, std::string& err) { return token_forward(m, s, sink, err); };
    std::string err;
    CollectReport rep;
    // the whole run: the reference every other run is compared with
    g_token_dir = g_tmp / "token";
    CollectOptions o = options(g_token_dir);
    o.checkpoint_every = 5;
    if (!collect_token_mode(set, schema, runner, o, rep, err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
        return;
    }
    CHECK(rep.samples == (int64_t) set.samples.size() && rep.tokens == set.tokens());
    CHECK(fs::exists(g_token_dir / "meta.json") && !fs::exists(g_token_dir / "checkpoint.bin"));
    CHECK(!collect_token_mode(set, schema, runner, o, rep, err) && err.find("already holds complete") != std::string::npos);

    // the language experts carry their language: in every MoE layer, 95% of the rust tokens' router weight is on 0..3
    for (int l = 0; l < L; ++l) {
        if (n_expert_of(l) == 0) continue;
        LayerStats s;
        CHECK(read_layer_stats(g_token_dir.string(), l, schema, s, err));
        double own = 0.0, all = 0.0;
        for (int c = 0; c < 4; ++c)
            for (int e = 0; e < E; ++e) {
                all += (double) s.at(c, e, LayerStats::kGate);
                if (e < 4) own += (double) s.at(c, e, LayerStats::kGate);
            }
        CHECK(all > 0 && own / all > 0.95);
        uint64_t tok = 0;
        for (int c = 0; c < (int) set.categories.size(); ++c) tok += s.tokens(c);
        CHECK(tok == (uint64_t) set.tokens());
    }

    // a run split in two and merged is the whole run (exact)
    {
        CalibSet h1 = set, h2 = set;
        h1.samples.resize(set.samples.size() / 2);
        h2.samples.erase(h2.samples.begin(), h2.samples.begin() + (long) (set.samples.size() / 2));
        CollectOptions o1 = options(g_tmp / "half1"), o2 = options(g_tmp / "half2");
        CHECK(collect_token_mode(h1, schema, runner, o1, rep, err) && collect_token_mode(h2, schema, runner, o2, rep, err));
        for (int l = 0; l < L; ++l) {
            if (n_expert_of(l) == 0) continue;
            LayerStats a, b, whole;
            CHECK(read_layer_stats(o1.out_dir, l, schema, a, err) && read_layer_stats(o2.out_dir, l, schema, b, err) &&
                  read_layer_stats(g_token_dir.string(), l, schema, whole, err));
            a.merge(b);
            CHECK(a == whole);
        }
    }

    // an abort after sample 13 (checkpoints every 4): the resumed run writes the same bytes, meta.json included
    {
        CollectOptions oa = options(g_tmp / "token_abort");
        oa.checkpoint_every = 4;
        oa.fault = [](const char* point, int64_t i, int64_t) { return std::strcmp(point, "sample") == 0 && i == 13; };
        CHECK(!collect_token_mode(set, schema, runner, oa, rep, err) && err.find("aborted") != std::string::npos);
        CHECK(fs::exists(fs::path(oa.out_dir) / "checkpoint.bin") && !fs::exists(fs::path(oa.out_dir) / "meta.json"));
        oa.fault = nullptr;
        CHECK(!collect_token_mode(set, schema, runner, oa, rep, err) && err.find("--resume") != std::string::npos);
        oa.resume = true;
        CHECK(collect_token_mode(set, schema, runner, oa, rep, err));
        CHECK(rep.resumed && rep.samples_skipped == 12 && rep.samples == (int64_t) set.samples.size() - 12);
        CHECK(stats_files(oa.out_dir, true) == stats_files(g_token_dir, true));
        // resuming a run that read other samples is refused
        CollectOptions ob = options(g_tmp / "token_abort2");
        ob.checkpoint_every = 2;
        ob.fault = [](const char* point, int64_t i, int64_t) { return std::strcmp(point, "sample") == 0 && i == 5; };
        CHECK(!collect_token_mode(set, schema, runner, ob, rep, err));
        CalibSet other = set;
        other.fnv ^= 1;
        ob.fault = nullptr;
        ob.resume = true;
        CHECK(!collect_token_mode(other, schema, runner, ob, rep, err) && err.find("same --prep") != std::string::npos);
    }

    // the memory frame: the accumulators and one sample's routing; one byte less is refused before anything is written
    {
        SampleRoutingSink sink(schema, true);
        uint64_t need = sink.bytes_for(set.longest());
        for (int l = 0; l < L; ++l)
            if (n_expert_of(l) > 0) need += LayerStats::bytes_for(E, K, (int32_t) set.categories.size());
        CollectOptions os = options(g_tmp / "token_small");
        os.ram_budget = need - 1;
        CHECK(!collect_token_mode(set, schema, runner, os, rep, err) && err.find("RAM budget") != std::string::npos);
        CHECK(!fs::exists(fs::path(os.out_dir) / "checkpoint.bin"));
        os.ram_budget = need;
        CHECK(collect_token_mode(set, schema, runner, os, rep, err) && rep.ram_peak <= need);
    }
    std::puts("routing_stats_test: token mode (resume, split runs, the RAM frame) passed");
}

// ================================ 3. THE LAYER-WISE MODE ================================

/// Budgets that force several blocks and small batches: VRAM for a layer, an expert and a little more; RAM for a few
/// samples per block.
LayerwiseConfig small_budgets(const fs::path& scratch) {
    LayerwiseConfig c;
    c.vram_budget = DENSE_BYTES + EXPERT_BYTES + 40 * 3 * H * 4;   // one sequence of up to 40 tokens through the combine
    c.ram_budget = 56 * 1024;
    c.scratch_dir = scratch.string();
    return c;
}

bool same_as_token(const fs::path& dir) { return stats_files(dir, false) == stats_files(g_token_dir, false); }

void test_layerwise(const Mini& m, const CalibSet& set) {
    const RoutingStatsSchema schema = schema_for(m, set);
    std::string err;
    CollectReport rep;
    LayerwiseCounters cnt;
    const fs::path dir = g_tmp / "lw";
    LayerwiseConfig c = small_budgets(g_tmp / "lw_scratch");
    LayerwisePlan plan;
    RefSplit probe(m);
    CHECK(plan_layerwise(probe, set, schema, c, plan));
    const int64_t nb = (int64_t) plan.blocks.size();
    CHECK(nb >= 3 && plan.rows_per_batch < set.tokens());   // the budgets really constrain the run
    // a scratch folder the user names keeps whatever else is in it
    fs::create_directories(fs::path(c.scratch_dir) / "sub");
    { std::ofstream(fs::path(c.scratch_dir) / "keep.txt") << "mine"; }
    { std::ofstream(fs::path(c.scratch_dir) / "sub" / "inner.txt") << "mine"; }
    { std::ofstream(fs::path(c.scratch_dir) / "block_000000_layer_001.rows.bak") << "mine"; }
    {
        RefSplit rm(m);
        if (!collect_layerwise(rm, set, schema, c, options(dir, "layerwise"), rep, &cnt, err)) {
            std::fprintf(stderr, "  %s\n", err.c_str());
            ++g_fail;
            return;
        }
        // the same statistics as the token path, byte for byte
        CHECK(same_as_token(dir));
        // the budgets were kept (the scheduler's buffers and the model's "device" buffers)
        CHECK(rep.vram_peak > 0 && rep.vram_peak <= c.vram_budget && rep.ram_peak > 0 && rep.ram_peak <= c.ram_budget);
        // every layer once per block, from the drive once per block; every expert at most once per block and layer,
        // exactly the experts the block routed to
        for (int64_t b = 0; b < nb; ++b)
            for (int64_t l = 0; l < L; ++l) CHECK(cnt.layer_loads[(size_t) (b * L + l)] == 1);
        for (int64_t l = 0; l < L; ++l) CHECK(rm.dense_reads[(size_t) l] == nb);
        CHECK((int64_t) rm.loads.size() == nb * L && (int64_t) rm.routed.size() == nb * L);
        for (int64_t i = 0; i < nb * L; ++i) {
            const int64_t l = i % L;
            std::set<int32_t> loaded;
            for (int32_t e = 0; e < E; ++e) {
                const int32_t n = cnt.expert_loads[(size_t) (i * cnt.max_expert + e)];
                CHECK(n == 0 || n == 1);
                if (n == 1) loaded.insert(e);
            }
            CHECK(loaded == rm.routed[(size_t) i]);
            if (n_expert_of(l) == 0) CHECK(loaded.empty());
        }
        for (int64_t l = 0; l < L; ++l)
            for (int32_t e = 0; e < E; ++e) CHECK(rm.expert_reads[(size_t) (l * E + e)] <= nb);
        // at most two boundaries on the drive at once; the scratch folder is gone after the run
        CHECK(cnt.boundary_files_max >= 1 && cnt.boundary_files_max <= 2);
        CHECK(!fs::exists(dir / "progress.bin") && fs::exists(dir / "meta.json"));
        CHECK(fs::exists(fs::path(c.scratch_dir) / "keep.txt") && fs::exists(fs::path(c.scratch_dir) / "sub" / "inner.txt") &&
              fs::exists(fs::path(c.scratch_dir) / "block_000000_layer_001.rows.bak"));
        int left = 0;
        for (const auto& e : fs::directory_iterator(c.scratch_dir)) left += e.path().filename().string().rfind("strata-layerwise-", 0) == 0;
        CHECK(left == 0);   // the run's own folder is gone
        CHECK(rep.scratch_peak <= plan.scratch_need);
        std::printf("routing_stats_test: layer-wise run: %lld blocks of up to %lld tokens, batches of %lld rows, VRAM peak "
                    "%llu of %llu B, RAM peak %llu of %llu B, scratch peak %llu B\n", (long long) nb,
                    (long long) plan.block_tokens, (long long) plan.rows_per_batch, (unsigned long long) rep.vram_peak,
                    (unsigned long long) c.vram_budget, (unsigned long long) rep.ram_peak,
                    (unsigned long long) c.ram_budget, (unsigned long long) rep.scratch_peak);
    }
    // budgets too small: refused up front, with the numbers
    {
        RefSplit rm(m);
        LayerwiseConfig t = small_budgets(g_tmp / "lw_small_scratch");
        t.vram_budget = DENSE_BYTES + EXPERT_BYTES / 2;
        CHECK(!collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_small"), rep, nullptr, err) &&
              err.find("VRAM budget") != std::string::npos);
        t = small_budgets(g_tmp / "lw_small_scratch");
        t.ram_budget = 8 * 1024;
        CHECK(!collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_small2"), rep, nullptr, err) &&
              err.find("RAM budget") != std::string::npos);
        t = small_budgets(g_tmp / "lw_small_scratch");
        t.block_tokens = 100000;
        CHECK(!collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_small3"), rep, nullptr, err) &&
              err.find("budgets hold blocks of") != std::string::npos);
        t = small_budgets(g_tmp / "lw_small_scratch");
        t.scratch_max = 64;
        CHECK(!collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_small4"), rep, nullptr, err));
    }
    std::puts("routing_stats_test: layer-wise = token path, budgets kept, each layer once per block passed");

    // ---- an abort at every point of a layer, in a middle block and the last layer: the resumed run continues there
    const char* points[] = {"after_route", "mid_experts", "after_boundary", "after_stats", "after_progress"};
    const std::pair<int64_t, int64_t> where[] = {{1, 1}, {nb - 1, L - 1}, {0, 0}, {2, DENSE_LAYER}};
    int n_case = 0;
    for (const char* pt : points)
        for (const auto& [ab, al] : where) {
            if ((std::strcmp(pt, "after_route") == 0 || std::strcmp(pt, "mid_experts") == 0) && n_expert_of(al) == 0)
                continue;   // a dense layer routes nothing and has no experts
            const fs::path d = g_tmp / ("lw_abort_" + std::to_string(n_case++));
            LayerwiseConfig t = small_budgets(d / "scratch");
            RefSplit first(m);
            LayerwiseCounters c1, c2;
            CollectOptions oa = options(d / "out", "layerwise");
            const std::string want = pt;
            oa.fault = [&](const char* p, int64_t b, int64_t l) { return want == p && b == ab && l == al; };
            CHECK(!collect_layerwise(first, set, schema, t, oa, rep, &c1, err) && err.find("aborted") != std::string::npos);
            oa.fault = nullptr;
            CHECK(!collect_layerwise(first, set, schema, t, oa, rep, nullptr, err) && err.find("--resume") != std::string::npos);
            oa.resume = true;
            RefSplit second(m);
            if (!collect_layerwise(second, set, schema, t, oa, rep, &c2, err)) {
                std::fprintf(stderr, "  resume after %s at block %lld layer %lld: %s\n", pt, (long long) ab, (long long) al,
                             err.c_str());
                ++g_fail;
                continue;
            }
            CHECK(rep.resumed);
            CHECK(same_as_token(d / "out"));
            // it continued where it stopped: the layer in flight again (or the next one, once its progress was written),
            // nothing before it
            const bool next = want == "after_progress";
            const int64_t at = ab * L + al + (next ? 1 : 0);
            if (at < nb * L) CHECK(!second.loads.empty() && second.loads.front() == at % L);
            else CHECK(second.loads.empty());   // only the end was left: the files of a complete run
            for (int64_t i = 0; i < nb * L; ++i) {
                const int64_t total = c1.layer_loads[(size_t) i] + c2.layer_loads[(size_t) i];
                CHECK(total == (i == ab * L + al && !next ? 2 : 1));
            }
        }
    // resuming with other blocks is refused
    {
        const fs::path d = g_tmp / "lw_abort_blocks";
        LayerwiseConfig t = small_budgets(d / "scratch");
        RefSplit rm(m);
        CollectOptions oa = options(d / "out", "layerwise");
        oa.fault = [](const char* p, int64_t b, int64_t) { return std::strcmp(p, "mid_experts") == 0 && b == 1; };
        CHECK(!collect_layerwise(rm, set, schema, t, oa, rep, nullptr, err));
        oa.fault = nullptr;
        oa.resume = true;
        t.block_tokens = 50;
        RefSplit rm2(m);
        CHECK(!collect_layerwise(rm2, set, schema, t, oa, rep, nullptr, err) && err.find("other blocks") != std::string::npos);
    }
    std::printf("routing_stats_test: layer-wise resume after %d simulated aborts (mid-layer included) passed\n", n_case);

    // ---- the fused form (the engine's prompt path over one layer): the same statistics, one forward per sequence
    {
        RefFused rf(m);
        LayerwiseConfig t = small_budgets(g_tmp / "lw_fused_scratch");
        LayerwiseCounters cf;
        CHECK(collect_layerwise(rf, set, schema, t, options(g_tmp / "lw_fused", "layerwise"), rep, &cf, err));
        CHECK(same_as_token(g_tmp / "lw_fused"));
        CHECK(rf.forwards == (int64_t) set.samples.size() * L);
        for (int64_t v : cf.layer_loads) CHECK(v == 1);
        CHECK(cf.boundary_files_max <= 2);
    }
    // ---- the scratch folder is the output folder: the statistics stay
    {
        RefSplit rm(m);
        LayerwiseConfig t = small_budgets(g_tmp / "lw_same");
        t.scratch_dir = (g_tmp / "lw_same").string();
        CHECK(collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_same", "layerwise"), rep, nullptr, err));
        CHECK(same_as_token(g_tmp / "lw_same") && fs::exists(g_tmp / "lw_same" / "meta.json"));
    }
    // ---- the input of the layer to continue with is gone (a power loss undid its rename): the block starts again at
    //      layer 0, and the layers that already counted it do not count it twice
    {
        const fs::path d = g_tmp / "lw_lost";
        LayerwiseConfig t = small_budgets(d / "scratch");
        RefSplit first(m), second(m);
        CollectOptions oa = options(d / "out", "layerwise");
        oa.fault = [](const char* p, int64_t b, int64_t l) { return std::strcmp(p, "after_progress") == 0 && b == 1 && l == 1; };
        CHECK(!collect_layerwise(first, set, schema, t, oa, rep, nullptr, err));
        int removed = 0;
        for (const auto& e : fs::recursive_directory_iterator(d / "scratch"))
            if (e.path().filename().string() == "block_000001_layer_002.rows") removed += fs::remove(e.path()) ? 1 : 0;
        CHECK(removed == 1);
        oa.fault = nullptr;
        oa.resume = true;
        CHECK(collect_layerwise(second, set, schema, t, oa, rep, nullptr, err));
        CHECK(same_as_token(d / "out"));
        CHECK(!second.loads.empty() && second.loads.front() == 0);
    }
    // ---- a read-ahead that fails stops the run with its reason (and nothing writes into freed memory)
    {
        RefSplit rm(m);
        rm.fail_prefetch = 2;
        LayerwiseConfig t = small_budgets(g_tmp / "lw_pf_fail_scratch");
        t.prefetch = true;
        t.ram_budget += DENSE_BYTES;
        CHECK(!collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_pf_fail", "layerwise"), rep, nullptr, err) &&
              err.find("reading layer 2 ahead") != std::string::npos);
    }
    // ---- the read-ahead: the next layer read into RAM while this one computes; still once per block from the drive
    {
        RefSplit rm(m);
        LayerwiseConfig t = small_budgets(g_tmp / "lw_pf_scratch");
        t.prefetch = true;
        t.ram_budget += DENSE_BYTES;
        LayerwiseCounters cp;
        CHECK(collect_layerwise(rm, set, schema, t, options(g_tmp / "lw_pf", "layerwise"), rep, &cp, err));
        CHECK(same_as_token(g_tmp / "lw_pf"));
        CHECK(cp.prefetched > 0 && rep.ram_peak <= t.ram_budget);
        for (int64_t l = 0; l < L; ++l) CHECK(rm.dense_reads[(size_t) l] == (int64_t) (cp.layer_loads.size() / L));
    }
    // ---- 16-bit scratch files: the run completes, every token is counted, the routing barely moves
    for (SpillPrecision p : {SpillPrecision::BF16, SpillPrecision::F16}) {
        RefSplit rm(m);
        LayerwiseConfig t = small_budgets(g_tmp / "lw_16_scratch");
        t.spill = p;
        const fs::path d = g_tmp / (std::string("lw_") + spill_precision_name(p));
        CHECK(collect_layerwise(rm, set, schema, t, options(d, "layerwise"), rep, nullptr, err));
        uint64_t diff = 0, all = 0;
        for (int l = 0; l < L; ++l) {
            if (n_expert_of(l) == 0) continue;
            LayerStats a, b;
            CHECK(read_layer_stats(d.string(), l, schema, a, err) && read_layer_stats(g_token_dir.string(), l, schema, b, err));
            for (int cc = 0; cc < (int) set.categories.size(); ++cc) {
                CHECK(a.tokens(cc) == b.tokens(cc));
                for (int e = 0; e < E; ++e) {
                    const uint64_t x = a.at(cc, e, LayerStats::kCount), y = b.at(cc, e, LayerStats::kCount);
                    diff += x > y ? x - y : y - x;
                    all += y;
                }
            }
        }
        CHECK(all > 0 && (double) diff / (double) all < 0.05);
    }
    // the half and bfloat16 conversions round to nearest even and keep 0, sign, inf
    CHECK(half_to_float(float_to_half(1.0f)) == 1.0f && half_to_float(float_to_half(-2.5f)) == -2.5f);
    CHECK(float_to_half(65520.0f) == 0x7c00 && std::isinf(half_to_float(0x7c00)) && half_to_float(float_to_half(0.0f)) == 0.0f);
    CHECK(std::fabs(half_to_float(float_to_half(1e-6f)) - 1e-6f) < 1e-7f);
    CHECK(bf16_to_float(float_to_bf16(1.0f)) == 1.0f && float_to_bf16(1.00390625f) == 0x3f80);   // ties to even
    std::puts("routing_stats_test: fused form, read-ahead and 16-bit scratch files passed");
}

// ================================ 4. THE SIMILARITY DATA (spec 4.4b) ================================

constexpr double kSimFraction = 0.25;
constexpr uint64_t kSimSeed = 7;
fs::path g_sim_dir;   // token mode with the similarity data on the twins model: the reference

RoutingStatsSchema sim_schema(const Mini& m, const CalibSet& set) {
    RoutingStatsSchema s = schema_for(m, set);
    s.similarity.fraction = kSimFraction;
    s.similarity.seed = kSimSeed;
    return s;
}

/// The similarity files of a run, by name.
std::map<std::string, std::string> sim_files(const fs::path& dir) {
    std::map<std::string, std::string> out;
    for (int l = 0; l < L; ++l)
        if (n_expert_of(l) > 0) out[similarity_file_name(l)] = slurp(dir / similarity_file_name(l));
    return out;
}

void test_similarity(const CalibSet& set) {
    const Mini mt(17, true);
    const RoutingStatsSchema schema = sim_schema(mt, set), plain = schema_for(mt, set);
    auto runner = [&](const CalibSample& s, RoutingSink& sink, std::string& err) { return token_forward(mt, s, sink, err); };
    std::string err;
    CollectReport rep;
    // ---- sampling: a fixed share by the sample id and the position, the same on every call
    int64_t sampled = 0;
    for (const CalibSample& s : set.samples) sampled += similarity_sampled_count(schema.similarity, s.id, (int64_t) s.tokens.size());
    CHECK(sampled > set.tokens() / 6 && sampled < set.tokens() / 3);   // about a quarter
    CHECK(similarity_sampled(schema.similarity, "x", 3) == similarity_sampled(schema.similarity, "x", 3));
    SimilarityConfig all{1.0, 1}, none{0.0, 1};
    CHECK(similarity_sampled(all, "x", 0) && !similarity_sampled(none, "x", 0));

    // ---- token mode: the reference.  The statistics are the same as without the similarity data
    g_sim_dir = g_tmp / "sim_token";
    CollectOptions o = options(g_sim_dir);
    o.checkpoint_every = 5;
    if (!collect_token_mode(set, schema, runner, o, rep, err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
        return;
    }
    CHECK(rep.similarity_tokens == sampled);
    CollectOptions op = options(g_tmp / "sim_plain");
    CHECK(collect_token_mode(set, plain, runner, op, rep, err));
    CHECK(stats_files(op.out_dir, false) == stats_files(g_sim_dir, false));
    CHECK(!fs::exists(fs::path(op.out_dir) / similarity_file_name(0)));
    const std::string meta = slurp(g_sim_dir / "meta.json");
    CHECK(meta.find("\"similarity\": {\"fraction\": 0.25, \"seed\": 7") != std::string::npos);
    CHECK(slurp(fs::path(op.out_dir) / "meta.json").find("\"similarity\"") == std::string::npos);

    // ---- the twins: expert 4's output is expert 0's, expert 5's is half of expert 1's; unrelated experts are not alike
    for (int l = 0; l < L; ++l) {
        if (n_expert_of(l) == 0) continue;
        LayerSimilarity s;
        CHECK(read_layer_similarity(g_sim_dir.string(), l, schema, s, err));
        CHECK(s.tokens == (uint64_t) sampled);
        CHECK(s.row_tokens(TWIN_OF_0) > 0 && s.row_tokens(HALF_OF_1) > 0 && s.row_tokens(0) > 0 && s.row_tokens(1) > 0);
        CHECK(std::fabs(s.mean_cos(TWIN_OF_0, 0) - 1.0) < 1e-6 && s.mean_rel(TWIN_OF_0, 0) < 1e-6);
        CHECK(std::fabs(s.mean_cos(HALF_OF_1, 1) - 1.0) < 1e-6 && std::fabs(s.mean_rel(HALF_OF_1, 1) - 1.0) < 1e-6);
        CHECK(std::fabs(s.mean_rel(1, HALF_OF_1) - 0.5) < 1e-6);   // relative to the picked expert's output
        CHECK(std::fabs(s.mean_cos(0, 0) - 1.0) < 1e-6 && s.mean_rel(0, 0) == 0.0);
        CHECK(s.mean_cos(TWIN_OF_0, 9) < 0.95 && s.mean_rel(TWIN_OF_0, 9) > 0.1);
        CHECK(std::isnan(s.mean_cos(12, 0)) == (s.row_weight(12) == 0));   // no picks: no mean
        // every pick of a sampled token added its weight to its row
        uint64_t rows = 0;
        for (int e = 0; e < E; ++e) rows += s.row_tokens(e);
        CHECK(rows == (uint64_t) (K * sampled));
    }

    // ---- exact merges: two halves of the samples add up to the whole run, bit for bit
    {
        CalibSet h1 = set, h2 = set;
        h1.samples.resize(set.samples.size() / 2);
        h2.samples.erase(h2.samples.begin(), h2.samples.begin() + (long) (set.samples.size() / 2));
        CollectOptions o1 = options(g_tmp / "sim_half1"), o2 = options(g_tmp / "sim_half2");
        CHECK(collect_token_mode(h1, schema, runner, o1, rep, err) && collect_token_mode(h2, schema, runner, o2, rep, err));
        for (int l = 0; l < L; ++l) {
            if (n_expert_of(l) == 0) continue;
            LayerSimilarity a, b, whole;
            CHECK(read_layer_similarity(o1.out_dir, l, schema, a, err) && read_layer_similarity(o2.out_dir, l, schema, b, err) &&
                  read_layer_similarity(g_sim_dir.string(), l, schema, whole, err));
            LayerSimilarity ab = a, ba = b;
            ab.merge(b);
            ba.merge(a);
            CHECK(ab == whole && ba == whole);
        }
        // a file of another configuration is refused
        RoutingStatsSchema other = schema;
        other.similarity.seed = 8;
        LayerSimilarity x;
        CHECK(!read_layer_similarity(g_sim_dir.string(), 0, other, x, err) && err.find("another sample share") != std::string::npos);
    }

    // ---- token mode resumed after an abort: the same bytes; a resume with another share is refused
    {
        CollectOptions oa = options(g_tmp / "sim_abort");
        oa.checkpoint_every = 4;
        oa.fault = [](const char* point, int64_t i, int64_t) { return std::strcmp(point, "sample") == 0 && i == 13; };
        CHECK(!collect_token_mode(set, schema, runner, oa, rep, err));
        oa.fault = nullptr;
        oa.resume = true;
        RoutingStatsSchema other = schema;
        other.similarity.fraction = 0.5;
        CHECK(!collect_token_mode(set, other, runner, oa, rep, err) && err.find("another share or seed") != std::string::npos);
        CHECK(!collect_token_mode(set, plain, runner, oa, rep, err) && err.find("collected similarity data") != std::string::npos);
        CHECK(collect_token_mode(set, schema, runner, oa, rep, err));
        CHECK(stats_files(oa.out_dir, true) == stats_files(g_sim_dir, true) && sim_files(oa.out_dir) == sim_files(g_sim_dir));
    }

    // ---- the layer-wise split form: the same files, every expert loaded once in a block that samples a token
    {
        RefSplit sm(mt);
        LayerwiseCounters cnt;
        LayerwiseConfig c = small_budgets(g_tmp / "sim_scratch");
        c.ram_budget = 64 * 1024;
        LayerwisePlan plan;
        CHECK(plan_layerwise(sm, set, schema, c, plan));
        CHECK(plan.similarity_rows_max > 0 && plan.similarity_bytes == (uint64_t) plan.similarity_rows_max * E * H * 4);
        CollectOptions ol = options(g_tmp / "sim_layerwise", "layerwise");
        ol.run.mode = "token";   // the same run description as the reference, so meta.json can be compared too
        ol.run.run_params_json = "{\"mode\": \"token\", \"ram_budget\": 0}";
        if (!collect_layerwise(sm, set, schema, c, ol, rep, &cnt, err)) {
            std::fprintf(stderr, "  %s\n", err.c_str());
            ++g_fail;
        }
        CHECK(stats_files(ol.out_dir, true) == stats_files(g_sim_dir, true) && sim_files(ol.out_dir) == sim_files(g_sim_dir));
        CHECK(rep.similarity_tokens == sampled && rep.ram_peak <= c.ram_budget);
        for (size_t b = 0; b < plan.blocks.size(); ++b) {
            int64_t sb = 0;
            for (int64_t i = plan.blocks[b].first; i < plan.blocks[b].second; ++i)
                sb += similarity_sampled_count(schema.similarity, set.samples[(size_t) i].id, (int64_t) set.samples[(size_t) i].tokens.size());
            for (int l = 0; l < L; ++l)
                for (int e = 0; e < n_expert_of(l); ++e) {
                    const int32_t loads = cnt.expert_loads[(size_t) ((b * L + l) * cnt.max_expert + e)];
                    CHECK(sb > 0 ? loads == 1 : loads <= 1);
                }
        }
        // a RAM budget that holds the block but not the sampled tokens' outputs is refused up front, with its reason
        LayerwiseConfig tight = c;
        LayerwisePlan p0;
        RoutingStatsSchema nosim = schema;
        nosim.similarity = SimilarityConfig{};
        CHECK(plan_layerwise(sm, set, nosim, c, p0));
        tight.block_tokens = p0.block_tokens;
        tight.ram_budget = p0.ram_need + 2 * LayerSimilarity::bytes_for(E) + SimilarityToken::bytes_for(E, K) + 1;   // 1 B spare
        LayerwisePlan p1;
        CHECK(!plan_layerwise(sm, set, schema, tight, p1));
        bool said = false;
        for (const std::string& p : p1.problems) said = said || p.find("similarity data") != std::string::npos;
        CHECK(said);
        // resumed after an abort between the statistics and the similarity file, and after it: the same bytes
        for (const char* point : {"after_stats", "after_similarity", "mid_experts"}) {
            RefSplit sr(mt);
            CollectOptions oa = options(g_tmp / (std::string("sim_lw_abort_") + point), "layerwise");
            oa.run = ol.run;
            const std::string pt = point;
            oa.fault = [pt](const char* p, int64_t b, int64_t l) { return pt == p && b == 1 && l == 1; };
            CHECK(!collect_layerwise(sr, set, schema, c, oa, rep, nullptr, err) && err.find("aborted") != std::string::npos);
            oa.fault = nullptr;
            oa.resume = true;
            RefSplit fresh(mt);   // a new process: the aborted one may have left a layer loaded
            if (!collect_layerwise(fresh, set, schema, c, oa, rep, nullptr, err)) {
                std::fprintf(stderr, "  resume after %s: %s\n", point, err.c_str());
                ++g_fail;
            }
            CHECK(stats_files(oa.out_dir, true) == stats_files(g_sim_dir, true) && sim_files(oa.out_dir) == sim_files(g_sim_dir));
        }
    }

    // ---- the fused form (the model hands over the outputs itself): the same files
    {
        RefFused fm(mt);
        CollectOptions of = options(g_tmp / "sim_fused", "layerwise");
        of.run.mode = "token";
        of.run.run_params_json = "{\"mode\": \"token\", \"ram_budget\": 0}";
        LayerwiseConfig c;
        c.scratch_dir = (g_tmp / "sim_fused_scratch").string();
        CHECK(collect_layerwise(fm, set, schema, c, of, rep, nullptr, err));
        CHECK(stats_files(of.out_dir, true) == stats_files(g_sim_dir, true) && sim_files(of.out_dir) == sim_files(g_sim_dir));
    }

    // ---- a path that gives no expert outputs cannot collect similarity data: refused, nothing written
    {
        g_no_outputs = true;
        CollectOptions on = options(g_tmp / "sim_none");
        CHECK(!collect_token_mode(set, schema, runner, on, rep, err) && err.find("does not collect similarity data") != std::string::npos);
        RefFused fm(mt);
        CollectOptions onf = options(g_tmp / "sim_none_fused", "layerwise");
        LayerwiseConfig c;
        c.scratch_dir = (g_tmp / "sim_none_scratch").string();
        CHECK(!collect_layerwise(fm, set, schema, c, onf, rep, nullptr, err) &&
              err.find("does not collect similarity data") != std::string::npos);
        CHECK(!fs::exists(fs::path(on.out_dir) / "meta.json") && !fs::exists(fs::path(onf.out_dir) / "meta.json"));
        g_no_outputs = false;
    }
    std::puts("routing_stats_test: similarity data (twins found, exact merges, resume, layer-wise and fused = token mode, "
              "refusals) passed");
}

// ================================ 5. THE ENGINE'S SHAPE: CHUNKS, ROUTED EXPERTS ONLY ================================

/// The engine's prompt path in miniature: a sample in chunks of `C` positions, each chunk layer by layer (all its rows
/// through a layer before the next), the routed experts only.  The expert input of the rows the collector samples goes
/// to `feeder` after the chunk's routing, as Prefill::on_routing and Prefill::on_moe_rows hand them over; every
/// expert's output on them is the feeder's to compute.  The same per-row math as token_forward, so the same bits.
bool chunk_forward(const Mini& m, const CalibSample& s, ChunkOutputFeeder& feeder, int64_t C, std::string& err) {
    std::vector<std::vector<float>> kc((size_t) L), vc((size_t) L);
    const int64_t n = (int64_t) s.tokens.size();
    for (int64_t p0 = 0; p0 < n; p0 += C) {
        const int64_t T = std::min<int64_t>(C, n - p0);
        std::vector<float> x((size_t) (T * H)), moe_in((size_t) (T * H));
        for (int64_t r = 0; r < T; ++r) std::memcpy(&x[(size_t) (r * H)], &m.embed[(size_t) s.tokens[(size_t) (p0 + r)] * H], H * 4);
        for (int64_t l = 0; l < L; ++l) {
            const float* d = m.dense(l);
            for (int64_t r = 0; r < T; ++r) {
                float q[A], k[A], v[A];
                qkv_row(d, &x[(size_t) (r * H)], q, k, v);
                kc[(size_t) l].insert(kc[(size_t) l].end(), k, k + A);
                vc[(size_t) l].insert(vc[(size_t) l].end(), v, v + A);
                attend_row(d, q, kc[(size_t) l].data(), vc[(size_t) l].data(), p0 + r + 1, &x[(size_t) (r * H)]);
            }
            std::memcpy(moe_in.data(), x.data(), moe_in.size() * 4);
            std::vector<float> moe_out((size_t) (T * H), 0.0f);
            if (n_expert_of(l) > 0) {
                std::vector<int32_t> ids((size_t) (T * K));
                std::vector<float> w((size_t) (T * K));
                for (int64_t r = 0; r < T; ++r) route_row(d, &moe_in[(size_t) (r * H)], &ids[(size_t) (r * K)], &w[(size_t) (r * K)]);
                feeder.routed(l, p0, T, ids.data(), w.data());
                std::vector<int32_t> rows;
                feeder.wanted(l, p0, T, rows);
                std::vector<float> xr(rows.size() * H);   // the download of the wanted rows
                for (size_t i = 0; i < rows.size(); ++i) std::memcpy(&xr[i * H], &moe_in[(size_t) rows[i] * H], H * 4);
                feeder.moe_rows(l, p0, rows.data(), (int64_t) rows.size(), xr.data());
                for (int64_t r = 0; r < T; ++r) {
                    int order[K];
                    for (int j = 0; j < K; ++j) order[j] = j;
                    std::sort(order, order + K, [&](int a, int b) { return ids[(size_t) (r * K + a)] < ids[(size_t) (r * K + b)]; });
                    float y[H];
                    for (int jj = 0; jj < K; ++jj) {
                        const int j = order[jj];
                        ffn_row(m.expert(l, ids[(size_t) (r * K + j)]), &moe_in[(size_t) (r * H)], y);
                        for (int h = 0; h < H; ++h) moe_out[(size_t) (r * H + h)] += w[(size_t) (r * K + j)] * y[h];
                    }
                }
            }
            for (int64_t r = 0; r < T; ++r) combine_row(d, l, &x[(size_t) (r * H)], &moe_in[(size_t) (r * H)], &moe_out[(size_t) (r * H)]);
        }
    }
    return feeder.ok(err);
}

void test_engine_shape(const CalibSet& set) {
    const Mini mt(17, true);
    RoutingStatsSchema schema = sim_schema(mt, set);
    schema.has_contrib = false;   // the engine's prompt path gives no per-pick output norm
    std::string err;
    CollectReport rep;
    auto token_runner = [&](const CalibSample& s, RoutingSink& sink, std::string& e) { return token_forward(mt, s, sink, e); };
    CollectOptions ot = options(g_tmp / "engine_ref");
    CHECK(collect_token_mode(set, schema, token_runner, ot, rep, err));
    // every expert's output on the downloaded rows: what the engine's CPU expert kernels do
    int64_t computed = 0;
    auto outputs = [&](int64_t l, const float* x, int64_t n, float* y, std::string&) {
        for (int64_t i = 0; i < n; ++i) all_expert_outputs(mt, l, x + i * H, y + (size_t) i * E * H);
        computed += n;
        return true;
    };
    for (int64_t C : {1, 5, 64}) {   // the chunk size does not matter
        auto chunk_runner = [&](const CalibSample& s, RoutingSink& sink, std::string& e) {
            ChunkOutputFeeder feeder(sink, K, E, H, outputs);
            return chunk_forward(mt, s, feeder, C, e);
        };
        CollectOptions oc = options(g_tmp / ("engine_chunk_" + std::to_string(C)));
        computed = 0;
        if (!collect_token_mode(set, schema, chunk_runner, oc, rep, err)) {
            std::fprintf(stderr, "  %s\n", err.c_str());
            ++g_fail;
        }
        CHECK(stats_files(oc.out_dir, true) == stats_files(ot.out_dir, true) && sim_files(oc.out_dir) == sim_files(ot.out_dir));
        int64_t moe_layers = 0;
        for (int l = 0; l < L; ++l) moe_layers += n_expert_of(l) > 0;
        CHECK(computed == rep.similarity_tokens * moe_layers);   // only the sampled rows were computed, once per layer
    }
    // a failed expert computation, and expert input without its routing, stop the run with the reason
    {
        auto failing = [&](int64_t, const float*, int64_t, float*, std::string& e) { e = "no CPU kernel"; return false; };
        auto runner = [&](const CalibSample& s, RoutingSink& sink, std::string& e) {
            ChunkOutputFeeder feeder(sink, K, E, H, failing);
            return chunk_forward(mt, s, feeder, 8, e);
        };
        CollectOptions of = options(g_tmp / "engine_fail");
        CHECK(!collect_token_mode(set, schema, runner, of, rep, err) && err.find("no CPU kernel") != std::string::npos);
        struct Null : RoutingSink {
            void routed(int64_t, int64_t, int64_t, const int32_t*, const float*, const float*) override {}
        } null;
        ChunkOutputFeeder f(null, K, E, H, outputs);
        const int32_t row = 0;
        const float x[H] = {};
        f.moe_rows(0, 0, &row, 1, x);
        CHECK(!f.ok(err) && err.find("without its routing") != std::string::npos);
    }
    std::puts("routing_stats_test: the engine's shape (chunks, routed experts only, the sampled rows' outputs computed "
              "apart) = token mode passed");
}

// ================================ THE FIXTURE ================================

/// The similarity fixture (`<fixture>-sim`): the twins model's token-mode statistics with the similarity data, for the
/// Python test's substitute build.
void sim_fixture(const std::string& dir, bool write) {
    const fs::path d(dir);
    if (g_sim_dir.empty() || !fs::exists(g_sim_dir / "meta.json")) return;
    auto files = stats_files(g_sim_dir, true);
    for (auto& [name, bytes] : sim_files(g_sim_dir)) files[name] = bytes;
    if (write) {
        fs::create_directories(d);
        for (const auto& [name, bytes] : files) {
            std::ofstream f(d / name, std::ios::binary);
            f << bytes;
        }
        std::printf("routing_stats_test: wrote the similarity fixture to %s\n", dir.c_str());
        return;
    }
    if (!fs::exists(d / "meta.json")) {
        std::printf("routing_stats_test: %s not found, the similarity fixture was not compared\n", dir.c_str());
        return;
    }
    std::map<std::string, std::string> have;
    for (const auto& [name, bytes] : files) have[name] = slurp(d / name);
    CHECK(have == files);
    std::puts("routing_stats_test: the committed similarity fixture is current");
}

void fixture(const Mini& m, const CalibSet& set, const std::string& dir, bool write) {
    std::string err;
    const fs::path d(dir);
    const RoutingStatsSchema schema = schema_for(m, set);
    if (write) {
        fs::create_directories(d);
        for (const auto& [name, bytes] : stats_files(g_token_dir, true)) {
            std::ofstream f(d / name, std::ios::binary);
            f << bytes;
        }
        CHECK(write_calib_samples((d / "samples.bin").string(), set, err));
        std::printf("routing_stats_test: wrote the fixture to %s\n", dir.c_str());
        return;
    }
    if (!fs::exists(d / "meta.json")) {
        std::printf("routing_stats_test: %s not found, the fixture was not compared\n", dir.c_str());
        return;
    }
    CHECK(stats_files(d, true) == stats_files(g_token_dir, true));
    CalibSet back;
    CHECK(read_calib_samples((d / "samples.bin").string(), back, err));
    CHECK(back.samples.size() == set.samples.size() && back.categories.size() == set.categories.size());
    CHECK(back.samples[3].tokens == set.samples[3].tokens && back.samples[3].cat == set.samples[3].cat);
    std::puts("routing_stats_test: the committed fixture is current");
}

}  // namespace

int main(int argc, char** argv) {
    std::string fixture_dir = "tests/data/expert_whitelist/stats-mini";
    bool write = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--write-fixture" && i + 1 < argc) {
            fixture_dir = argv[++i];
            write = true;
        } else {
            fixture_dir = a;
        }
    }
    g_tmp = fs::temp_directory_path() / ("strata_routing_stats_test_" + std::to_string((long long) std::time(nullptr)));
    fs::remove_all(g_tmp);
    fs::create_directories(g_tmp);
    const Mini m(17);
    CalibSet set = make_samples();
    {   // the sample file round trip (prepare's format)
        std::string err;
        CHECK(write_calib_samples((g_tmp / "samples.bin").string(), set, err));
        CalibSet back;
        CHECK(read_calib_samples((g_tmp / "samples.bin").string(), back, err));
        CHECK(back.tokens() == set.tokens() && back.samples.back().id == set.samples.back().id);
        set.fnv = back.fnv;
    }
    test_merge();
    test_token_mode(m, set);
    if (!g_token_dir.empty() && fs::exists(g_token_dir / "meta.json")) {
        test_layerwise(m, set);
        fixture(m, set, fixture_dir, write);
    }
    test_similarity(set);
    test_engine_shape(set);
    sim_fixture(fixture_dir + "-sim", write);
    fs::remove_all(g_tmp);
    if (g_fail) {
        std::fprintf(stderr, "routing_stats_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::puts("routing_stats_test: passed");
    return 0;
}
