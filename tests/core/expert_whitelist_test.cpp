// tests/core/expert_whitelist_test.cpp - the expert whitelist on a synthetic mini-MoE (no model, no GPU).
//
// The routing filter (`whitelist_route.hpp`) is the same code the CUDA/HIP kernels run per token; this test runs it
// around a host copy of the router's top-k (router_top10.cu's semantics) on a 3-layer, 16-expert, top-2 model with
// real (tiny) expert matmuls, through four paths shaped like the engine's: the prompt (all rows of a layer routed at
// once), generation (one row), a verify window (n + 1 rows) and speculative decoding (a one-layer draft model with its
// own whitelist or none, checked by the verify window).  What it checks:
//
//   1. the manifest: parse, every validation error, the fingerprint (and the pack's index sidecar);
//   2. no whitelist: every path's tokens and logits are bitwise the baseline's; an allow-all whitelist too;
//   3. a whitelist: only allowed experts are selected and executed (the run's counters), in every path;
//   4. the slot map: a non-contiguous whitelist (experts 1, 5, 6, 14) on a compact store gives exactly the output
//      of the full weights with masking;
//   5. loading: the compact store reads only the allowed experts' bytes (a counting reader), nothing else;
//   6. substitute mode: every routed token's result equals the rule recomputed from the model's own top-k (the table's
//      replacement with the weight moved, added weights, the fallback below the similarity threshold), and no blocked
//      expert is loaded;
//   7. tiered mode through the engine's OnDemandCache: on_demand experts are read only when selected, the cache budget
//      holds, a blocked expert is a hard error, and hits, misses, evictions and bytes equal a reference count; the
//      prompt path's copies hold an entry only while they copy it, and threads copying and taking pointers agree;
//   plus the fix-up after an underflowed top-k.
#include "strata/core/expert_whitelist.hpp"
#include "strata/core/whitelist_route.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;

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

constexpr int L = 3, E = 16, K = 2, H = 8, FF = 4, V = 32;
constexpr int BLOB_FLOATS = 3 * FF * H;                 // gate (FF x H) | up (FF x H) | down (H x FF)
constexpr uint64_t BLOB = BLOB_FLOATS * sizeof(float);

struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    float next() {   // [-1, 1)
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return (float) ((double) (s >> 40) / (double) (1ull << 23) - 1.0);
    }
};

/// The weights.  The experts live in a "pack" byte image laid out like experts.bin: layer l at l * E * BLOB.
struct Model {
    int layers = L;
    std::vector<float> router;   // [layer][E][H]
    std::vector<uint8_t> pack;   // [layer][E] blobs
    std::vector<float> embed;    // [V][H]
    std::vector<float> out;      // [V][H]
    explicit Model(uint64_t seed, int n_layers = L) : layers(n_layers) {
        Lcg r(seed);
        router.resize((size_t) layers * E * H);
        for (float& v : router) v = 2.0f * r.next();
        pack.resize((size_t) layers * E * BLOB);
        auto* f = (float*) pack.data();
        for (size_t k = 0; k < (size_t) layers * E * BLOB_FLOATS; ++k) f[k] = 0.5f * r.next();
        embed.resize((size_t) V * H);
        for (float& v : embed) v = r.next();
        out.resize((size_t) V * H);
        for (float& v : out) v = r.next();
    }
    std::vector<uint64_t> layer_offset() const {
        std::vector<uint64_t> o;
        for (int l = 0; l < layers; ++l) o.push_back((uint64_t) l * E * BLOB);
        return o;
    }
    std::vector<uint64_t> blob_bytes() const { return std::vector<uint64_t>((size_t) layers, BLOB); }
    WhitelistModelInfo info() const {
        WhitelistModelInfo m;
        m.n_layers = layers;
        m.experts_per_layer.assign((size_t) layers, E);
        m.top_k = K;
        m.hidden = H;
        m.layer_offset = layer_offset();
        m.blob_bytes = blob_bytes();
        return m;
    }
};

/// The tiered mode's drive: reads one expert's blob from the pack and logs (layer, expert) - what the on_demand cache
/// read, in order.
struct DriveReader : OnDemandCache::Reader {
    const Model* m = nullptr;
    std::vector<std::pair<int, int>> log;
    uint64_t bytes = 0;
    bool read(int64_t l, int64_t e, uint8_t* dst, uint64_t n) override {
        if (n != BLOB) return false;
        std::memcpy(dst, m->pack.data() + ((size_t) l * E + (size_t) e) * BLOB, BLOB);
        log.emplace_back((int) l, (int) e);
        bytes += n;
        return true;
    }
};

/// Where an expert's blob comes from: the full pack, or a compact store with a slot map (and, in tiered mode, the
/// on_demand experts read on first use - through the engine's OnDemandCache when `cache` is set).  Every execution
/// is counted.
struct Store {
    const Model* m = nullptr;
    const ExpertWhitelist* wl = nullptr;     // null: the full pack
    CompactExpertPlan plan;
    std::vector<uint8_t> compact;
    std::map<std::pair<int, int>, std::vector<uint8_t>> on_demand;   // tiered: loaded when routed
    uint64_t on_demand_reads = 0;
    OnDemandCache* cache = nullptr;          // tiered: the engine's cache instead of the map above
    DriveReader* drive = nullptr;
    std::vector<std::pair<int, int>> asked;  // every blob asked for, in order
    WhitelistStats* stats = nullptr;
    bool failed = false;

    void begin_layer(int l) { if (cache) cache->begin_layer(l); }
    const float* blob(int l, int e) {
        if (stats) stats->executed(l, e);
        asked.emplace_back(l, e);
        if (wl == nullptr) return (const float*) (m->pack.data() + ((size_t) l * E + (size_t) e) * BLOB);
        if (!wl->loadable(l, e)) {   // the engine's hard error: never read
            whitelist_note_violation(l, e, "the mini-MoE store");
            failed = true;
            return nullptr;
        }
        const int32_t s = plan.map.slot(l, e);
        if (s >= 0) return (const float*) (compact.data() + plan.dst_off[(size_t) s]);
        if (cache != nullptr) {   // the engine's on_demand tier
            const uint8_t* b = cache->get(l, e, *drive);
            if (b == nullptr) failed = true;
            return (const float*) b;
        }
        auto it = on_demand.find({l, e});
        if (it == on_demand.end()) {   // an on_demand miss: read it now (the drive)
            std::vector<uint8_t> b(BLOB);
            std::memcpy(b.data(), m->pack.data() + ((size_t) l * E + (size_t) e) * BLOB, BLOB);
            on_demand_reads += BLOB;
            if (stats) stats->on_demand_miss(BLOB);
            it = on_demand.emplace(std::make_pair(l, e), std::move(b)).first;
        } else if (stats) {
            stats->on_demand_hit();
        }
        return (const float*) it->second.data();
    }
};

/// A counting reader over the pack image: every (offset, length) it was asked for.
struct CountingReader : BlobReader {
    const std::vector<uint8_t>* pack = nullptr;
    std::vector<std::pair<uint64_t, uint64_t>> reads;
    uint64_t bytes = 0;
    bool read(uint64_t off, uint64_t n, uint8_t* dst) override {
        if (off + n > pack->size()) return false;
        reads.emplace_back(off, n);
        bytes += n;
        std::memcpy(dst, pack->data() + off, (size_t) n);
        return true;
    }
};

/// router_top10.cu's semantics: p = softmax (float max, double exp and sum, p stored as float), top-k by p with
/// strict greater and ties to the lower index, w = p / max(sum, 2^-14) in double.
void topk_ref(const float* logits, int n, int k, int32_t* ids, float* w) {
    float mx = -INFINITY;
    for (int e = 0; e < n; ++e) mx = std::max(mx, logits[e]);
    std::vector<double> ex((size_t) n);
    double z = 0.0;
    for (int e = 0; e < n; ++e) { ex[(size_t) e] = std::exp((double) logits[e] - (double) mx); z += ex[(size_t) e]; }
    std::vector<float> p((size_t) n);
    for (int e = 0; e < n; ++e) p[(size_t) e] = (float) (ex[(size_t) e] / z);
    std::vector<uint8_t> taken((size_t) n, 0);
    double s = 0.0;
    for (int j = 0; j < k; ++j) {
        int bi = -1;
        for (int e = 0; e < n; ++e)
            if (!taken[(size_t) e] && (bi < 0 || p[(size_t) e] > p[(size_t) bi])) bi = e;
        taken[(size_t) bi] = 1;
        ids[j] = bi;
        s += p[(size_t) bi];
    }
    s = std::max(s, 6.103515625e-5);
    for (int j = 0; j < k; ++j) w[j] = (float) ((double) p[(size_t) ids[j]] / s);
}

/// One routed token as the router saw it: the model's own logits, the model's own top-k (no whitelist) and what the
/// filter made of it.  The substitute and tiered tests recompute the expected result from these with their own rules.
struct RouteRec {
    int layer = 0;
    float logits[E] = {};
    int32_t raw_ids[K] = {}, ids[K] = {};
    float raw_w[K] = {}, w[K] = {};
};
std::vector<RouteRec>* g_trace = nullptr;   // set by a test: every routed token is recorded

/// One router call on `rows` rows of layer `l` - the engine's shape: GEMV, then (whitelist) the mask, the top-k,
/// then (whitelist) the fix.  `wl` null: the filter is not called at all (the default path).
void route_rows(const Model& m, const ExpertWhitelist* wl, WhitelistStats* stats, int l, const float* x, int rows,
                int32_t* ids, float* w) {
    std::vector<float> logits((size_t) E);
    std::vector<uint8_t> routable = wl ? wl->routable_row(l) : std::vector<uint8_t>();
    const int32_t* sub = wl && !wl->substitute.empty() ? wl->substitute[(size_t) l].data() : nullptr;
    for (int t = 0; t < rows; ++t) {
        for (int e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (int h = 0; h < H; ++h) acc += m.router[((size_t) l * E + (size_t) e) * H + (size_t) h] * x[(size_t) t * H + (size_t) h];
            logits[(size_t) e] = acc;
        }
        RouteRec rec;
        if (g_trace) {
            rec.layer = l;
            std::memcpy(rec.logits, logits.data(), sizeof rec.logits);
            topk_ref(logits.data(), E, K, rec.raw_ids, rec.raw_w);
        }
        if (wl) wlroute::before_topk((int) wl->mode, logits.data(), routable.data(), E);
        topk_ref(logits.data(), E, K, ids + t * K, w + t * K);
        if (wl) wlroute::after_topk((int) wl->mode, logits.data(), routable.data(), sub, E, ids + t * K, w + t * K, K);
        if (stats)
            for (int j = 0; j < K; ++j) stats->selected(l, ids[t * K + j]);
        if (g_trace) {
            std::memcpy(rec.ids, ids + t * K, sizeof rec.ids);
            std::memcpy(rec.w, w + t * K, sizeof rec.w);
            g_trace->push_back(rec);
        }
    }
}

float silu(float v) { return v / (1.0f + std::exp(-v)); }

/// One expert on one row: down(silu(gate x) * up x).
void expert(const float* blob, const float* x, float* y) {
    const float* gate = blob;
    const float* up = blob + FF * H;
    const float* down = blob + 2 * FF * H;
    float hbuf[FF];
    for (int f = 0; f < FF; ++f) {
        float g = 0.0f, u = 0.0f;
        for (int h = 0; h < H; ++h) { g += gate[f * H + h] * x[h]; u += up[f * H + h] * x[h]; }
        hbuf[f] = silu(g) * u;
    }
    for (int h = 0; h < H; ++h) {
        float acc = 0.0f;
        for (int f = 0; f < FF; ++f) acc += down[h * FF + f] * hbuf[f];
        y[h] = acc;
    }
}

/// `rows` rows through every layer (layer-major, as the prompt path and the verify window run), then the head.
/// Returns false when the store refused an expert.
bool forward(const Model& m, Store& st, const ExpertWhitelist* wl, WhitelistStats* stats, std::vector<float>& x, int rows,
             std::vector<float>& logits) {
    std::vector<int32_t> ids((size_t) rows * K);
    std::vector<float> w((size_t) rows * K), y(H);
    for (int l = 0; l < m.layers; ++l) {
        route_rows(m, wl, stats, l, x.data(), rows, ids.data(), w.data());
        st.begin_layer(l);
        for (int t = 0; t < rows; ++t) {
            float acc[H] = {};
            for (int j = 0; j < K; ++j) {
                const float* b = st.blob(l, ids[(size_t) (t * K + j)]);
                if (b == nullptr) return false;
                expert(b, x.data() + (size_t) t * H, y.data());
                for (int h = 0; h < H; ++h) acc[h] += w[(size_t) (t * K + j)] * y[(size_t) h];
            }
            for (int h = 0; h < H; ++h) x[(size_t) t * H + h] = std::tanh(x[(size_t) t * H + h] + acc[h]);
        }
    }
    logits.assign((size_t) rows * V, 0.0f);
    for (int t = 0; t < rows; ++t)
        for (int v = 0; v < V; ++v) {
            float acc = 0.0f;
            for (int h = 0; h < H; ++h) acc += m.out[(size_t) v * H + h] * x[(size_t) t * H + h];
            logits[(size_t) t * V + v] = acc;
        }
    return true;
}

/// The input of position t: its token's embedding plus a quarter of the previous one's (a tiny causal context).
void input_rows(const Model& m, const std::vector<int>& toks, int from, int rows, std::vector<float>& x) {
    x.assign((size_t) rows * H, 0.0f);
    for (int t = 0; t < rows; ++t) {
        const int p = from + t;
        for (int h = 0; h < H; ++h) {
            float v = m.embed[(size_t) toks[(size_t) p] * H + h];
            if (p > 0) v += 0.25f * m.embed[(size_t) toks[(size_t) p - 1] * H + h];
            x[(size_t) t * H + h] = v;
        }
    }
}

int argmax(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[b]) b = i;
    return b;
}

struct Run {
    std::vector<int> tokens;          // generated
    std::vector<float> logits;        // every scored row, in order
    bool ok = true;
};

/// The prompt (all rows at once), then `n` tokens one at a time (generation).
Run run_generate(const Model& m, Store& st, const ExpertWhitelist* wl, WhitelistStats* stats, std::vector<int> toks, int n) {
    Run r;
    std::vector<float> x, lg;
    input_rows(m, toks, 0, (int) toks.size(), x);
    if (!forward(m, st, wl, stats, x, (int) toks.size(), lg)) { r.ok = false; return r; }
    r.logits.insert(r.logits.end(), lg.end() - V, lg.end());
    int next = argmax(lg.data() + lg.size() - V, V);
    for (int i = 0; i < n; ++i) {
        r.tokens.push_back(next);
        toks.push_back(next);
        input_rows(m, toks, (int) toks.size() - 1, 1, x);
        if (!forward(m, st, wl, stats, x, 1, lg)) { r.ok = false; return r; }
        r.logits.insert(r.logits.end(), lg.begin(), lg.end());
        next = argmax(lg.data(), V);
    }
    return r;
}

/// Speculative decoding: a one-layer draft model proposes up to 3 tokens (one at a time, its own store and
/// whitelist), the main model checks them in one verify window of up to 4 rows and keeps the longest agreeing
/// prefix plus its own next token.  Greedy, so the output must equal `run_generate`'s.
Run run_speculative(const Model& m, Store& st, const ExpertWhitelist* wl, WhitelistStats* stats, const Model& dm,
                    Store& dst, const ExpertWhitelist* dwl, WhitelistStats* dstats, std::vector<int> toks, int n) {
    Run r;
    std::vector<float> x, lg;
    input_rows(m, toks, 0, (int) toks.size(), x);
    if (!forward(m, st, wl, stats, x, (int) toks.size(), lg)) { r.ok = false; return r; }
    int next = argmax(lg.data() + lg.size() - V, V);
    while ((int) r.tokens.size() < n) {
        r.tokens.push_back(next);
        toks.push_back(next);
        if ((int) r.tokens.size() >= n) break;
        // the draft: up to 3 tokens after `next`
        std::vector<int> draft, dt = toks;
        for (int d = 0; d < 3; ++d) {
            input_rows(dm, dt, (int) dt.size() - 1, 1, x);
            if (!forward(dm, dst, dwl, dstats, x, 1, lg)) { r.ok = false; return r; }
            const int g = argmax(lg.data(), V);
            draft.push_back(g);
            dt.push_back(g);
        }
        // the verify window: the row of `next` and the draft rows, in one call
        std::vector<int> vt = toks;
        vt.insert(vt.end(), draft.begin(), draft.end());
        const int rows = 1 + (int) draft.size();
        input_rows(m, vt, (int) toks.size() - 1, rows, x);
        if (!forward(m, st, wl, stats, x, rows, lg)) { r.ok = false; return r; }
        int acc = 0;
        int model_tok = argmax(lg.data(), V);
        while (acc < (int) draft.size() && draft[(size_t) acc] == model_tok && (int) r.tokens.size() < n) {
            r.tokens.push_back(model_tok);
            toks.push_back(model_tok);
            ++acc;
            model_tok = argmax(lg.data() + (size_t) acc * V, V);
        }
        next = model_tok;
    }
    r.tokens.resize((size_t) n);
    return r;
}

// ================================ MANIFESTS ================================

std::string manifest(const Model& m, const std::map<int, std::vector<int>>& allowed, const std::string& extra_top = "",
                     const std::map<int, std::string>& extra_layer = {}, const std::string& fp_override = "") {
    std::string s = "{\n  \"format_version\": 1,\n  \"kind\": \"strata-expert-whitelist\",\n";
    s += "  \"model_fingerprint\": \"" + (fp_override.empty() ? whitelist_fingerprint(m.info()) : fp_override) + "\",\n";
    s += "  \"num_layers\": " + std::to_string(m.layers) + ",\n  \"experts_per_layer\": 16,\n  \"top_k\": 2,\n";
    s += extra_top;
    s += "  \"layers\": [";
    bool first = true;
    for (const auto& [l, ids] : allowed) {
        s += first ? "\n" : ",\n";
        first = false;
        s += "    {\"layer\": " + std::to_string(l) + ", \"allowed\": [";
        for (size_t i = 0; i < ids.size(); ++i) s += (i ? ", " : "") + std::to_string(ids[i]);
        s += "]";
        if (auto it = extra_layer.find(l); it != extra_layer.end()) s += ", " + it->second;
        s += "}";
    }
    s += "\n  ],\n  \"generation\": {\"profile\": \"test\", \"scoring\": {\"method\": \"hand\"}, \"calibration_tokens\": 0, "
         "\"created\": \"2026-10-21\"}\n}\n";
    return s;
}

bool fails_with(const std::string& text, const Model& m, const std::string& needle, const WhitelistMode* mode = nullptr) {
    ExpertWhitelist wl;
    std::string err;
    if (parse_expert_whitelist(text, m.info(), mode, wl, err)) {
        std::fprintf(stderr, "  expected an error containing \"%s\", the manifest was accepted\n", needle.c_str());
        return false;
    }
    if (err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "  expected \"%s\" in: %s\n", needle.c_str(), err.c_str());
        return false;
    }
    return true;
}

ExpertWhitelist parse_ok(const std::string& text, const Model& m, const WhitelistMode* mode = nullptr) {
    ExpertWhitelist wl;
    std::string err;
    if (!parse_expert_whitelist(text, m.info(), mode, wl, err)) {
        std::fprintf(stderr, "  unexpected manifest error: %s\n", err.c_str());
        ++g_fail;
    }
    wl.path = "(test)";
    return wl;
}

void test_manifest() {
    const Model m(1);
    const std::map<int, std::vector<int>> good = {{0, {1, 5, 6, 14}}, {1, {0, 2, 3}}, {2, {7, 8}}};
    const ExpertWhitelist wl = parse_ok(manifest(m, good), m);
    CHECK(wl.active() && wl.mode == WhitelistMode::Strict && wl.top_k == 2 && wl.n_layers == 3);
    CHECK(wl.routable(0, 1) && wl.routable(0, 14) && !wl.routable(0, 2) && !wl.routable(0, 16) && !wl.routable(3, 0));
    CHECK(wl.count(0, kWlResident) == 4 && wl.count(2, kWlResident) == 2 && wl.total(kWlResident) == 9);
    CHECK(wl.profile == "test" && wl.scoring == "hand" && wl.created == "2026-10-21");

    // the fingerprint: stable text and value (tools/test_expert_whitelist.py checks the same value)
    CHECK(whitelist_fingerprint_text(m.info()) ==
          "strata-expert-whitelist-fingerprint-v1\nnum_layers=3\ntop_k=2\nhidden=8\nexperts_per_layer=16,16,16\n"
          "pack_index=0:0:384;1:6144:384;2:12288:384\n");
    CHECK(whitelist_fingerprint(m.info()) == "fnv1a64:25d1f5de79ecddd4");
    WhitelistModelInfo other = m.info();
    other.blob_bytes[1] = 400;   // another pack layout: another fingerprint
    CHECK(whitelist_fingerprint(other) != whitelist_fingerprint(m.info()));

    // every validation error, by its message
    CHECK(fails_with(manifest(m, good, "", {}, "fnv1a64:0000000000000000"), m, "made for another model"));
    {
        std::string t = manifest(m, good);
        t.replace(t.find("\"format_version\": 1"), 19, "\"format_version\": 2");
        CHECK(fails_with(t, m, "unknown format_version 2"));
    }
    CHECK(fails_with(manifest(m, {{0, {1, 2}}, {1, {1, 2}}, {2, {1, 2}}, {3, {1, 2}}}), m, "layer id 3 is out of range"));
    CHECK(fails_with(manifest(m, {{0, {1, 16}}, {1, {1, 2}}, {2, {1, 2}}}), m, "expert id 16 in \"allowed\" is out of range"));
    CHECK(fails_with(manifest(m, {{0, {1, -1}}, {1, {1, 2}}, {2, {1, 2}}}), m, "expert id -1"));
    CHECK(fails_with(manifest(m, {{0, {1, 5, 5}}, {1, {1, 2}}, {2, {1, 2}}}), m, "expert 5 is listed twice"));
    CHECK(fails_with(manifest(m, {{0, {1, 5}}, {2, {1, 2}}}), m, "layer 1 is an MoE layer and is missing"));
    CHECK(fails_with(manifest(m, {{0, {1}}, {1, {1, 2}}, {2, {1, 2}}}), m, "layer 0 allows 1 experts in strict mode, fewer than the model's top_k 2"));
    {
        std::string t = manifest(m, good);
        const size_t at = t.find("{\"layer\": 2");
        t.insert(at, "{\"layer\": 2, \"allowed\": [3, 4]},\n    ");
        CHECK(fails_with(t, m, "layer 2 is listed twice"));
    }
    CHECK(fails_with("{\"format_version\": 1, \"format_version\": 1}", m, "duplicate key"));
    CHECK(fails_with("{\"format_version\": 1,", m, "JSON line 1"));
    CHECK(fails_with("[1, 2]", m, "not a JSON object"));
    {
        std::string t = manifest(m, good);
        t.replace(t.find("\"top_k\": 2"), 10, "\"top_k\": 3");
        CHECK(fails_with(t, m, "\"top_k\" is 3, the model routes 2"));
        t = manifest(m, good);
        t.replace(t.find("\"num_layers\": 3"), 15, "\"num_layers\": 4");
        CHECK(fails_with(t, m, "\"num_layers\" is 4"));
        t = manifest(m, good);
        t.replace(t.find("\"experts_per_layer\": 16"), 23, "\"experts_per_layer\": 8");
        CHECK(fails_with(t, m, "\"experts_per_layer\" is 8"));
        t = manifest(m, good);
        const std::string kind = "\"kind\": \"strata-expert-whitelist\"";
        t.replace(t.find(kind), kind.size(), "\"kind\": \"something-else\"");
        CHECK(fails_with(t, m, "not an expert whitelist"));
    }
    CHECK(fails_with(manifest(m, good, "  \"mode\": \"loose\",\n"), m, "unknown \"mode\""));
    CHECK(fails_with(manifest(m, good, "", {{0, "\"on_demand\": [5]"}}), m, "both \"allowed\" and \"on_demand\""));
    CHECK(fails_with(manifest(m, good, "", {{0, "\"substitutes\": [{\"expert\": 2, \"with\": 3}]"}}), m, "which is not in \"allowed\""));
    CHECK(fails_with(manifest(m, good, "", {{0, "\"substitutes\": [{\"expert\": 5, \"with\": 1}]"}}), m, "is allowed and cannot have a substitute"));
    CHECK(fails_with(manifest(m, good, "", {{0, "\"substitutes\": [{\"expert\": 2, \"with\": 1}, {\"expert\": 2, \"with\": 5}]"}}), m, "two substitutes"));

    // tiered: one allowed + one on_demand routes top-2 in tiered mode, but not once the command line says strict
    {
        const std::string t = manifest(m, {{0, {1}}, {1, {1, 2}}, {2, {1, 2}}}, "  \"mode\": \"tiered\",\n", {{0, "\"on_demand\": [9]"}});
        const ExpertWhitelist tw = parse_ok(t, m);
        CHECK(tw.mode == WhitelistMode::Tiered && tw.routable(0, 9) && !tw.resident(0, 9) && tw.on_demand(0, 9));
        const WhitelistMode strict = WhitelistMode::Strict;
        CHECK(fails_with(t, m, "layer 0 allows 1 experts in strict mode", &strict));
        const ExpertWhitelist sw = parse_ok(manifest(m, good, "  \"mode\": \"tiered\",\n", {{0, "\"on_demand\": [9]"}}), m, &strict);
        CHECK(sw.mode == WhitelistMode::Strict && sw.manifest_mode == WhitelistMode::Tiered && !sw.routable(0, 9));
    }

    // the report and the pair filter
    const std::vector<std::string> rep = whitelist_report(wl);
    CHECK(rep.size() >= 3 && rep[1].find("9 of 48 routed experts allowed") != std::string::npos);
    std::vector<std::pair<int32_t, int32_t>> pairs = {{0, 1}, {0, 2}, {1, 3}, {2, 9}, {2, 8}};
    CHECK(whitelist_filter_pairs(pairs, wl) == 2 && pairs.size() == 3 && pairs[2] == std::make_pair(2, 8));
    std::puts("expert_whitelist_test: manifest checks passed");
}

void test_sidecar() {
    const Model m(1);
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_wl_sidecar_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    bool present = true;
    std::string err;
    CHECK(check_expert_index_sidecar(dir.string(), m.info(), present, err) && !present);   // an old pack: fine
    auto write = [&](uint64_t off1) {
        std::ofstream f(dir / "experts.index.json");
        f << "{\"format_version\": 1, \"kind\": \"strata-expert-index\", \"num_layers\": 3, \"layers\": [";
        for (int l = 0; l < 3; ++l)
            f << (l ? ", " : "") << "{\"layer\": " << l << ", \"n_expert\": 16, \"offset\": " << (l == 1 ? off1 : (uint64_t) l * 6144)
              << ", \"blob_bytes\": 384}";
        f << "]}\n";
    };
    write(6144);
    CHECK(check_expert_index_sidecar(dir.string(), m.info(), present, err) && present);
    write(6145);
    CHECK(!check_expert_index_sidecar(dir.string(), m.info(), present, err) && err.find("stale") != std::string::npos);
    std::filesystem::remove_all(dir);
    std::puts("expert_whitelist_test: index sidecar checks passed");
}

// ================================ THE ROUTING FILTER ================================

void test_filter_rules() {
    // strict fix-up after an underflowed top-k: allowed {0, 3}; expert 3's p underflows to 0 and ties with the masked
    // zeros, so the top-k takes expert 1 (blocked, the lower index) - the fix replaces it with 3
    {
        float lg[4] = {0.0f, 1.0f, 2.0f, -200.0f};
        const uint8_t ok[4] = {1, 0, 0, 1};
        int32_t ids[2];
        float w[2];
        wlroute::mask_row(lg, ok, 4);
        CHECK(std::isinf(lg[1]) && lg[1] < 0 && lg[0] == 0.0f && lg[3] == -200.0f);
        topk_ref(lg, 4, 2, ids, w);
        CHECK(ids[0] == 0 && ids[1] == 1);   // the underflow tie, exactly as the GPU kernels resolve it
        CHECK(wlroute::fix_row(lg, ok, 4, ids, w, 2) == 1);
        CHECK(ids[0] == 0 && ids[1] == 3 && w[0] == 1.0f && w[1] == 0.0f);
        CHECK(wlroute::fix_row(lg, ok, 4, ids, w, 2) == 0);   // idempotent
    }
    // an allow-all mask leaves every logit's bits
    {
        float lg[4] = {-0.0f, 1.5f, -3.25f, 7.0f};
        float copy[4];
        std::memcpy(copy, lg, sizeof lg);
        const uint8_t ok[4] = {1, 1, 1, 1};
        wlroute::mask_row(lg, ok, 4);
        CHECK(std::memcmp(lg, copy, sizeof lg) == 0);
    }
    // substitute: the model's top-3 is {4 (blocked), 2, 6 (blocked)}; 4 -> 5, 6 -> 2 (already chosen: weights add,
    // its rank refilled with the best allowed one not chosen, weight 0)
    {
        const float lg[8] = {0.1f, 0.0f, 2.0f, 0.5f, 3.0f, -1.0f, 1.0f, 0.2f};
        const uint8_t ok[8] = {1, 1, 1, 1, 0, 1, 0, 1};
        const int32_t sub[8] = {-1, -1, -1, -1, 5, -1, 2, -1};
        int32_t ids[3];
        float w[3];
        topk_ref(lg, 8, 3, ids, w);
        CHECK(ids[0] == 4 && ids[1] == 2 && ids[2] == 6);
        const float w0 = w[0], w1 = w[1], w2 = w[2];
        CHECK(wlroute::substitute_row(lg, ok, sub, 8, ids, w, 3) == 2);
        CHECK(ids[0] == 5 && w[0] == w0);                 // the replacement takes the weight along
        CHECK(ids[1] == 2 && w[1] == w1 + w2);            // already chosen: the weights add
        CHECK(ids[2] == 3 && w[2] == 0.0f);               // the freed rank: best allowed not chosen (logit 0.5)
        // no table: each blocked pick falls back to the best allowed expert not chosen, with its weight
        topk_ref(lg, 8, 3, ids, w);
        CHECK(wlroute::substitute_row(lg, ok, nullptr, 8, ids, w, 3) == 2);
        CHECK(ids[0] == 3 && ids[1] == 2 && ids[2] == 7 && w[0] == w0 && w[2] == w2);
        std::set<int> distinct(ids, ids + 3);
        CHECK(distinct.size() == 3);
    }
    std::puts("expert_whitelist_test: routing filter rules passed");
}

// ================================ THE PATHS ================================

Store full_store(const Model& m, WhitelistStats* stats = nullptr) {
    Store s;
    s.m = &m;
    s.stats = stats;
    return s;
}

Store compact_store(const Model& m, const ExpertWhitelist& wl, CountingReader& rd, WhitelistStats* stats) {
    Store s;
    s.m = &m;
    s.wl = &wl;
    s.stats = stats;
    std::string err;
    if (!plan_compact_experts(wl, m.layer_offset(), m.blob_bytes(), false, s.plan, err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
    }
    s.compact.assign((size_t) s.plan.total, 0);
    rd.pack = &m.pack;
    if (!load_compact_experts(s.plan, rd, s.compact.data(), err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
    }
    return s;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void test_paths() {
    const Model m(7), dm(11, 1);   // the main model and a one-layer draft model (the MTP layer's shape)
    const std::vector<int> prompt = {3, 17, 8, 25, 1, 30};
    const int n = 14;

    // ---- 2. no whitelist: the default path, and an allow-all whitelist, are bitwise the baseline
    Store base = full_store(m);
    const Run g0 = run_generate(m, base, nullptr, nullptr, prompt, n);
    Store dbase = full_store(dm);
    const Run s0 = run_speculative(m, base, nullptr, nullptr, dm, dbase, nullptr, nullptr, prompt, n);
    CHECK(g0.ok && s0.ok && g0.tokens == s0.tokens);   // speculative decoding is exact
    {
        std::map<int, std::vector<int>> all;
        for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) all[l].push_back(e);
        const ExpertWhitelist aw = parse_ok(manifest(m, all), m);
        Store as = full_store(m);
        const Run g1 = run_generate(m, as, &aw, nullptr, prompt, n);
        CHECK(g1.tokens == g0.tokens && same_bits(g1.logits, g0.logits));
        // the batch shape does not matter: a prompt routed row by row gives the same bits as all rows at once
        Store bs = full_store(m);
        std::vector<float> x, lg_all, lg_one;
        input_rows(m, prompt, 0, (int) prompt.size(), x);
        CHECK(forward(m, bs, nullptr, nullptr, x, (int) prompt.size(), lg_all));
        for (int t = 0; t < (int) prompt.size(); ++t) {
            std::vector<float> xr, lr;
            input_rows(m, prompt, t, 1, xr);
            CHECK(forward(m, bs, nullptr, nullptr, xr, 1, lr));
            lg_one.insert(lg_one.end(), lr.begin(), lr.end());
        }
        CHECK(same_bits(lg_all, lg_one));
    }

    // ---- 3. a whitelist: only allowed experts selected and executed, in the prompt, generation, verify windows
    //         and speculative decoding (the draft with its own whitelist, and with none)
    const std::map<int, std::vector<int>> allowed = {{0, {1, 5, 6, 14}}, {1, {0, 3, 9, 12, 13}}, {2, {2, 4, 7}}};
    const ExpertWhitelist wl = parse_ok(manifest(m, allowed), m);
    const ExpertWhitelist dwl = parse_ok(manifest(dm, {{0, {0, 2, 4, 6, 8, 10}}}), dm);
    {
        WhitelistStats st;
        st.reset(L, E);
        CountingReader rd;
        Store cs = compact_store(m, wl, rd, &st);
        const Run g = run_generate(m, cs, &wl, &st, prompt, n);
        CHECK(g.ok && !cs.failed);
        CHECK(st.selected_outside(wl) == 0 && st.executed_outside(wl) == 0);
        uint64_t sel = 0;
        for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) sel += st.selected_at(l, e);
        CHECK(sel == (uint64_t) (L * K * (prompt.size() + n)));   // every row of every layer was counted
        // the whitelist did change the model: the baseline uses blocked experts on this prompt
        WhitelistStats st0;
        st0.reset(L, E);
        Store b0 = full_store(m, &st0);
        (void) run_generate(m, b0, nullptr, &st0, prompt, n);
        CHECK(st0.selected_outside(wl) > 0);

        // speculative: main whitelist + the draft's own whitelist; then the draft with none
        WhitelistStats ms, ds;
        ms.reset(L, E);
        ds.reset(1, E);
        CountingReader rd2, drd;
        Store mstore = compact_store(m, wl, rd2, &ms);
        Store dstore = compact_store(dm, dwl, drd, &ds);
        const Run s = run_speculative(m, mstore, &wl, &ms, dm, dstore, &dwl, &ds, prompt, n);
        CHECK(s.ok && s.tokens == g.tokens);   // still exact: the verify window uses the same whitelist
        CHECK(ms.selected_outside(wl) == 0 && ms.executed_outside(wl) == 0);
        CHECK(ds.selected_outside(dwl) == 0 && ds.executed_outside(dwl) == 0);
        WhitelistStats ms2, ds2;
        ms2.reset(L, E);
        ds2.reset(1, E);
        Store mstore2 = compact_store(m, wl, rd2, &ms2);
        Store dfull = full_store(dm, &ds2);
        const Run s2 = run_speculative(m, mstore2, &wl, &ms2, dm, dfull, nullptr, &ds2, prompt, n);
        CHECK(s2.ok && s2.tokens == g.tokens && ms2.selected_outside(wl) == 0);
        CHECK(ds2.selected_outside(dwl) > 0);   // the draft without a whitelist routes over all of its experts

        // the report lines exist and say 0 outside
        const std::vector<std::string> rep = st.report(&wl, 1.0);
        CHECK(!rep.empty() && rep[(size_t) L].find("outside the whitelist 0 selected, 0 executed") != std::string::npos);
    }

    // ---- 4. the slot map: a non-contiguous whitelist on a compact store gives exactly the full weights + masking
    {
        CountingReader rd;
        Store cs = compact_store(m, wl, rd, nullptr);
        std::string err;
        CHECK(cs.plan.map.check(err));
        CHECK(cs.plan.map.slot(0, 1) == 0 && cs.plan.map.slot(0, 5) == 1 && cs.plan.map.slot(0, 6) == 2 &&
              cs.plan.map.slot(0, 14) == 3 && cs.plan.map.slot(0, 2) == -1 && cs.plan.map.slot(1, 0) == 4);
        CHECK(cs.plan.map.id_of[3] == std::make_pair(0, 14));
        Store fs = full_store(m);   // the full pack, masked by the same whitelist
        const Run a = run_generate(m, cs, &wl, nullptr, prompt, n);
        const Run b = run_generate(m, fs, &wl, nullptr, prompt, n);
        CHECK(a.ok && b.ok && a.tokens == b.tokens && same_bits(a.logits, b.logits));
        // a broken table is caught
        ExpertSlotMap bad = cs.plan.map;
        std::swap(bad.id_of[0], bad.id_of[1]);
        CHECK(!bad.check(err));
    }

    // ---- 5. loading: only the allowed experts' bytes are read, in merged runs, nothing blocked
    {
        CountingReader rd;
        Store cs = compact_store(m, wl, rd, nullptr);
        CHECK(rd.bytes == (uint64_t) wl.total(kWlResident) * BLOB && cs.plan.total == rd.bytes);
        // experts 5 and 6 of layer 0 are neighbours: one read; 1 and 14 alone; layer 1's 12 and 13 together
        CHECK(rd.reads.size() == 10);
        for (const auto& [off, len] : rd.reads)
            for (uint64_t b = off; b < off + len; b += BLOB) {
                const int l = (int) (b / (E * BLOB)), e = (int) ((b % (E * BLOB)) / BLOB);
                CHECK(wl.resident(l, e));
            }
        // the engine's hard error: a blocked expert is never read, the request is reported with layer and expert
        whitelist_violation_reset();
        CHECK(cs.blob(0, 2) == nullptr && cs.failed && whitelist_violated());
        CHECK(whitelist_violation().find("expert 2 of layer 0") != std::string::npos);
        whitelist_violation_reset();
    }

    // ---- tiered: the on_demand tier is routed, read on first use (counted), never loaded at start
    {
        const std::string t = manifest(m, {{0, {1, 5}}, {1, {0, 3}}, {2, {2, 4}}}, "  \"mode\": \"tiered\",\n",
                                       {{0, "\"on_demand\": [6, 14]"}, {1, "\"on_demand\": [9]"}});
        const ExpertWhitelist tw = parse_ok(t, m);
        WhitelistStats st;
        st.reset(L, E);
        CountingReader rd;
        Store cs = compact_store(m, tw, rd, &st);
        CHECK(rd.bytes == 6 * BLOB);   // the resident tier only
        const Run g = run_generate(m, cs, &tw, &st, prompt, n);
        CHECK(g.ok && st.selected_outside(tw) == 0 && st.executed_outside(tw) == 0);
        CHECK(cs.on_demand_reads <= 3 * BLOB);   // each on_demand expert is read at most once
        uint64_t od_sel = 0;
        for (auto [l, e] : std::vector<std::pair<int, int>>{{0, 6}, {0, 14}, {1, 9}}) od_sel += st.selected_at(l, e);
        CHECK(od_sel > 0 && cs.on_demand_reads > 0);
    }

    // ---- substitute mode: the model's own top-k, blocked picks replaced; only allowed experts run
    {
        const std::string t = manifest(m, allowed, "  \"mode\": \"substitute\",\n",
                                       {{0, "\"substitutes\": [{\"expert\": 2, \"with\": 1}, {\"expert\": 3, \"with\": 5, \"similarity\": 0.8}]"}});
        const ExpertWhitelist sw = parse_ok(t, m);
        CHECK(sw.mode == WhitelistMode::Substitute && sw.substitutes() == 2);
        WhitelistStats st;
        st.reset(L, E);
        CountingReader rd;
        Store cs = compact_store(m, sw, rd, &st);
        const Run g = run_generate(m, cs, &sw, &st, prompt, n);
        CHECK(g.ok && st.selected_outside(sw) == 0 && st.executed_outside(sw) == 0);
    }
    std::puts("expert_whitelist_test: prompt, generation, verify and speculative paths passed");
}

// ================================ SUBSTITUTE MODE (spec 5.5, test 1) ================================

/// The substitute mode's result for one routed token, recomputed from the model's own top-k with the rule as the
/// documentation states it (not with wlroute::substitute_row): a blocked pick goes to its replacement from the table
/// and takes its router weight along; a replacement that is already chosen gets the two weights added, and the freed
/// rank takes the best allowed expert not chosen with weight 0; a blocked pick without a usable replacement (none, or
/// below the similarity threshold) falls back to the best allowed expert not chosen, which takes its weight.
/// Returns the rule that applied to the token's blocked picks: 1 replaced, 2 added, 4 fell back (or-ed).
int expect_substitute(const RouteRec& r, const ExpertWhitelist& wl, int32_t* ids, float* w) {
    std::memcpy(ids, r.raw_ids, sizeof r.raw_ids);
    std::memcpy(w, r.raw_w, sizeof r.raw_w);
    const std::vector<int32_t>* sub = wl.substitute.empty() ? nullptr : &wl.substitute[(size_t) r.layer];
    auto chosen = [&](int e) { for (int q = 0; q < K; ++q) if (ids[q] == e) return q; return -1; };
    auto best_allowed_free = [&]() {
        int best = -1;
        for (int e = 0; e < E; ++e)
            if (wl.resident(r.layer, e) && chosen(e) < 0 && (best < 0 || r.logits[e] > r.logits[best])) best = e;
        return best;
    };
    int rules = 0;
    for (int j = 0; j < K; ++j) {
        const int e = ids[j];
        if (wl.resident(r.layer, e)) continue;
        const int s = sub != nullptr ? (*sub)[(size_t) e] : -1;
        if (s >= 0) {
            const int at = chosen(s);
            if (at < 0) {
                ids[j] = s;
                rules |= 1;
                continue;
            }
            w[at] += w[j];
            w[j] = 0.0f;
            rules |= 2;
        } else {
            rules |= 4;
        }
        ids[j] = -1;
        ids[j] = best_allowed_free();
    }
    return rules;
}

void test_substitute_mode() {
    const Model m(7);
    const std::vector<int> prompt = {3, 17, 8, 25, 1, 30, 12, 5};
    const int n = 14;
    const std::map<int, std::vector<int>> allowed = {{0, {1, 5, 6, 14}}, {1, {0, 3, 9, 12, 13}}, {2, {2, 4, 7}}};
    // ---- the table, from what the full model routes on this prompt: per layer, a blocked expert that shares a token's
    // top-2 with an allowed one gets that one (the weights must add), the next blocked experts in turn a replacement
    // above the threshold (similarity 0.9), one below it (0.3: never used), one without a similarity (always used) and
    // none (the fallback)
    std::vector<RouteRec> base_trace;
    g_trace = &base_trace;
    Store b0 = full_store(m);
    (void) run_generate(m, b0, nullptr, nullptr, prompt, n);
    g_trace = nullptr;
    std::map<int, std::string> subs;
    std::map<int, std::map<int, std::pair<int, std::string>>> table;   // layer -> blocked -> (with, similarity text)
    for (int l = 0; l < L; ++l) {
        auto is_allowed = [&](int e) { const auto& a = allowed.at(l); return std::find(a.begin(), a.end(), e) != a.end(); };
        std::map<int, int> freq;   // blocked expert -> how often the full model picks it
        for (const RouteRec& r : base_trace)
            if (r.layer == l)
                for (int j = 0; j < K; ++j)
                    if (!is_allowed(r.raw_ids[j])) ++freq[r.raw_ids[j]];
        for (const RouteRec& r : base_trace) {   // the "add" case: a blocked pick beside an allowed one
            if (r.layer != l) continue;
            const int a = r.raw_ids[0], b = r.raw_ids[1];
            if (is_allowed(a) != is_allowed(b)) {
                const int blocked = is_allowed(a) ? b : a, with = is_allowed(a) ? a : b;
                table[l][blocked] = {with, "0.9"};
                break;
            }
        }
        std::vector<std::pair<int, int>> order;   // the other blocked experts, most picked first
        for (const auto& [e, c] : freq)
            if (!table[l].count(e)) order.emplace_back(-c, e);
        std::sort(order.begin(), order.end());
        const char* kinds[] = {"0.9", "0.3", "", "none"};
        for (size_t i = 0; i < order.size() && i < 4; ++i) {
            if (std::string(kinds[i]) == "none") break;
            const int e = order[i].second;
            const int with = allowed.at(l)[(size_t) (e + (int) i) % allowed.at(l).size()];
            table[l][e] = {with, kinds[i]};
        }
        std::string s = "\"substitutes\": [";
        bool first = true;
        for (const auto& [e, ws] : table[l]) {
            s += std::string(first ? "" : ", ") + "{\"expert\": " + std::to_string(e) + ", \"with\": " + std::to_string(ws.first) +
                 (ws.second.empty() ? "" : ", \"similarity\": " + ws.second) + "}";
            first = false;
        }
        subs[l] = s + "]";
    }
    const std::string text = manifest(m, allowed, "  \"mode\": \"substitute\",\n  \"substitute_threshold\": 0.6,\n", subs);
    const ExpertWhitelist sw = parse_ok(text, m);
    CHECK(sw.mode == WhitelistMode::Substitute && sw.has_substitute_threshold && sw.substitute_threshold == 0.6);
    // the table as parsed: 0.3 is below the threshold and not used, a replacement without a similarity is
    int below = 0;
    for (const auto& [l, t] : table)
        for (const auto& [e, ws] : t) {
            const bool used = ws.second != "0.3";
            below += !used;
            CHECK(sw.substitute_listed[(size_t) l][(size_t) e] == ws.first);
            CHECK(sw.substitute[(size_t) l][(size_t) e] == (used ? ws.first : -1));
        }
    CHECK(below > 0 && sw.substitutes_below_threshold() == below);

    // ---- the run: every routed token's result equals the rule's, recomputed from the model's own top-k
    std::vector<RouteRec> trace;
    g_trace = &trace;
    WhitelistStats st;
    st.reset(L, E);
    CountingReader rd;
    Store cs = compact_store(m, sw, rd, &st);
    whitelist_violation_reset();
    const Run g = run_generate(m, cs, &sw, &st, prompt, n);
    g_trace = nullptr;
    CHECK(g.ok && !cs.failed && !whitelist_violated());
    CHECK(trace.size() == (size_t) L * (prompt.size() + n));
    int rules = 0, below_used = 0;
    for (const RouteRec& r : trace) {
        int32_t ids[K];
        float w[K];
        const int rule = expect_substitute(r, sw, ids, w);
        rules |= rule;
        CHECK(std::memcmp(ids, r.ids, sizeof ids) == 0);   // exactly the table's replacement, or the fallback
        CHECK(std::memcmp(w, r.w, sizeof w) == 0);         // the weight moved (or added) bit for bit
        float s0 = 0.0f, s1 = 0.0f;
        for (int j = 0; j < K; ++j) { s0 += r.raw_w[j]; s1 += r.w[j]; }
        CHECK(std::fabs(s0 - s1) <= 1e-6f);                // the router's sum is kept
        for (int j = 0; j < K; ++j) {
            CHECK(sw.resident(r.layer, r.ids[j]));          // only allowed experts are left
            // a pick whose listed replacement is below the threshold took the fallback, not the replacement
            const int e = r.raw_ids[j];
            if (!sw.resident(r.layer, e) && table[r.layer].count(e) && table[r.layer][e].second == "0.3") {
                ++below_used;
                CHECK(rule & 4);
            }
        }
    }
    CHECK((rules & 1) && (rules & 2) && (rules & 4));   // every rule was exercised on this prompt
    CHECK(below_used > 0);                              // and a pick below the threshold fell back
    // blocked experts are never loaded: the store asked for allowed experts only, the reader read the resident tier
    CHECK(st.selected_outside(sw) == 0 && st.executed_outside(sw) == 0);
    CHECK(rd.bytes == (uint64_t) sw.total(kWlResident) * BLOB);
    for (const auto& [l, e] : cs.asked) CHECK(sw.resident(l, e));

    // ---- the threshold from the command line: 0.95 drops the 0.9 replacements too; none keeps every listed one
    ExpertWhitelist ow = parse_ok(text, m);
    const int64_t all = ow.substitutes() + ow.substitutes_below_threshold();
    ow.apply_substitute_threshold(true, 0.95);
    for (const auto& [l, t] : table)
        for (const auto& [e, ws] : t) CHECK(ow.substitute[(size_t) l][(size_t) e] == (ws.second.empty() ? ws.first : -1));
    ow.apply_substitute_threshold(false, 0.0);
    CHECK(ow.substitutes() == all && ow.substitutes_below_threshold() == 0);
    // the manifest's checks of the new fields
    CHECK(fails_with(manifest(m, allowed, "  \"substitute_threshold\": 1.5,\n"), m, "\"substitute_threshold\" must be a number"));
    CHECK(fails_with(manifest(m, allowed, "  \"substitute_threshold\": \"high\",\n"), m, "\"substitute_threshold\" must be a number"));
    CHECK(fails_with(manifest(m, allowed, "", {{0, "\"substitutes\": [{\"expert\": 2, \"with\": 1, \"similarity\": -2}]"}}), m,
                     "has a \"similarity\" that is not a number"));
    std::puts("expert_whitelist_test: substitute mode (replacement, added weights, threshold fallback, nothing blocked loaded) passed");
}

// ================================ TIERED MODE (spec 5.5, test 2) ================================

void test_tiered_cache() {
    const Model m(7);
    const std::vector<int> prompt = {3, 17, 8, 25, 1, 30, 12, 5};
    const int n = 14;
    const std::string text = manifest(m, {{0, {1, 5}}, {1, {0, 3}}, {2, {2, 4}}}, "  \"mode\": \"tiered\",\n",
                                      {{0, "\"on_demand\": [2, 6, 14]"}, {1, "\"on_demand\": [9, 12, 13]"},
                                       {2, "\"on_demand\": [7, 8, 11]"}});
    const ExpertWhitelist tw = parse_ok(text, m);
    std::string err;

    // ---- the run through the engine's cache: a budget of two experts, pure least-recently-used (no pinned epochs)
    std::vector<RouteRec> trace;
    g_trace = &trace;
    WhitelistStats st;
    st.reset(L, E);
    set_whitelist_stats(&st);   // the cache reports to the run's counters, as in the engine
    CountingReader rd;
    Store cs = compact_store(m, tw, rd, &st);
    OnDemandCache cache;
    DriveReader drive;
    drive.m = &m;
    const uint64_t budget = 2 * BLOB;
    CHECK(cache.init(tw, m.blob_bytes(), budget, 0, err));
    cs.cache = &cache;
    cs.drive = &drive;
    whitelist_violation_reset();
    const Run g = run_generate(m, cs, &tw, &st, prompt, n);
    g_trace = nullptr;
    set_whitelist_stats(nullptr);
    CHECK(g.ok && !cs.failed && !whitelist_violated());
    CHECK(rd.bytes == (uint64_t) tw.total(kWlResident) * BLOB);   // at start: the resident tier only
    CHECK(st.selected_outside(tw) == 0 && st.executed_outside(tw) == 0);

    // ---- the reference: the selections in the order the forward runs them (the router's trace), an independent LRU of
    // two entries over the on_demand ones
    std::vector<std::pair<int, int>> sel;
    std::map<std::pair<int, int>, uint64_t> sel_count;
    for (const RouteRec& r : trace)
        for (int j = 0; j < K; ++j) {
            sel.emplace_back(r.layer, r.ids[j]);
            ++sel_count[{r.layer, r.ids[j]}];
        }
    CHECK(cs.asked == sel);   // the store asked for exactly what the router selected, in that order
    std::vector<std::pair<int, int>> lru, ref_reads;
    uint64_t ref_hits = 0, ref_misses = 0, ref_evict = 0;
    for (const auto& p : sel) {
        if (!tw.on_demand(p.first, p.second)) continue;
        const auto it = std::find(lru.begin(), lru.end(), p);
        if (it != lru.end()) {
            ++ref_hits;
            lru.erase(it);
        } else {
            ++ref_misses;
            ref_reads.push_back(p);
            if (lru.size() == 2) { lru.erase(lru.begin()); ++ref_evict; }
        }
        lru.push_back(p);
    }
    CHECK(ref_hits > 0 && ref_misses > 0 && ref_evict > 0);   // the prompt exercises all three
    // on_demand experts were read only when selected, exactly the reference's misses, in order
    CHECK(drive.log == ref_reads);
    CHECK(cache.hits() == ref_hits && cache.misses() == ref_misses && cache.evictions() == ref_evict);
    CHECK(cache.bytes_read() == ref_misses * BLOB && drive.bytes == ref_misses * BLOB);
    CHECK(st.on_demand_hits() == ref_hits && st.on_demand_misses() == ref_misses && st.on_demand_bytes() == ref_misses * BLOB);
    CHECK(st.on_demand_evictions() == ref_evict && st.on_demand_without_room() == 0);
    // the budget held throughout
    CHECK(cache.peak() <= budget && cache.used() <= budget && cache.entries() <= 2);
    for (const auto& [p, c] : sel_count) CHECK(st.selected_at(p.first, p.second) == c);

    // ---- a blocked expert is still the run's hard error, never a read; a resident one is not the cache's
    OnDemandCache::Got got = OnDemandCache::Got::Hit;
    const size_t reads = drive.log.size();
    CHECK(cache.get(0, 3, drive, &got) == nullptr && got == OnDemandCache::Got::Refused && whitelist_violated());
    CHECK(whitelist_violation().find("expert 3 of layer 0") != std::string::npos);
    whitelist_violation_reset();
    CHECK(cache.get(0, 1, drive, &got) == nullptr && got == OnDemandCache::Got::Refused && !whitelist_violated());
    CHECK(drive.log.size() == reads);
    {   // the store's path: a blocked request fails the run with its layer and expert
        Store bs = compact_store(m, tw, rd, nullptr);
        bs.cache = &cache;
        bs.drive = &drive;
        CHECK(bs.blob(2, 0) == nullptr && bs.failed && whitelist_violation().find("expert 0 of layer 2") != std::string::npos);
        whitelist_violation_reset();
    }

    // ---- pinned epochs: with one, an expert asked for in this layer is not evicted - the next miss finds no room and
    // is not read (the caller reads it another way); the next layer frees it
    {
        OnDemandCache c1;
        DriveReader d1;
        d1.m = &m;
        CHECK(c1.init(tw, m.blob_bytes(), BLOB, 1, err));
        c1.begin_layer(0);
        CHECK(c1.get(0, 6, d1, &got) != nullptr && got == OnDemandCache::Got::Miss);
        CHECK(c1.get(0, 14, d1, &got) == nullptr && got == OnDemandCache::Got::NoRoom && c1.used() == BLOB);
        CHECK(c1.get(0, 6, d1, &got) != nullptr && got == OnDemandCache::Got::Hit);
        c1.begin_layer(1);
        CHECK(c1.get(0, 14, d1, &got) != nullptr && got == OnDemandCache::Got::Miss && c1.evictions() == 1);
        CHECK(c1.peak() == BLOB && c1.no_room() == 1 && d1.log.size() == 2);
    }
    // ---- copies (the prompt path's stager): the entry is held only while it is copied and not pinned for the epochs,
    // so copying never keeps the cache from evicting it; a pointer handed out does pin it
    {
        OnDemandCache c4;
        DriveReader d4;
        d4.m = &m;
        CHECK(c4.init(tw, m.blob_bytes(), BLOB, 1, err));
        auto blob_at = [&](int l, int e) { return m.pack.data() + ((size_t) l * E + (size_t) e) * BLOB; };
        std::vector<uint8_t> out(BLOB);
        c4.begin_layer(0);
        CHECK(c4.copy(0, 6, d4, out.data(), &got) && got == OnDemandCache::Got::Miss);
        CHECK(std::memcmp(out.data(), blob_at(0, 6), BLOB) == 0);
        CHECK(c4.copy(0, 14, d4, out.data(), &got) && got == OnDemandCache::Got::Miss && c4.evictions() == 1);
        CHECK(std::memcmp(out.data(), blob_at(0, 14), BLOB) == 0);
        CHECK(c4.copy(0, 14, d4, out.data(), &got) && got == OnDemandCache::Got::Hit);
        CHECK(c4.get(0, 14, d4, &got) != nullptr && got == OnDemandCache::Got::Hit);   // now pinned in this layer
        CHECK(!c4.copy(0, 6, d4, out.data(), &got) && got == OnDemandCache::Got::NoRoom);
        CHECK(c4.peak() == BLOB && d4.log.size() == 2 && c4.no_room() == 1);
        CHECK(!c4.copy(0, 3, d4, out.data(), &got) && got == OnDemandCache::Got::Refused && whitelist_violated());
        whitelist_violation_reset();
        // threads copying and taking pointers at once, a budget of three experts for nine: every byte is right.  The
        // pointers are taken in one epoch (pinned from then on) for two experts only, so one entry is left for the copies
        OnDemandCache c5;
        CHECK(c5.init(tw, m.blob_bytes(), 3 * BLOB, 1, err));
        c5.begin_layer(0);
        std::vector<std::pair<int, int>> od;
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e)
                if (tw.on_demand(l, e)) od.emplace_back(l, e);
        std::vector<DriveReader> dr(4);
        std::atomic<int> bad{0};
        std::vector<std::thread> th;
        for (int t = 0; t < 4; ++t) {
            dr[(size_t) t].m = &m;
            th.emplace_back([&, t] {
                std::vector<uint8_t> o(BLOB);
                for (int rep = 0; rep < 40; ++rep)
                    for (size_t i = 0; i < od.size(); ++i) {
                        const auto [l, e] = od[(i * 7 + (size_t) t) % od.size()];
                        OnDemandCache::Got g = OnDemandCache::Got::Failed;
                        if (t % 2 == 0) {
                            if (!c5.copy(l, e, dr[(size_t) t], o.data(), &g)) { if (g != OnDemandCache::Got::NoRoom) ++bad; continue; }
                        } else {
                            const auto [pl, pe] = od[i % 2];   // pinned in this epoch: its pointer stays valid
                            const uint8_t* b = c5.get(pl, pe, dr[(size_t) t], &g);
                            if (b == nullptr) { if (g != OnDemandCache::Got::NoRoom) ++bad; continue; }
                            std::memcpy(o.data(), b, BLOB);
                            if (std::memcmp(o.data(), blob_at(pl, pe), BLOB) != 0) ++bad;
                            continue;
                        }
                        if (std::memcmp(o.data(), blob_at(l, e), BLOB) != 0) ++bad;
                    }
            });
        }
        for (auto& x : th) x.join();
        CHECK(bad == 0 && c5.peak() <= 3 * BLOB && c5.hits() > 0 && c5.evictions() > 0);
    }
    // ---- refused set-ups: a budget below one expert, a whitelist that is not tiered
    {
        OnDemandCache c2;
        CHECK(!c2.init(tw, m.blob_bytes(), BLOB - 1, 0, err) && err.find("cannot hold one on_demand expert") != std::string::npos);
        const WhitelistMode strict = WhitelistMode::Strict;
        const ExpertWhitelist sw = parse_ok(manifest(m, {{0, {1, 5}}, {1, {0, 3}}, {2, {2, 4}}}), m, &strict);
        CHECK(!c2.init(sw, m.blob_bytes(), budget, 0, err) && !c2.active());
    }
    // ---- threads: four of them ask for every on_demand expert many times; each is read once, everything else hits
    {
        OnDemandCache c3;
        CHECK(c3.init(tw, m.blob_bytes(), 16 * BLOB, 0, err));
        std::vector<std::pair<int, int>> od;
        for (int l = 0; l < L; ++l)
            for (int e = 0; e < E; ++e)
                if (tw.on_demand(l, e)) od.emplace_back(l, e);
        std::vector<DriveReader> dr(4);
        std::vector<std::thread> th;
        std::atomic<int> bad{0};
        for (int t = 0; t < 4; ++t) {
            dr[(size_t) t].m = &m;
            th.emplace_back([&, t] {
                for (int rep = 0; rep < 50; ++rep)
                    for (size_t i = 0; i < od.size(); ++i) {
                        const auto [l, e] = od[(i + (size_t) t) % od.size()];
                        const uint8_t* b = c3.get(l, e, dr[(size_t) t]);
                        if (b == nullptr || std::memcmp(b, m.pack.data() + ((size_t) l * E + (size_t) e) * BLOB, BLOB) != 0) ++bad;
                    }
            });
        }
        for (auto& x : th) x.join();
        size_t read = 0;
        for (const auto& d : dr) read += d.log.size();
        CHECK(bad == 0 && read == od.size() && c3.misses() == od.size() && c3.hits() == 4 * 50 * od.size() - od.size());
    }
    std::puts("expert_whitelist_test: tiered mode (read when selected, the cache budget, hard errors, counters = reference) passed");
}

/// The manifests tools/expert_whitelist.py writes (tests/data/expert_whitelist, made by its manifest_document and
/// checked to be current by tools/test_expert_whitelist.py): the engine reads what the generator writes.
void test_python_fixtures(const std::string& dir) {
    const Model m(1);
    ExpertWhitelist a, b;
    std::string err;
    if (!load_expert_whitelist(dir + "/mini-strict.json", m.info(), nullptr, a, err) ||
        !load_expert_whitelist(dir + "/mini-tiered.json", m.info(), nullptr, b, err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
        return;
    }
    CHECK(a.mode == WhitelistMode::Strict && a.total(kWlResident) == 12 && a.routable(0, 14) && !a.routable(0, 2));
    CHECK(a.profile == "rust-coding" && a.scoring == "fixture");
    CHECK(b.mode == WhitelistMode::Tiered && b.total(kWlResident) == 6 && b.total(kWlOnDemand) == 3);
    CHECK(b.routable(0, 6) && b.on_demand(0, 6) && !b.resident(0, 6) && b.substitutes() == 2);
    CHECK(!b.substitute.empty() && b.substitute[0][2] == 1 && b.substitute[0][3] == 5 && b.substitute[1][2] == -1);
    // the substitute mode's fields: the same table as tools/test_expert_whitelist.py reads from the same file
    ExpertWhitelist c;
    if (!load_expert_whitelist(dir + "/mini-substitute.json", m.info(), nullptr, c, err)) {
        std::fprintf(stderr, "  %s\n", err.c_str());
        ++g_fail;
        return;
    }
    CHECK(c.mode == WhitelistMode::Substitute && c.has_substitute_threshold && c.substitute_threshold == 0.6);
    CHECK(c.substitute_listed[0][2] == 1 && c.substitute_listed[0][3] == 5 && c.substitute_listed[0][4] == 6);
    CHECK(c.substitute[0][2] == 1 && c.substitute[0][3] == -1 && c.substitute[0][4] == 6 && c.substitute[2][0] == 7);
    CHECK(c.substitutes() == 3 && c.substitutes_below_threshold() == 1);
    CHECK(std::fabs(c.substitute_similarity[0][2] - 0.91f) < 1e-6f && std::isnan(c.substitute_similarity[0][0]));
    std::puts("expert_whitelist_test: the Python tool's manifests passed");
}

}  // namespace

int main(int argc, char** argv) {
    test_manifest();
    test_sidecar();
    test_filter_rules();
    test_paths();
    test_substitute_mode();
    test_tiered_cache();
    // the fixture folder: the argument (CMake passes it), else the repository's, from the current directory
    const std::string dir = argc > 1 ? argv[1] : "tests/data/expert_whitelist";
    if (std::filesystem::exists(dir + "/mini-strict.json")) test_python_fixtures(dir);
    else std::printf("expert_whitelist_test: %s not found, the Python fixtures were not checked\n", dir.c_str());
    if (g_fail) {
        std::fprintf(stderr, "expert_whitelist_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::puts("expert_whitelist_test: passed");
    return 0;
}
