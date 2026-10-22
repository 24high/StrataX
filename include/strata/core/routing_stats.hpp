// include/strata/core/routing_stats.hpp - routing statistics for the expert whitelist generator (opt-in).
//
// The generator's `collect` stage (docs/EXPERT_WHITELIST.md section 7) runs prepared calibration samples through the
// model and counts, per MoE layer, routed expert and category of token:
//
//   count        tokens for which the router put the expert in its top-k
//   gate_sum     the sum of the expert's final (normalised) router weight over those tokens
//   gate_sq_sum  the sum of its square
//   contrib_sum  the sum of weight x ||expert output|| (only where the path can give it, else 0)
//   rank_hist    how often the expert was at rank 0, 1, ..., k-1 of the top-k
//
// and per category the tokens that passed the layer (the denominator).  A category is one data set (the profile's
// languages, a base requirement, the reference corpus), language, task, region (system, prompt, response, tool) and
// origin (prefill: teacher-forced, generated: the model's own tokens).
//
// Every value is an unsigned 64-bit integer; the weight sums are fixed point (kGateFracBits, kContribFracBits).  So
// adding is exact: statistics merged in any order, split over any number of runs, interrupted and resumed, or computed
// layer by layer instead of token by token give the same bits.  Raw per-token data is never kept.
//
// Host-only on purpose (no CUDA header): the tests build it with a plain C++ compiler.
#pragma once

#include "strata/core/expert_similarity.hpp"
#include "strata/core/expert_whitelist.hpp"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

constexpr int kGateFracBits = 32;      ///< gate_sum and gate_sq_sum: value * 2^32, rounded to nearest
constexpr int kContribFracBits = 20;   ///< contrib_sum: value * 2^20
constexpr uint32_t kRoutingStatsVersion = 1;
constexpr uint32_t kCalibSamplesVersion = 1;

/// One category of tokens.  The key is "dataset|language|task|region|origin"; no field may contain '|'.
struct RoutingCategory {
    std::string key;
    std::string dataset, language, task, region, origin;
    static bool from_key(const std::string& key, RoutingCategory& out, std::string& err);
};

/// One prepared calibration sample (tools/whitelist_gen.py prepare): the token ids of the whole sequence (template,
/// prompt and reference response) and each token's category index.
struct CalibSample {
    std::string id;
    std::vector<int32_t> tokens;
    std::vector<uint16_t> cat;
};

/// A prepared sample file (`samples.bin` or `holdout.bin` of a prep directory).
struct CalibSet {
    std::vector<RoutingCategory> categories;
    std::vector<CalibSample> samples;
    uint64_t fnv = 0;   ///< FNV-1a 64 of the file's bytes: a resumed run must read the same file
    int64_t tokens() const;
    int64_t longest() const;
};
bool read_calib_samples(const std::string& path, CalibSet& out, std::string& err);
bool write_calib_samples(const std::string& path, const CalibSet& set, std::string& err);

/// What every statistics file of a run is checked against: the model (its fingerprint's inputs) and the categories.
struct RoutingStatsSchema {
    WhitelistModelInfo model;
    std::string fingerprint;            ///< whitelist_fingerprint(model)
    std::vector<RoutingCategory> categories;
    bool has_contrib = false;
    std::string quantization;           ///< what the weights were (the pack's format), for the manifest
    uint64_t dense_bytes = 0;           ///< the model's non-expert weights, when the runtime knows them (0: unknown)
    /// The experts' output similarity (expert_similarity.hpp), collected beside the statistics; off by default
    SimilarityConfig similarity;
    /// FNV-1a 64 of the category keys, one per line: a layer file of another schema is refused
    uint64_t schema_fnv() const;
    uint64_t fingerprint_fnv() const;
    std::vector<int64_t> moe_layers() const;   ///< the layers with routed experts
};

/// The accumulators of one MoE layer: tokens[c], and per category and expert 4 + top_k values
/// (count, gate_sum, gate_sq_sum, contrib_sum, rank_hist[0..k-1]).
class LayerStats {
public:
    static constexpr int kCount = 0, kGate = 1, kGateSq = 2, kContrib = 3, kRank = 4;

    void reset(int32_t n_expert, int32_t top_k, int32_t n_cat);
    bool empty() const { return n_cat_ == 0; }
    int32_t n_expert() const { return n_expert_; }
    int32_t top_k() const { return top_k_; }
    int32_t n_cat() const { return n_cat_; }
    int32_t width() const { return 4 + top_k_; }
    /// One token of category `c` passed the layer; its top-k is `ids`/`w` (k entries, rank order; an id < 0 is no pick),
    /// `contrib` (may be null) the weight x ||output|| of each pick.
    void add_token(int32_t c, const int32_t* ids, const float* w, const float* contrib);
    /// The same split up: the token (the denominator) and one pick of rank `rank`.  add_token is these two.
    void add_denominator(int32_t c, uint64_t n = 1) { tokens_[(size_t) c] += n; }
    void add_pick(int32_t c, int32_t e, int32_t rank, float w, float contrib);
    /// Element-wise sum (the schemas must agree: the caller checks).
    void merge(const LayerStats& o);
    uint64_t tokens(int32_t c) const { return tokens_[(size_t) c]; }
    uint64_t at(int32_t c, int32_t e, int k) const { return v_[((size_t) c * n_expert_ + (size_t) e) * width() + (size_t) k]; }
    const std::vector<uint64_t>& token_array() const { return tokens_; }
    const std::vector<uint64_t>& value_array() const { return v_; }
    std::vector<uint64_t>& token_array() { return tokens_; }
    std::vector<uint64_t>& value_array() { return v_; }
    /// The bytes these accumulators take in RAM.
    uint64_t bytes() const { return (uint64_t) (tokens_.size() + v_.size()) * 8; }
    static uint64_t bytes_for(int32_t n_expert, int32_t top_k, int32_t n_cat) {
        return (uint64_t) n_cat * 8 + (uint64_t) n_cat * (uint64_t) n_expert * (uint64_t) (4 + top_k) * 8;
    }
    bool operator==(const LayerStats& o) const {
        return n_expert_ == o.n_expert_ && top_k_ == o.top_k_ && n_cat_ == o.n_cat_ && tokens_ == o.tokens_ && v_ == o.v_;
    }

    uint64_t blocks = 0;   ///< layer-wise collection: the blocks accumulated so far (the header's progress field)

private:
    int32_t n_expert_ = 0, top_k_ = 0, n_cat_ = 0;
    std::vector<uint64_t> tokens_, v_;
};

/// Fixed point of one weight (or a square, or a contribution): floor(v * 2^bits + 0.5), negative and non-finite
/// values 0, saturated at 2^64 - 1.
uint64_t routing_fixed(double v, int bits);
/// weight x ||y|| of one pick (y: the expert's n output floats; the norm in double), as every path computes it.
float routing_contrib(float w, const float* y, int64_t n);

/// `layer_NNN.bin` (docs/EXPERT_WHITELIST.md 7.3): a 64-byte little-endian header, then tokens[C], then values[C][E][4+k],
/// all u64.  Written to a temporary name and renamed, so a reader never sees half a file.
std::string layer_stats_name(int64_t layer);
bool write_layer_stats(const std::string& dir, int64_t layer, const LayerStats& s, const RoutingStatsSchema& schema,
                       std::string& err);
/// Reads and checks one layer file (the model fingerprint, the schema, the geometry).  `out` is reset first.
bool read_layer_stats(const std::string& dir, int64_t layer, const RoutingStatsSchema& schema, LayerStats& out,
                      std::string& err);

/// The run's description for meta.json.
struct RoutingStatsRun {
    std::string mode;                  ///< "token" or "layerwise"
    std::string collect_json = "{}";   ///< a JSON object from the caller, copied verbatim (inputs' hashes, parameters)
    std::string generator = "strata-routing-stats/1";
    std::string created;               ///< UTC date; empty: today
    std::string run_params_json = "{}";   ///< the collector's own parameters (block size, budgets, precision)
};

/// meta.json: the format, the model, the categories, the run, the processed sample ids and each layer file's size and
/// CRC-32.  Reads the layer files back for their checksums, so it is written after them.
bool write_stats_meta(const std::string& dir, const RoutingStatsSchema& schema, const RoutingStatsRun& run,
                      const std::vector<std::string>& samples_done, uint64_t tokens, std::string& err);

// ================================ COLLECTION ================================

/// Where a path reports the final routing of a run of positions of one layer (the prompt path calls it once per layer
/// and chunk, the token path once per layer and position).  ids/w are [rows][k] in rank order; contrib may be null.
class RoutingSink {
public:
    virtual ~RoutingSink() = default;
    virtual void routed(int64_t layer, int64_t pos0, int64_t rows, const int32_t* ids, const float* w,
                        const float* contrib) = 0;
    /// The similarity collection (expert_similarity.hpp): whether the path should run every expert of `layer` on
    /// position `pos` and hand the outputs to `expert_outputs`.  False when the run collects no similarity data.
    virtual bool wants_expert_outputs(int64_t layer, int64_t pos) const { (void) layer, (void) pos; return false; }
    /// `y`: [n_expert][n] every expert's output on the position's expert input; ids/w its final top-k (rank order).
    /// A run that wants them checks that every sampled position of every MoE layer came exactly once.
    virtual void expert_outputs(int64_t layer, int64_t pos, const float* y, int64_t n, const int32_t* ids, const float* w) {
        (void) layer, (void) pos, (void) y, (void) n, (void) ids, (void) w;
    }
};

/// The prompt path's side of the similarity collection (docs 7.4).  The engine's prompt path computes only the routed
/// experts, so it hands over, per MoE layer and chunk, the routing (`routed`) and then the expert input of the rows the
/// sink samples (`wanted` names them, the path downloads them, `moe_rows` takes them).  This computes every expert's
/// output on those rows with `outputs` (the engine: the CPU expert kernels) and passes them to the sink.  Each row's
/// routing must come before its expert input.
class ChunkOutputFeeder {
public:
    /// Every expert of `layer` on `n` rows of expert input (`x`: n x width floats); `y`: [n][n_expert][width].
    using ExpertOutputs = std::function<bool(int64_t layer, const float* x, int64_t n, float* y, std::string& err)>;
    ChunkOutputFeeder(RoutingSink& sink, int32_t top_k, int32_t n_expert, int64_t width, ExpertOutputs outputs)
        : sink_(sink), k_(top_k), e_(n_expert), width_(width), outputs_(std::move(outputs)) {}
    /// The routing of positions pos0 .. pos0+T-1 of `layer`: to the sink, and kept for the rows' outputs.
    void routed(int64_t layer, int64_t pos0, int64_t T, const int32_t* ids, const float* w);
    /// The rows of that chunk (0 .. T-1) whose expert input the sink wants, ascending.
    void wanted(int64_t layer, int64_t pos0, int64_t T, std::vector<int32_t>& rows) const;
    /// The expert input of those rows (`rows[i]` relative to pos0; x: n x width floats).
    void moe_rows(int64_t layer, int64_t pos0, const int32_t* rows, int64_t n, const float* x);
    /// False (and why) when a step failed: the expert outputs could not be computed, or rows came without routing.
    bool ok(std::string& err) const;
    int64_t rows_done() const { return rows_done_; }

private:
    RoutingSink& sink_;
    int32_t k_, e_;
    int64_t width_;
    ExpertOutputs outputs_;
    int64_t layer_ = -1, pos0_ = 0, T_ = 0;
    std::vector<int32_t> ids_;
    std::vector<float> w_, y_;
    int64_t rows_done_ = 0;
    std::string err_;
};

/// A memory frame: what the collector's own buffers (and a model's that report here) may take.  `take` fails, and
/// says what and how much, past the budget (0: no limit, only counted).  Thread-safe (a prefetch thread takes too).
class MemoryMeter {
public:
    explicit MemoryMeter(uint64_t budget = 0, const char* name = "RAM") : budget_(budget), name_(name) {}
    MemoryMeter(const MemoryMeter&) = delete;
    MemoryMeter& operator=(const MemoryMeter&) = delete;
    bool take(uint64_t bytes, const char* what, std::string& err);
    void give(uint64_t bytes);
    uint64_t used() const;
    uint64_t peak() const;
    uint64_t budget() const { return budget_; }
    const char* name() const { return name_; }

private:
    uint64_t budget_ = 0, used_ = 0, peak_ = 0;
    const char* name_;
    mutable std::mutex mu_;
};

/// A test's simulated abort: called at named points ("sample", "after_route", "mid_experts", "after_boundary",
/// "after_stats", "after_progress"); true stops the run there as a crash would (nothing more is written).
using CollectFault = std::function<bool(const char* point, int64_t block_or_sample, int64_t layer)>;

struct CollectOptions {
    std::string out_dir;
    bool resume = false;
    int64_t checkpoint_every = 64;   ///< token mode: samples between checkpoints
    uint64_t ram_budget = 0;         ///< the collector's own buffers (0: no limit)
    RoutingStatsRun run;
    CollectFault fault;
    /// progress, one line each (null: quiet)
    std::function<void(const std::string&)> log;
};

struct CollectReport {
    int64_t samples = 0, samples_skipped = 0, tokens = 0;
    int64_t similarity_tokens = 0;    ///< the tokens sampled for the similarity data (counted once, not per layer)
    uint64_t ram_peak = 0, vram_peak = 0;
    uint64_t scratch_peak = 0;        ///< layer-wise: the largest the scratch files were at once
    double seconds = 0.0;
    bool resumed = false;
};

/// Token mode: runs each sample through the model as it normally runs (the prompt path, teacher-forced over the whole
/// sequence) and counts every routed token.  Checkpoints (all layers' accumulators and the samples done, one atomic
/// file) every `checkpoint_every` samples; with `resume` a run continues after the last checkpoint.  At the end the
/// statistics directory is written and the checkpoint removed.
using SampleRunner = std::function<bool(const CalibSample& s, RoutingSink& sink, std::string& err)>;
bool collect_token_mode(const CalibSet& set, const RoutingStatsSchema& schema, const SampleRunner& run,
                        const CollectOptions& o, CollectReport& rep, std::string& err);

/// Token mode's sink: one sample's routing, held until the whole sample has run, then added to the accumulators in one
/// step (a sample that fails half way adds nothing).  Every MoE layer must report every position exactly once, and
/// with the similarity collection on, the experts' outputs of every sampled position (`schema.similarity`).
class SampleRoutingSink : public RoutingSink {
public:
    SampleRoutingSink(const RoutingStatsSchema& schema, bool has_contrib);
    /// The bytes the buffers take for a sample of `n` tokens (routing only).
    uint64_t bytes_for(int64_t n) const;
    /// The same for one sample, with its sampled tokens' similarity contributions.
    uint64_t bytes_for(const CalibSample& s) const;
    void begin(const CalibSample& s);
    void routed(int64_t layer, int64_t pos0, int64_t rows, const int32_t* ids, const float* w,
                const float* contrib) override;
    bool wants_expert_outputs(int64_t layer, int64_t pos) const override;
    void expert_outputs(int64_t layer, int64_t pos, const float* y, int64_t n, const int32_t* ids, const float* w) override;
    /// Checks that every MoE layer saw every position once (and every sampled one's expert outputs), then adds the
    /// sample to `layers` and `sims` (indexed by layer; `sims` may be null when the similarity collection is off).
    bool commit(std::vector<LayerStats>& layers, std::vector<LayerSimilarity>* sims, std::string& err);
    /// The positions of the current sample that are sampled for the similarity data.
    int64_t sampled() const { return n_sampled_; }

private:
    const RoutingStatsSchema& schema_;
    std::vector<int64_t> moe_;
    int32_t k_ = 0;
    bool contrib_ = false;
    const CalibSample* s_ = nullptr;
    std::vector<std::vector<int32_t>> ids_;    // [layer][pos * k]
    std::vector<std::vector<float>> w_, c_;    // [layer][pos * k]
    std::vector<std::vector<uint8_t>> seen_;   // [layer][pos]
    // the similarity collection: which positions are sampled, which reported their outputs, their contributions
    std::vector<uint8_t> sampled_;                          // [pos]
    int64_t n_sampled_ = 0;
    std::vector<std::vector<uint8_t>> sim_seen_;            // [layer][pos]
    std::vector<std::vector<SimilarityToken>> sim_;         // [layer][the sampled positions, in the order reported]
    std::string err_;
};

/// A file written under a temporary name and moved into place by `commit` (after a flush to the drive): a reader, or
/// a run resumed after a crash, sees the old file or the whole new one.  Destroyed without commit, the temporary goes.
class AtomicFile {
public:
    AtomicFile() = default;
    AtomicFile(const AtomicFile&) = delete;
    AtomicFile& operator=(const AtomicFile&) = delete;
    ~AtomicFile() { abandon(); }
    bool open(const std::string& path, std::string& err);
    bool write(const void* p, size_t n);
    bool commit(std::string& err);
    void abandon();
    uint64_t written() const { return written_; }

private:
    std::FILE* f_ = nullptr;
    std::string path_, tmp_;
    bool failed_ = false;
    uint64_t written_ = 0;
};
/// fsync-and-rename: `path` gets exactly `bytes`, or keeps what it had.
bool atomic_write_file(const std::string& path, const void* bytes, size_t n, std::string& err);
/// CRC-32 (zlib's) of a file's bytes, what meta.json records per layer file (Python checks it with zlib.crc32);
/// false and `err` when the file cannot be read.
bool file_crc32(const std::string& path, uint32_t& crc, uint64_t& bytes, std::string& err);
uint32_t crc32_update(uint32_t crc, const void* data, size_t n);
std::string json_escape(const std::string& s);
std::string utc_date_today();

}  // namespace strata::core
