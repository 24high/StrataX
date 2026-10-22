// include/strata/core/layerwise_calib.hpp - layer-wise calibration: routing statistics for a model that does not fit
// in RAM and VRAM (docs/EXPERT_WHITELIST.md section 7.4).
//
// Token by token, a model larger than RAM + VRAM reads every routed expert from the drive again for every token.
// This mode turns the loops around.  The calibration sequences are cut into blocks; for one block, every sequence
// goes through layer L before any goes through layer L + 1:
//
//   for each block b:
//     for each layer l:
//       load layer l (its dense weights) - once per block
//       per sequence: the layer's part before the experts (attention or recurrence, with a KV cache of this layer
//         only, dropped after the sequence), reading its input rows from the drive (layer 0: the embeddings)
//       route every token of the block
//       per expert with at least one token, in id order: load it once, run it on all of its tokens (with the similarity
//         collection on: every expert, also on the block's sampled tokens - expert_similarity.hpp)
//       per sequence: the layer's part after the experts, writing its output rows to the drive
//       release layer l; count the block's routing into layer l's statistics
//
// The hidden rows at a layer boundary go to scratch files (F32, the engine's residual precision, or F16 / BF16);
// at most two boundaries exist at a time, the input and the output of the current layer, and they are read and
// written one sequence at a time.  A run is resumable at block and layer level: after each layer the output
// boundary, the layer's statistics and a progress record are written in an order that makes any crash point safe.
//
// The model side is `LayerwiseModel`.  A "split" model hands the scheduler the pieces (mixer, router, one expert,
// combine): the scheduler groups the experts across the whole block.  A "fused" model runs a whole layer for one
// sequence by itself (the engine's prompt path over one layer, `Prefill::run_layer`); the scheduler then only orders
// the work, keeps the boundaries and counts the routing.
//
// Host-only on purpose (no CUDA header): the tests build it with a plain C++ compiler.
#pragma once

#include "strata/core/routing_stats.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// The scratch files' precision.  F32 keeps the rows exactly (the layer-wise run then routes as the token path does).
enum class SpillPrecision : int { F32 = 0, F16 = 1, BF16 = 2 };
bool parse_spill_precision(const std::string& s, SpillPrecision& out);
const char* spill_precision_name(SpillPrecision p);
int spill_bytes(SpillPrecision p);

/// The model, one layer at a time.  Rows are host floats.  Every method may fail with a message in `err`.
class LayerwiseModel {
public:
    virtual ~LayerwiseModel() = default;
    virtual int64_t n_layers() const = 0;
    virtual int64_t row_floats() const = 0;       ///< floats per token at a layer boundary (the residual)
    virtual int64_t moe_floats() const = 0;       ///< floats per token of the experts' input and output
    virtual int32_t n_expert(int64_t l) const = 0;   ///< 0: no routed experts in layer l
    virtual int32_t top_k() const = 0;
    virtual uint64_t layer_bytes(int64_t l) const = 0;    ///< the layer's dense weights (what load_layer places)
    virtual uint64_t expert_bytes(int64_t l) const = 0;   ///< one routed expert of layer l
    /// The memory frames the scheduler keeps; a model that places its own buffers takes from them too.
    virtual void bind_meters(MemoryMeter* ram, MemoryMeter* vram) { (void) ram; (void) vram; }

    // ---- a fused model: one layer for one sequence, its own expert handling
    virtual bool fused() const { return false; }
    /// Layer `l` over sample `s`: `in` its input rows (null for layer 0: the model embeds the tokens), `out` its output
    /// rows (n x row_floats); the layer's final routing goes to `sink` (positions 0..n-1), and, for every position
    /// `sink.wants_expert_outputs` names, every expert's output (`sink.expert_outputs`).
    virtual bool forward_layer(int64_t l, const CalibSample& s, const float* in, float* out, RoutingSink& sink,
                               std::string& err) {
        (void) l, (void) s, (void) in, (void) out, (void) sink;
        err = "this model has no fused layer path";
        return false;
    }

    // ---- a split model
    virtual bool embed(const CalibSample& s, float* rows, std::string& err) = 0;
    virtual bool load_layer(int64_t l, std::string& err) = 0;
    virtual void release_layer(int64_t l) = 0;
    /// Read layer l's dense weights from the drive into RAM ahead of time (runs on another thread while the current
    /// layer computes; the RAM it takes is the scheduler's to grant).  Default: nothing.
    virtual bool prefetch_layer(int64_t l, std::string& err) { (void) l, (void) err; return true; }
    /// The part before the experts for one sequence: `rows` in place (the residual after it), `moe_in` the router's
    /// and the experts' input (n x moe_floats).  Attention state lives for this call only.
    virtual bool mixer(int64_t l, const CalibSample& s, float* rows, float* moe_in, std::string& err) = 0;
    /// The router on n rows: ids/w [n][k] in rank order (the model's own top-k and normalisation).
    virtual bool route(int64_t l, const float* moe_in, int64_t n, int32_t* ids, float* w, std::string& err) = 0;
    virtual bool load_expert(int64_t l, int32_t e, std::string& err) = 0;
    virtual void release_expert(int64_t l, int32_t e) = 0;
    /// One loaded expert on n gathered rows: y = expert(x), n x moe_floats each.
    virtual bool run_expert(int64_t l, int32_t e, const float* x, int64_t n, float* y, std::string& err) = 0;
    /// The part after the experts for one sequence: `rows` (the residual from mixer) becomes the layer's output, from
    /// `moe_in` and `moe_out` (the weighted sum of the token's experts, summed in ascending expert id).
    virtual bool combine(int64_t l, const CalibSample& s, float* rows, const float* moe_in, const float* moe_out,
                         std::string& err) = 0;
};

/// The block size when no budget and no --block-tokens limit it.
constexpr int64_t kDefaultBlockTokens = 16384;

struct LayerwiseConfig {
    uint64_t vram_budget = 0;      ///< bytes; 0: no limit
    uint64_t ram_budget = 0;       ///< bytes; 0: no limit (the scheduler's buffers, the accumulators, the prefetch)
    int64_t block_tokens = 0;      ///< tokens per block; 0: the largest the budgets hold (kDefaultBlockTokens unbounded)
    std::string scratch_dir;       ///< where the boundary files go
    uint64_t scratch_max = 0;      ///< bytes the boundary files may take at once; 0: no limit (free space is checked)
    SpillPrecision spill = SpillPrecision::F32;
    bool prefetch = false;         ///< read the next layer into RAM while this one computes (when the RAM budget holds it)
    bool keep_scratch = false;     ///< tests: leave the scratch directory
};

/// What a run needs, from the model's numbers and the budgets alone (nothing is read): the `estimate` stage's numbers,
/// and the check every run makes before it starts.
struct LayerwisePlan {
    int64_t block_tokens = 0;              ///< the block size the run uses
    int64_t max_block_tokens = 0;          ///< the largest the RAM budget holds (0: unlimited)
    std::vector<std::pair<int64_t, int64_t>> blocks;   ///< [first sample, end sample) per block
    uint64_t ram_need = 0, vram_need = 0;  ///< at the chosen block size
    uint64_t scratch_need = 0;             ///< the two boundaries of the largest block
    uint64_t dense_read_bytes = 0;         ///< every layer's dense weights, once per block
    uint64_t expert_read_upper = 0;        ///< every expert of every MoE layer once per block (the bound)
    uint64_t expert_read_expected = 0;     ///< the experts a uniform router would touch per block
    int64_t rows_per_batch = 0;            ///< rows of one router or expert batch that fit the VRAM budget
    bool prefetch = false;                 ///< the read-ahead is on (asked for, and the RAM budget holds it)
    /// The similarity collection (schema.similarity): the most tokens one block samples, and the RAM their outputs of
    /// every expert take (split form).  With it on, a block that samples a token loads every expert of every MoE layer.
    int64_t similarity_rows_max = 0;
    uint64_t similarity_bytes = 0;
    std::vector<std::string> problems;     ///< empty: the budgets suffice; else what does not fit, with the numbers
    uint64_t plan_fnv = 0;                 ///< identifies the plan: a resumed run must have the same one
};
bool plan_layerwise(const LayerwiseModel& m, const CalibSet& set, const RoutingStatsSchema& schema,
                    const LayerwiseConfig& c, LayerwisePlan& plan);

/// Counters of what the run loaded (the tests check them): per (block, layer) the dense loads, and per
/// (block, layer, expert) the expert loads.
struct LayerwiseCounters {
    std::vector<int64_t> layer_loads;          ///< [block * n_layers + layer]
    std::vector<int32_t> expert_loads;         ///< [(block * n_layers + layer) * max_expert + expert]
    int64_t max_expert = 0;
    int64_t n_layers = 0;
    int64_t prefetched = 0;                    ///< layers read ahead
    int64_t boundary_files_max = 0;            ///< the most boundary files that existed at once
};

/// Runs the layer-wise collection into `o.out_dir` (layer files with a progress count, progress.bin, at the end
/// meta.json), resuming from what is there when `o.resume`.  The scratch directory is removed after a complete run
/// unless `c.keep_scratch`.
bool collect_layerwise(LayerwiseModel& m, const CalibSet& set, const RoutingStatsSchema& schema, const LayerwiseConfig& c,
                       const CollectOptions& o, CollectReport& rep, LayerwiseCounters* counters, std::string& err);

/// Float <-> half / bfloat16 (round to nearest even), for the scratch files.
uint16_t float_to_half(float f);
float half_to_float(uint16_t h);
uint16_t float_to_bf16(float f);
float bf16_to_float(uint16_t h);

}  // namespace strata::core
