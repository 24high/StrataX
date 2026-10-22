// include/strata/core/expert_similarity.hpp - how alike the routed experts' outputs are: the data the substitute mode's
// replacement table is built from (docs/EXPERT_WHITELIST.md sections 7.3 and 7.6).  Inference only, no weight changes.
//
// For a sample of the tokens (a fixed share, chosen by a hash of the sample id and the position, so every path and
// every block layout picks the same tokens) the collector runs EVERY expert of a layer on the token's real expert
// input.  For each expert i the router picked for that token (w_i: its final, normalised router weight) and every
// expert j of the layer it adds:
//
//     w_i * (1 + cos(y_i, y_j))                  cos: the cosine similarity of the two outputs, in [-1, 1]
//     w_i * min(|y_i - y_j| / |y_i|, cap)        the relative L2 distance, clipped at kSimilarityRelCap
//
// and per expert i its weight sum and its token count.  The means are sum / weight sum (the cosine's minus 1): the
// similarity of j's output to i's, weighted by how much the router gave i - which is what a replacement of i by j
// would take over.  Every term is fixed point (x 2^32) and rounded on its own, so sums merged in any order, split over
// runs, resumed or collected layer by layer give the same bits.  One E x E matrix per layer; nothing per token is kept.
//
// Host-only on purpose (no CUDA header): the tests build it with a plain C++ compiler.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

struct RoutingStatsSchema;

constexpr uint32_t kSimilarityVersion = 1;
constexpr int kSimilarityFracBits = 32;
constexpr double kSimilarityRelCap = 4.0;   ///< two unrelated outputs of equal norm are sqrt(2) apart

/// Which tokens are sampled.  fraction 0: the collection is off.
struct SimilarityConfig {
    double fraction = 0.0;
    uint64_t seed = 1;
    bool on() const { return fraction > 0.0; }
    /// FNV-1a 64 of the configuration's text: a file or a resumed run of another configuration is refused.
    uint64_t fnv() const;
};

/// Whether position `pos` of sample `id` is sampled: FNV-1a 64 of "<seed>:<id>:<pos>" below fraction x 2^64.
bool similarity_sampled(const SimilarityConfig& c, const std::string& id, int64_t pos);
/// How many positions of a sample of `n` tokens are sampled.
int64_t similarity_sampled_count(const SimilarityConfig& c, const std::string& id, int64_t n);

/// One sampled token's contribution: per expert the router picked (with a weight above 0 and an output that is not
/// zero), its row of the matrix.
struct SimilarityToken {
    int32_t n_expert = 0;
    std::vector<int32_t> rows;        ///< the picked experts, in rank order
    std::vector<uint64_t> weight;     ///< per row: fixed(w)
    std::vector<uint64_t> cos, rel;   ///< per row: n_expert values each
    void clear() { rows.clear(); weight.clear(); cos.clear(); rel.clear(); }
    /// The bytes one token takes at most (k rows).
    static uint64_t bytes_for(int32_t n_expert, int32_t top_k) {
        return (uint64_t) top_k * (12 + 16 * (uint64_t) n_expert);
    }
};
/// The contribution of one token: `y` is [n_expert][n], every expert's output on the token's expert input; ids/w its
/// final top-k (k entries, rank order; an id < 0 is no pick).  Norms, dots and distances in double.
void similarity_token(const float* y, int32_t n_expert, int64_t n, const int32_t* ids, const float* w, int32_t k,
                      SimilarityToken& out);

/// The accumulators of one layer: row_w[E], row_n[E], cos[E][E], rel[E][E] (u64).
class LayerSimilarity {
public:
    void reset(int32_t n_expert);
    bool empty() const { return n_ == 0; }
    int32_t n_expert() const { return n_; }
    void add(const SimilarityToken& t);
    /// Element-wise sum (same expert count: the caller checks).
    void merge(const LayerSimilarity& o);
    uint64_t row_weight(int32_t i) const { return v_[(size_t) i]; }
    uint64_t row_tokens(int32_t i) const { return v_[(size_t) n_ + (size_t) i]; }
    uint64_t cos_sum(int32_t i, int32_t j) const { return v_[2 * (size_t) n_ + (size_t) i * n_ + (size_t) j]; }
    uint64_t rel_sum(int32_t i, int32_t j) const {
        return v_[2 * (size_t) n_ + (size_t) n_ * n_ + (size_t) i * n_ + (size_t) j];
    }
    /// The weighted means; NaN when expert i has no weight.
    double mean_cos(int32_t i, int32_t j) const;
    double mean_rel(int32_t i, int32_t j) const;
    const std::vector<uint64_t>& data() const { return v_; }
    std::vector<uint64_t>& data() { return v_; }
    static uint64_t bytes_for(int32_t n_expert) { return (2 * (uint64_t) n_expert + 2 * (uint64_t) n_expert * n_expert) * 8; }
    bool operator==(const LayerSimilarity& o) const {
        return n_ == o.n_ && tokens == o.tokens && v_ == o.v_;
    }

    uint64_t tokens = 0;   ///< the sampled tokens counted
    uint64_t blocks = 0;   ///< layer-wise collection: the blocks counted so far (the header's progress field)

private:
    int32_t n_ = 0;
    std::vector<uint64_t> v_;
};

/// `similarity_NNN.bin` (docs 7.3): a 64-byte little-endian header, then row_w, row_n, cos, rel, all u64.  Written to a
/// temporary name and renamed.
std::string similarity_file_name(int64_t layer);
bool write_layer_similarity(const std::string& dir, int64_t layer, const LayerSimilarity& s, const RoutingStatsSchema& schema,
                            std::string& err);
/// Reads and checks one file (the model fingerprint, the configuration, the expert count).  `out` is reset first.
bool read_layer_similarity(const std::string& dir, int64_t layer, const RoutingStatsSchema& schema, LayerSimilarity& out,
                           std::string& err);

}  // namespace strata::core
