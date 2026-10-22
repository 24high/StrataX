// include/strata/core/whitelist_route.hpp - the routing filter of an expert whitelist, for ONE token.
//
// This header is the whole arithmetic of the filter, and it is compiled twice: by the CUDA/HIP kernels in
// `src/kernels/cuda/expert_whitelist.cu` (one thread per token) and by the host tests and the host router
// look-ahead.  One copy of the code is what lets a CPU-only test check what the GPU runs.
//
// The router it sits around (router_top10.cu, native_router.cu, prefill's route):
//
//     p   = softmax(logits)               over all experts
//     ids = top-k of p                    strict greater, ties to the lower index
//     w   = p[ids] / max(sum(p[ids]), 2^-14)
//
// STRICT and TIERED: `mask_row` sets every expert the router may not pick to -inf BEFORE that top-k.  exp(-inf - m)
// is 0, so the softmax is the model's softmax over the routable experts and the weights are normalised exactly as
// the model normalises them.  The allowed logits keep their bits, so a whitelist that allows everything changes
// nothing.  `fix_row` runs AFTER the top-k: it is a no-op unless a blocked id got through anyway - possible only
// when a routable expert's p underflowed to 0 and tied with the masked zeros (a logit gap over ~87), or when a later
// pass (STRATA_ROUTE_PRIOR, the prompt's debug override) wrote one.  Then it takes the best routable expert not yet
// chosen (largest logit, ties to the lower index) and recomputes that token's weights over its final ids.
//
// SUBSTITUTE: the top-k runs on the model's own logits.  `substitute_row` then walks the ranks: a blocked pick goes
// to its replacement from the manifest and keeps its weight; if the replacement was already chosen the two weights
// add, and the freed rank takes the best allowed expert not chosen (weight 0, so the sum is unchanged and the token
// still has k distinct experts); a blocked pick with no replacement falls back to the best allowed expert not chosen,
// which takes its weight.  The sum of the weights is the router's.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_WL_HD __host__ __device__ inline
#else
#define STRATA_WL_HD inline
#endif

namespace strata::core::wlroute {

/// The largest top-k the filter takes (the router kernels' own limit).
constexpr int kMaxK = 64;
/// The router's renormalisation floor, 2^-14 (ggml_clamp in llama-graph.cpp).
constexpr double kSumFloor = 6.103515625e-5;

/// Strict and tiered, before the top-k: every expert with `routable[e] == 0` gets -inf.  The others are untouched.
STRATA_WL_HD void mask_row(float* logits, const uint8_t* routable, int n_expert) {
    for (int e = 0; e < n_expert; ++e)
        if (!routable[e]) logits[e] = -INFINITY;
}

/// The best expert with `ok[e] != 0` that is not in ids[0..k): the largest logit, ties to the lower index (a NaN
/// logit ranks below every number).  -1 when there is none.
STRATA_WL_HD int best_free(const float* logits, const uint8_t* ok, int n_expert, const int32_t* ids, int k) {
    int best = -1;
    float bv = -INFINITY;
    for (int e = 0; e < n_expert; ++e) {
        if (!ok[e]) continue;
        bool used = false;
        for (int q = 0; q < k; ++q) used = used || ids[q] == e;
        if (used) continue;
        const float v = logits[e] == logits[e] ? logits[e] : -INFINITY;
        if (best < 0 || v > bv) { best = e; bv = v; }
    }
    return best;
}

/// The router's weights for the token's final ids, over the experts with `ok[e] != 0`:
/// p = softmax over them, w = p[ids] / max(sum(p[ids]), 2^-14).  Only after a fix (never on the normal path).
STRATA_WL_HD void reweight(const float* logits, const uint8_t* ok, int n_expert, const int32_t* ids, float* w, int k) {
    float mx = -INFINITY;
    for (int e = 0; e < n_expert; ++e)
        if (ok[e] && logits[e] > mx) mx = logits[e];
    double z = 0.0;
    for (int e = 0; e < n_expert; ++e)
        if (ok[e] && logits[e] == logits[e]) z += exp((double) logits[e] - (double) mx);
    double p[kMaxK];
    double s = 0.0;
    for (int q = 0; q < k; ++q) {
        const int e = ids[q];
        p[q] = (e >= 0 && e < n_expert && z > 0.0 && logits[e] == logits[e]) ? exp((double) logits[e] - (double) mx) / z
                                                                             : 0.0;
        s += p[q];
    }
    if (s < kSumFloor) s = kSumFloor;
    for (int q = 0; q < k; ++q) w[q] = (float) (p[q] / s);
}

/// Strict and tiered, after the top-k (and after any pass that rewrites the ids).  Returns how many ranks it
/// replaced: 0 on the normal path, where nothing is written.
STRATA_WL_HD int fix_row(const float* logits, const uint8_t* routable, int n_expert, int32_t* ids, float* w, int k) {
    int fixed = 0;
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        if (e >= 0 && e < n_expert && routable[e]) continue;
        ids[j] = -1;   // not "taken" while its replacement is chosen
        ids[j] = best_free(logits, routable, n_expert, ids, k);
        ++fixed;
    }
    if (fixed > 0) reweight(logits, routable, n_expert, ids, w, k);
    return fixed;
}

/// Substitute mode, after the model's own top-k.  `allowed[e]` != 0: the expert may run; `sub[e]`: blocked e's
/// replacement or -1 (`sub` may be null: no table).  Returns how many ranks it changed.
STRATA_WL_HD int substitute_row(const float* logits, const uint8_t* allowed, const int32_t* sub, int n_expert,
                                int32_t* ids, float* w, int k) {
    int changed = 0;
    for (int j = 0; j < k; ++j) {
        const int e = ids[j];
        if (e >= 0 && e < n_expert && allowed[e]) continue;
        ++changed;
        const int s = (sub != nullptr && e >= 0 && e < n_expert) ? sub[e] : -1;
        if (s >= 0 && s < n_expert && allowed[s]) {
            int at = -1;
            for (int q = 0; q < k; ++q)
                if (q != j && ids[q] == s) at = q;
            if (at < 0) {   // the replacement takes e's place and e's weight
                ids[j] = s;
                continue;
            }
            w[at] += w[j];  // already chosen: the two weights add, and rank j is refilled below with weight 0
            w[j] = 0.0f;
        }
        // no replacement, or it was chosen already: the best allowed expert not chosen, as strict mode picks it
        ids[j] = -1;
        ids[j] = best_free(logits, allowed, n_expert, ids, k);
    }
    return changed;
}

/// The whole filter after the top-k for one token, by mode (0 strict, 1 substitute, 2 tiered; the WhitelistMode
/// values).  `routable` is the effective mode's table (substitute mode: the allowed experts).
STRATA_WL_HD int after_topk(int mode, const float* logits, const uint8_t* routable, const int32_t* sub, int n_expert,
                            int32_t* ids, float* w, int k) {
    return mode == 1 ? substitute_row(logits, routable, sub, n_expert, ids, w, k)
                     : fix_row(logits, routable, n_expert, ids, w, k);
}

/// Before the top-k, by mode: strict and tiered mask; substitute leaves the model's logits alone.
STRATA_WL_HD void before_topk(int mode, float* logits, const uint8_t* routable, int n_expert) {
    if (mode != 1) mask_row(logits, routable, n_expert);
}

}  // namespace strata::core::wlroute
