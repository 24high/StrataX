// src/kernels/cuda/expert_whitelist.cu - the expert whitelist's routing filter (include/strata/kernels/expert_whitelist.hpp).
//
// The per-token arithmetic is `strata/core/whitelist_route.hpp`, the same code tests/core/expert_whitelist_test.cpp
// runs on the host.  The mask is one thread per logit (it only writes -inf where `routable` is 0, which is
// `wlroute::mask_row` element by element); the fix is one thread per token (a token's ranks depend on each other).
// On the normal path the fix reads k bytes of the table per token and writes nothing.
#include "strata/kernels/expert_whitelist.hpp"

#include "strata/core/expert_whitelist.hpp"
#include "strata/core/whitelist_route.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace strata::kernels {
namespace {

constexpr int kMaxDevices = 16;

struct Tables {
    bool on = false;
    int mode = 0;
    int64_t n_layers = 0;
    int64_t stride = 0;                              ///< experts per layer (the largest)
    uint8_t* routable[kMaxDevices] = {};             ///< [n_layers * stride]
    int32_t* sub[kMaxDevices] = {};                  ///< [n_layers * stride], substitute mode with a table only
    int32_t* slot[kMaxDevices] = {};                 ///< [n_layers * stride], the compact store's slot map
    unsigned long long* counts[kMaxDevices] = {};    ///< [n_layers * stride + 1]: selected ids, then the unmapped count
    bool stats = false;
};
Tables g_tab[2];

__global__ void wl_mask_k(float* __restrict__ logits, const uint8_t* __restrict__ routable, int64_t n, int n_expert) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !routable[i % n_expert]) logits[i] = -INFINITY;
}

__global__ void wl_fix_k(int mode, const float* __restrict__ logits, const uint8_t* __restrict__ routable,
                         const int32_t* __restrict__ sub, int32_t* __restrict__ ids, float* __restrict__ w, int n_tok, int k,
                         int n_expert, unsigned long long* __restrict__ counts) {
    const int t = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n_tok) return;
    int32_t* id = ids + (size_t) t * (size_t) k;
    strata::core::wlroute::after_topk(mode, logits + (size_t) t * (size_t) n_expert, routable, sub, n_expert, id,
                                      w + (size_t) t * (size_t) k, k);
    if (counts != nullptr)
        for (int j = 0; j < k; ++j)
            if (id[j] >= 0 && id[j] < n_expert) atomicAdd(counts + id[j], 1ull);
}

__global__ void wl_slots_k(const int32_t* __restrict__ ids, int32_t* __restrict__ slots, const int32_t* __restrict__ map,
                           int n, int n_expert, unsigned long long* __restrict__ unmapped) {
    const int i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const int32_t e = ids[i];
    const int32_t s = e >= 0 && e < n_expert ? map[e] : -1;
    if (s < 0 && unmapped != nullptr) atomicAdd(unmapped, 1ull);
    slots[i] = s < 0 ? 0 : s;
}

void release(Tables& T) {
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kMaxDevices; ++d) {
        if (!T.routable[d] && !T.sub[d] && !T.slot[d] && !T.counts[d]) continue;
        cudaSetDevice(d);
        cudaFree(T.routable[d]);
        cudaFree(T.sub[d]);
        cudaFree(T.slot[d]);
        cudaFree(T.counts[d]);
    }
    cudaSetDevice(prev);
    T = Tables{};
}

/// The current device's index into the tables, or -1 (a hard error: that GPU routes without the whitelist).
int device_of(const Tables& T, int64_t layer) {
    int d = -1;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kMaxDevices || T.routable[d] == nullptr) {
        strata::core::whitelist_note_violation(layer, -1, "a router launch on a GPU without the whitelist's tables");
        return -1;
    }
    return d;
}

template <class X>
bool put(X*& dst, const std::vector<X>& src, std::string& err, const char* what, int dev) {
    if (src.empty()) return true;
    if (cudaMalloc((void**) &dst, src.size() * sizeof(X)) != cudaSuccess ||
        cudaMemcpy(dst, src.data(), src.size() * sizeof(X), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string("expert whitelist: cannot upload the ") + what + " to GPU " + std::to_string(dev);
        return false;
    }
    return true;
}

}  // namespace

bool whitelist_route_upload(int model, const strata::core::ExpertWhitelist* wl, const std::vector<int>& devices,
                            bool stats, std::string& err) {
    if (model != kWlMain && model != kWlDraft) { err = "expert whitelist: unknown model"; return false; }
    Tables& T = g_tab[model];
    release(T);
    if (wl == nullptr || !wl->active()) return true;
    T.mode = (int) wl->mode;
    T.n_layers = wl->n_layers;
    for (int32_t n : wl->experts_per_layer) T.stride = std::max<int64_t>(T.stride, n);
    const size_t cells = (size_t) (T.n_layers * T.stride);
    std::vector<uint8_t> routable(cells, 0);
    std::vector<int32_t> sub, slot(cells, -1);
    for (int64_t l = 0; l < T.n_layers; ++l)
        for (int64_t e = 0; e < T.stride; ++e) routable[(size_t) (l * T.stride + e)] = wl->routable(l, e) ? 1 : 0;
    if (wl->mode == strata::core::WhitelistMode::Substitute && !wl->substitute.empty()) {
        sub.assign(cells, -1);
        for (int64_t l = 0; l < T.n_layers; ++l)
            for (size_t e = 0; e < wl->substitute[(size_t) l].size(); ++e)
                sub[(size_t) l * (size_t) T.stride + e] = wl->substitute[(size_t) l][e];
    }
    const strata::core::ExpertSlotMap map = strata::core::ExpertSlotMap::build(*wl, false);
    if (!map.check(err)) return false;
    for (int64_t l = 0; l < T.n_layers; ++l)
        for (int64_t e = 0; e < T.stride; ++e) slot[(size_t) (l * T.stride + e)] = map.slot(l, e);
    std::vector<unsigned long long> zeros(cells + 1, 0ull);
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d : devices) {
        if (d < 0 || d >= kMaxDevices) { err = "expert whitelist: GPU " + std::to_string(d) + " is out of range"; break; }
        if (T.routable[d] != nullptr) continue;   // listed twice
        if (cudaSetDevice(d) != cudaSuccess) { err = "expert whitelist: cannot use GPU " + std::to_string(d); break; }
        if (!put(T.routable[d], routable, err, "routing table", d) || !put(T.sub[d], sub, err, "substitute table", d) ||
            !put(T.slot[d], slot, err, "slot map", d) || !put(T.counts[d], zeros, err, "counters", d))
            break;
    }
    cudaSetDevice(prev);
    if (!err.empty()) { release(T); return false; }
    T.stats = stats;
    T.on = true;
    return true;
}

bool whitelist_route_active(int model) { return (model == kWlMain || model == kWlDraft) && g_tab[model].on; }

void whitelist_route_mask(int model, int64_t layer, float* logits, int n_tok, int n_expert, void* stream) {
    if (!whitelist_route_active(model) || n_tok <= 0) return;
    const Tables& T = g_tab[model];
    if (T.mode == (int) strata::core::WhitelistMode::Substitute) return;   // the model's own top-k
    const int d = device_of(T, layer);
    if (d < 0 || layer < 0 || layer >= T.n_layers || n_expert != T.stride) {
        if (d >= 0) strata::core::whitelist_note_violation(layer, -1, "a router launch outside the whitelist's layers");
        return;
    }
    const int64_t n = (int64_t) n_tok * n_expert;
    wl_mask_k<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(
        logits, T.routable[d] + (size_t) (layer * T.stride), n, n_expert);
}

void whitelist_route_fix(int model, int64_t layer, const float* logits, int32_t* ids, float* weights, int n_tok, int k,
                         int n_expert, void* stream) {
    if (!whitelist_route_active(model) || n_tok <= 0) return;
    const Tables& T = g_tab[model];
    const int d = device_of(T, layer);
    if (d < 0) return;
    if (layer < 0 || layer >= T.n_layers || n_expert != T.stride || k < 1 || k > strata::core::wlroute::kMaxK) {
        strata::core::whitelist_note_violation(layer, -1, "a router launch outside the whitelist's layers");
        return;
    }
    const size_t off = (size_t) (layer * T.stride);
    wl_fix_k<<<(unsigned) ((n_tok + 63) / 64), 64, 0, (cudaStream_t) stream>>>(
        T.mode, logits, T.routable[d] + off, T.sub[d] != nullptr ? T.sub[d] + off : nullptr, ids, weights, n_tok, k,
        n_expert, T.stats ? T.counts[d] + off : nullptr);
}

void whitelist_ids_to_slots(int model, int64_t layer, const int32_t* ids, int32_t* slots, int n, void* stream) {
    if (!whitelist_route_active(model) || n <= 0) return;
    const Tables& T = g_tab[model];
    const int d = device_of(T, layer);
    if (d < 0 || layer < 0 || layer >= T.n_layers) return;
    wl_slots_k<<<(unsigned) ((n + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        ids, slots, T.slot[d] + (size_t) (layer * T.stride), n, (int) T.stride,
        T.counts[d] + (size_t) (T.n_layers * T.stride));
}

bool whitelist_route_counts(int model, std::vector<uint64_t>& selected, uint64_t& unmapped, std::string& err) {
    selected.clear();
    unmapped = 0;
    if (!whitelist_route_active(model)) return true;
    const Tables& T = g_tab[model];
    const size_t cells = (size_t) (T.n_layers * T.stride);
    std::vector<unsigned long long> h(cells + 1);
    if (T.stats) selected.assign(cells, 0);
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kMaxDevices; ++d) {
        if (T.counts[d] == nullptr) continue;
        cudaSetDevice(d);
        if (cudaMemcpy(h.data(), T.counts[d], h.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "expert whitelist: cannot read the counters of GPU " + std::to_string(d);
            cudaSetDevice(prev);
            return false;
        }
        if (T.stats)
            for (size_t i = 0; i < cells; ++i) selected[i] += h[i];
        unmapped += h[cells];
    }
    cudaSetDevice(prev);
    return true;
}

void whitelist_route_release() {
    release(g_tab[kWlMain]);
    release(g_tab[kWlDraft]);
}

}  // namespace strata::kernels
