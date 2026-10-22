// include/strata/kernels/expert_whitelist.hpp - the expert whitelist's routing filter on the GPU (host-callable).
//
// Two launches around every router top-k, both no-ops (nothing is launched, a captured graph is unchanged) while no
// whitelist is uploaded:
//
//   whitelist_route_mask   BEFORE the top-k: blocked experts' logits to -inf (strict and tiered modes)
//   whitelist_route_fix    AFTER the top-k and after any pass that rewrites the ids (STRATA_ROUTE_RESIDENT,
//                          STRATA_ROUTE_PRIOR, the prompt's debug override): strict/tiered's fix-up, substitute's
//                          replacement (strata/core/whitelist_route.hpp is the arithmetic, one thread per token)
//
// The tables are uploaded once at start to every GPU the run routes on, before any graph is captured, and are
// never written again.  A router launch on a GPU without them is a hard error (whitelist_note_violation).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {
struct ExpertWhitelist;
}

namespace strata::kernels {

/// Which whitelist: the main model's, or the MTP draft layer's own (one layer).
enum WhitelistModel : int { kWlMain = 0, kWlDraft = 1 };

/// Uploads `wl`'s routing tables (null: none) to each of `devices`.  `stats`: also a device counter per
/// (layer, expert) that the fix launch adds every final id to (--expert-whitelist-stats).  The current device is
/// unchanged on return.  Call before any session or graph is set up.
bool whitelist_route_upload(int model, const strata::core::ExpertWhitelist* wl, const std::vector<int>& devices,
                            bool stats, std::string& err);
bool whitelist_route_active(int model);
/// Before the top-k: logits (n_tok rows of n_expert floats) of layer `layer`.  Strict and tiered only.
void whitelist_route_mask(int model, int64_t layer, float* logits, int n_tok, int n_expert, void* stream);
/// After the top-k: ids / weights (n_tok rows of k) of layer `layer`, from the same logits.
void whitelist_route_fix(int model, int64_t layer, const float* logits, int32_t* ids, float* weights, int n_tok, int k,
                         int n_expert, void* stream);
/// The draft layer's compact expert store: ids -> slots of `layer` (the whitelist's slot map).  An id without a slot
/// (never after whitelist_route_fix) reads slot 0 and is counted in `whitelist_route_counts`' `unmapped`.
void whitelist_ids_to_slots(int model, int64_t layer, const int32_t* ids, int32_t* slots, int n, void* stream);
/// The device counters summed over the GPUs: selected[layer * n_expert + e] (empty without stats), and the ids the
/// slot remap found without a slot.  Synchronous; outside any capture.
bool whitelist_route_counts(int model, std::vector<uint64_t>& selected, uint64_t& unmapped, std::string& err);
void whitelist_route_release();

}  // namespace strata::kernels
