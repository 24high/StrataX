// sycl/src/kernels/expert_whitelist_stub.cpp - the expert whitelist's routing entry points on Intel (SYCL): not ported.
//
// The shared host code (src/program/generate.cpp, src/prefill/prefill.cpp) calls these around every router top-k
// (include/strata/kernels/expert_whitelist.hpp).  Without a whitelist they do nothing, as on CUDA.  The port has no
// migrated copy of src/kernels/cuda/expert_whitelist.cu yet, so a whitelist is refused at start rather than silently
// not applied (the same rule as metal/src/engine/expert_whitelist.cpp).
#include "strata/kernels/expert_whitelist.hpp"

#include "strata/core/expert_whitelist.hpp"

namespace strata::kernels {

bool whitelist_route_upload(int model, const strata::core::ExpertWhitelist* wl, const std::vector<int>& devices,
                            bool stats, std::string& err) {
    (void) model; (void) devices; (void) stats;
    if (wl == nullptr || !wl->active()) return true;
    err = "expert whitelist: the Intel (SYCL) build does not apply --expert-whitelist yet (its routing kernels have no "
          "mask); run without it";
    return false;
}
bool whitelist_route_active(int model) { (void) model; return false; }
void whitelist_route_mask(int, int64_t, float*, int, int, void*) {}
void whitelist_route_fix(int, int64_t, const float*, int32_t*, float*, int, int, int, void*) {}
void whitelist_ids_to_slots(int, int64_t, const int32_t*, int32_t*, int, void*) {}
bool whitelist_route_counts(int model, std::vector<uint64_t>& selected, uint64_t& unmapped, std::string& err) {
    (void) model; (void) err;
    selected.clear();
    unmapped = 0;
    return true;
}
void whitelist_route_release() {}

}  // namespace strata::kernels
