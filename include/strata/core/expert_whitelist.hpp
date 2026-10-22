// include/strata/core/expert_whitelist.hpp - the expert whitelist: which routed experts a run may use.
//
// A whitelist manifest (docs/EXPERT_WHITELIST.md, JSON) lists per MoE layer the ORIGINAL expert ids - the
// router's own numbering, `(layer, expert)` - that the run may route to and load.  The engine reads it once at
// start, checks it against the model, and then:
//
//   * the router never picks a blocked expert (its logit is -inf before the top-k, docs: "strict"), or picks as
//     the model does and hands a blocked pick to its replacement ("substitute");
//   * nothing loads a blocked expert into VRAM or RAM (the profile, the RAM copy, the adaptive tiers);
//   * in tiered mode the on_demand experts are read when picked and kept in a cache of fixed size (`OnDemandCache`);
//   * a path that still asks for one is a hard error naming the layer and the expert (`whitelist_note_violation`).
//
// Without a manifest nothing here is consulted: `active_expert_whitelist()` is null and every caller keeps its
// old path.  The shared expert is never routed and is not part of a whitelist.
//
// Host-only on purpose (no CUDA header): the tests build it with a plain C++ compiler.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

/// How blocked experts are handled (the manifest's "mode", overridable with --expert-whitelist-mode).
enum class WhitelistMode : int {
    Strict = 0,       ///< blocked experts are excluded before the top-k; the weights are normalised as the model does
    Substitute = 1,   ///< the model's own top-k; a blocked pick goes to its replacement and takes its weight along
    Tiered = 2,       ///< like strict, plus a second tier ("on_demand") that stays on the drive until it is routed
};
bool parse_whitelist_mode(const std::string& s, WhitelistMode& out);
const char* whitelist_mode_name(WhitelistMode m);

/// The tier of one original expert id.
enum : uint8_t {
    kWlBlocked = 0,    ///< never routed, never loaded
    kWlResident = 1,   ///< "allowed": routed and loaded at start (VRAM and RAM)
    kWlOnDemand = 2,   ///< tiered mode only: routed, read from the drive when the router picks it
};

/// What the engine knows about the model: the fingerprint's inputs and the ranges a manifest is checked against.
struct WhitelistModelInfo {
    int64_t n_layers = 0;
    std::vector<int32_t> experts_per_layer;    ///< per layer; 0 = no routed experts in that layer (never filtered)
    int64_t top_k = 0;
    int64_t hidden = 0;
    /// The pack index: where each layer's experts start in the experts.bin layout and the bytes of one expert, so
    /// expert (l, e) is at layer_offset[l] + e * blob_bytes[l].  Part of the fingerprint.
    std::vector<uint64_t> layer_offset, blob_bytes;
};

/// The canonical text the fingerprint hashes (tools/expert_whitelist.py builds the same text).
std::string whitelist_fingerprint_text(const WhitelistModelInfo& m);
/// "fnv1a64:" + 16 hex digits of FNV-1a 64 over `whitelist_fingerprint_text`.
std::string whitelist_fingerprint(const WhitelistModelInfo& m);
/// FNV-1a 64 of `n` bytes (the offset basis 14695981039346656037 when `h` is not given).
uint64_t whitelist_fnv1a64(const void* data, size_t n, uint64_t h = 14695981039346656037ull);

/// One loaded, validated manifest with the effective mode applied.
struct ExpertWhitelist {
    int format_version = 0;
    std::string path;
    std::string fingerprint;
    WhitelistMode manifest_mode = WhitelistMode::Strict;
    WhitelistMode mode = WhitelistMode::Strict;   ///< effective: the manifest's, or the command line's
    int64_t n_layers = 0;
    int64_t top_k = 0;
    std::vector<int32_t> experts_per_layer;
    /// [layer][expert]: kWlBlocked / kWlResident / kWlOnDemand.  A layer without routed experts has no entries.
    std::vector<std::vector<uint8_t>> tier;
    /// [layer][expert]: the allowed replacement of a blocked expert, -1 for none (substitute mode).  Empty when the
    /// manifest has no table.  Only the replacements at or above the similarity threshold (below).
    std::vector<std::vector<int32_t>> substitute;
    /// Every replacement the manifest lists, and its "similarity" (NaN where the manifest gives none), [layer][expert].
    std::vector<std::vector<int32_t>> substitute_listed;
    std::vector<std::vector<float>> substitute_similarity;
    /// The manifest's "substitute_threshold" (or the command line's): a listed replacement whose similarity is below
    /// it is not used, and its blocked expert falls back to the best allowed one.  A replacement without a similarity
    /// is always used.  `has_substitute_threshold` false: every listed replacement is used.
    bool has_substitute_threshold = false;
    double substitute_threshold = 0.0;
    // generation metadata, for the start-up report (not checked)
    std::string profile, scoring;
    int64_t calibration_tokens = -1;
    std::string created;

    bool active() const { return n_layers > 0; }
    bool moe_layer(int64_t l) const {
        return l >= 0 && l < n_layers && experts_per_layer[(size_t) l] > 0;
    }
    uint8_t tier_of(int64_t l, int64_t e) const {
        if (!moe_layer(l) || e < 0 || e >= experts_per_layer[(size_t) l]) return kWlBlocked;
        return tier[(size_t) l][(size_t) e];
    }
    /// The router may pick it in the effective mode: resident, and in tiered mode on_demand too.
    bool routable(int64_t l, int64_t e) const {
        const uint8_t t = tier_of(l, e);
        return t == kWlResident || (t == kWlOnDemand && mode == WhitelistMode::Tiered);
    }
    /// Placed in VRAM or RAM at start, and the only experts the caches and tiers may hold.
    bool resident(int64_t l, int64_t e) const { return tier_of(l, e) == kWlResident; }
    /// May be read from the pack at all (routable).  A request for anything else is a violation.
    bool loadable(int64_t l, int64_t e) const { return routable(l, e); }
    bool on_demand(int64_t l, int64_t e) const {
        return mode == WhitelistMode::Tiered && tier_of(l, e) == kWlOnDemand;
    }
    /// Experts of layer `l` in tier `t` (kWlResident, kWlOnDemand, kWlBlocked), counted as the manifest has them.
    int64_t count(int64_t l, uint8_t t) const;
    int64_t routable_count(int64_t l) const;
    int64_t total(uint8_t t) const;
    /// The replacements in use, and the listed ones the threshold leaves out.
    int64_t substitutes() const;
    int64_t substitutes_below_threshold() const;
    /// Sets the threshold (the command line's override) and rebuilds `substitute` from the listed replacements.
    void apply_substitute_threshold(bool has, double threshold);
    /// The routable table of layer `l` (1 = routable), n_expert bytes: what the routing kernels read.
    std::vector<uint8_t> routable_row(int64_t l) const;
};

/// Parses and checks a manifest.  Every rule of docs/EXPERT_WHITELIST.md section 2.2 is an error with a message
/// that names what is wrong: the format version, the fingerprint, the geometry, layer and expert ids out of range,
/// a duplicate, a missing MoE layer, fewer routable experts than top_k in a layer (top_k is never lowered).
/// `mode_override` (may be null) replaces the manifest's mode before the per-layer count is checked.
bool parse_expert_whitelist(const std::string& json, const WhitelistModelInfo& model, const WhitelistMode* mode_override,
                            ExpertWhitelist& out, std::string& err);
bool load_expert_whitelist(const std::string& path, const WhitelistModelInfo& model, const WhitelistMode* mode_override,
                           ExpertWhitelist& out, std::string& err);

/// The pack's optional index sidecar (`experts.index.json`, written by `tools/iq_pack.py index`): when the pack has
/// one it must agree with the layout the engine computed (`model.layer_offset` / `blob_bytes`).  Returns true with
/// `present` false when there is none (an older pack: the offsets are computed, as before).
bool check_expert_index_sidecar(const std::string& pack_dir, const WhitelistModelInfo& model, bool& present,
                                std::string& err);

/// Start-up report lines (no trailing newline): file, profile, mode, totals, the per-layer counts.
std::vector<std::string> whitelist_report(const ExpertWhitelist& wl);
/// The bytes the resident and on_demand tiers take, from the pack's per-layer blob size.
void whitelist_bytes(const ExpertWhitelist& wl, const std::vector<uint64_t>& blob_bytes, uint64_t& resident,
                     uint64_t& on_demand);

/// Keeps only the pairs `wl` lets the caches and tiers hold (its resident tier), in their order.  Returns how many
/// were dropped.
int64_t whitelist_filter_pairs(std::vector<std::pair<int32_t, int32_t>>& pairs, const ExpertWhitelist& wl);

// ================================ ORIGINAL ID <-> SLOT ================================

/// A compact store's explicit table: original (layer, expert) -> slot, and slot -> original.  Slots are numbered
/// in layer order, then expert order.  `check` verifies both directions agree (the tests, and debug builds at start).
struct ExpertSlotMap {
    int64_t n_layers = 0;
    int64_t stride = 0;                                   ///< the largest experts_per_layer
    std::vector<int32_t> slot_of;                         ///< [layer * stride + expert]: slot, or -1
    std::vector<std::pair<int32_t, int32_t>> id_of;       ///< [slot]: (layer, expert)

    /// The experts `wl` lets a store hold: resident, plus on_demand when `with_on_demand`.
    static ExpertSlotMap build(const ExpertWhitelist& wl, bool with_on_demand);
    int32_t slot(int64_t l, int64_t e) const {
        if (l < 0 || l >= n_layers || e < 0 || e >= stride) return -1;
        return slot_of[(size_t) (l * stride + e)];
    }
    int64_t slots() const { return (int64_t) id_of.size(); }
    bool check(std::string& err) const;
};

// ================================ SELECTIVE LOADING ================================

/// One read of the pack: `n` bytes at `off` into `dst`.  The file, a mapping, or a test's counting mock.
class BlobReader {
public:
    virtual ~BlobReader() = default;
    virtual bool read(uint64_t off, uint64_t n, uint8_t* dst) = 0;
};

/// Where each slot of a compact store comes from and goes to.
struct CompactExpertPlan {
    ExpertSlotMap map;
    std::vector<uint64_t> src_off;   ///< [slot]: the blob's offset in the pack (layer_offset + e * blob_bytes)
    std::vector<uint64_t> dst_off;   ///< [slot]: its offset in the compact store
    std::vector<uint64_t> bytes;     ///< [slot]
    uint64_t total = 0;              ///< the compact store's size
};
bool plan_compact_experts(const ExpertWhitelist& wl, const std::vector<uint64_t>& layer_offset,
                          const std::vector<uint64_t>& blob_bytes, bool with_on_demand, CompactExpertPlan& plan,
                          std::string& err);
/// Reads exactly the plan's blobs into `dst` (plan.total bytes): neighbouring blobs in one read, nothing else.
bool load_compact_experts(const CompactExpertPlan& plan, BlobReader& reader, uint8_t* dst, std::string& err);

// ================================ THE RUN'S STATE ================================

/// Counters per (layer, expert): selected by the router, executed (computed by the CPU pool or the GPU), and the
/// tiered mode's on_demand hits, misses, bytes, evictions and load time.  Cheap relaxed atomics; on with
/// --expert-whitelist-stats.
class WhitelistStats {
public:
    void reset(int64_t n_layers, int64_t n_expert);
    bool on() const { return n_layers_ > 0; }
    void selected(int64_t l, int64_t e, uint64_t n = 1) { add(sel_.get(), l, e, n); }
    void executed(int64_t l, int64_t e, uint64_t n = 1) { add(exe_.get(), l, e, n); }
    void on_demand_hit() { od_hit_.fetch_add(1, std::memory_order_relaxed); }
    void on_demand_miss(uint64_t bytes) {
        od_miss_.fetch_add(1, std::memory_order_relaxed);
        od_bytes_.fetch_add(bytes, std::memory_order_relaxed);
    }
    void on_demand_evicted(uint64_t n = 1) { od_evict_.fetch_add(n, std::memory_order_relaxed); }
    /// A miss the on_demand cache had no room for (every entry pinned): the file tier served it.
    void on_demand_no_room() { od_full_.fetch_add(1, std::memory_order_relaxed); }
    /// The time one on_demand read took (summed over the threads that read).
    void on_demand_load_ns(uint64_t ns) { od_ns_.fetch_add(ns, std::memory_order_relaxed); }
    void tokens(uint64_t n) { tokens_.fetch_add(n, std::memory_order_relaxed); }
    uint64_t selected_at(int64_t l, int64_t e) const { return get(sel_.get(), l, e); }
    uint64_t executed_at(int64_t l, int64_t e) const { return get(exe_.get(), l, e); }
    uint64_t on_demand_hits() const { return od_hit_.load(std::memory_order_relaxed); }
    uint64_t on_demand_misses() const { return od_miss_.load(std::memory_order_relaxed); }
    uint64_t on_demand_bytes() const { return od_bytes_.load(std::memory_order_relaxed); }
    uint64_t on_demand_evictions() const { return od_evict_.load(std::memory_order_relaxed); }
    uint64_t on_demand_without_room() const { return od_full_.load(std::memory_order_relaxed); }
    uint64_t on_demand_ns() const { return od_ns_.load(std::memory_order_relaxed); }
    uint64_t token_count() const { return tokens_.load(std::memory_order_relaxed); }
    /// Counts of entries outside `wl`'s routable set (both must be 0).
    uint64_t selected_outside(const ExpertWhitelist& wl) const;
    uint64_t executed_outside(const ExpertWhitelist& wl) const;
    /// Report lines: per layer the distinct ids selected and executed (and any outside the whitelist), then totals.
    std::vector<std::string> report(const ExpertWhitelist* wl, double seconds = 0.0) const;

private:
    void add(std::atomic<uint64_t>* a, int64_t l, int64_t e, uint64_t n) {
        if (a == nullptr || l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) return;
        a[(size_t) (l * n_expert_ + e)].fetch_add(n, std::memory_order_relaxed);
    }
    uint64_t get(const std::atomic<uint64_t>* a, int64_t l, int64_t e) const {
        if (a == nullptr || l < 0 || l >= n_layers_ || e < 0 || e >= n_expert_) return 0;
        return a[(size_t) (l * n_expert_ + e)].load(std::memory_order_relaxed);
    }
    int64_t n_layers_ = 0, n_expert_ = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> sel_, exe_;
    std::atomic<uint64_t> od_hit_{0}, od_miss_{0}, od_bytes_{0}, od_evict_{0}, od_full_{0}, od_ns_{0}, tokens_{0};
};

// ================================ THE TIERED MODE'S ON_DEMAND CACHE ================================

/// The on_demand tier's cache (tiered mode): a fixed byte budget that holds the on_demand experts the router picked,
/// the least recently used one out first.  It is the only place an on_demand expert is kept in RAM by the engine.
///
///   * Only on_demand experts of its whitelist go in.  A blocked one is a violation (`whitelist_note_violation`, the
///     run's hard error) and is never read; a resident one is refused (it has its own place).
///   * An entry asked for during the last `pin_epochs` epochs is never evicted (an epoch is a layer change,
///     `begin_layer`, as the expert source's stage buffers count them), so a pointer `get` returned holds through the
///     layer it was asked in and the `pin_epochs - 1` after it.  With pin_epochs 0 it holds until the next `get`.
///   * A miss that finds no room (every entry pinned) is not read: `get` answers `NoRoom` and the caller reads the
///     expert another way.  The cache never holds more than its budget.
///
/// Thread-safe: a miss is read outside the lock; a second thread asking for the same expert meanwhile waits for it.
/// The counters go to `whitelist_stats()` too when it is on (hits, misses with their bytes, evictions, no room, the
/// read time).
class OnDemandCache {
public:
    enum class Got { Hit, Miss, NoRoom, Refused, Failed };
    /// Reads one expert's blob (`bytes` long) into `dst`: the pack, or a test's counting reader.
    class Reader {
    public:
        virtual ~Reader() = default;
        virtual bool read(int64_t layer, int64_t expert, uint8_t* dst, uint64_t bytes) = 0;
    };

    OnDemandCache() = default;
    OnDemandCache(const OnDemandCache&) = delete;
    OnDemandCache& operator=(const OnDemandCache&) = delete;

    /// Refuses a whitelist that is not in tiered mode or has no on_demand expert, and a budget below the largest
    /// on_demand expert's blob (it could never hold one).
    bool init(const ExpertWhitelist& wl, const std::vector<uint64_t>& blob_bytes, uint64_t budget, uint64_t pin_epochs,
              std::string& err);
    bool active() const { return wl_ != nullptr; }
    /// A layer change starts a new epoch (the same layer again does not).
    void begin_layer(int64_t layer);
    /// The blob of (layer, expert): a hit, or a miss read now through `reader`.  Null for NoRoom, Refused and Failed
    /// (`err`, may be null, says why).
    const uint8_t* get(int64_t layer, int64_t expert, Reader& reader, Got* got = nullptr, std::string* err = nullptr);
    /// The same, but the blob is copied into `dst` (the prompt path's stager): the entry is held only while it is
    /// copied and is not pinned for the epochs, so a path that copies never keeps another one's pointers from being
    /// evicted, nor pins entries itself.  False for NoRoom, Refused and Failed.
    bool copy(int64_t layer, int64_t expert, Reader& reader, uint8_t* dst, Got* got = nullptr, std::string* err = nullptr);
    bool holds(int64_t layer, int64_t expert) const;

    uint64_t budget() const { return budget_; }
    uint64_t used() const;
    uint64_t peak() const;
    int64_t entries() const;
    uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
    uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
    uint64_t bytes_read() const { return bytes_.load(std::memory_order_relaxed); }
    uint64_t evictions() const { return evictions_.load(std::memory_order_relaxed); }
    uint64_t no_room() const { return no_room_.load(std::memory_order_relaxed); }
    uint64_t refused() const { return refused_.load(std::memory_order_relaxed); }
    uint64_t load_ns() const { return ns_.load(std::memory_order_relaxed); }

private:
    struct Entry {
        int64_t key = -1;
        std::unique_ptr<uint8_t[]> buf;
        uint64_t bytes = 0;
        uint64_t used_seq = 0;   ///< the last `get` that asked for it
        uint64_t epoch = 0;      ///< the epoch of the last `get` (a pointer handed out); copies leave it
        bool filling = false;    ///< being read (outside the lock): never a victim
        int copying = 0;         ///< `copy` calls reading it now: never a victim
        bool pinned = false;     ///< a pointer was handed out for it (`get`): the epoch rule applies
    };
    const uint8_t* fetch(int64_t layer, int64_t expert, Reader& reader, uint8_t* dst, Got* got, std::string* err);
    const ExpertWhitelist* wl_ = nullptr;
    std::vector<uint64_t> blob_bytes_;
    int64_t stride_ = 0;
    uint64_t budget_ = 0, pin_epochs_ = 0;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Entry> slots_;
    std::vector<size_t> free_;
    std::unordered_map<int64_t, size_t> index_;
    uint64_t used_ = 0, peak_ = 0, seq_ = 0, epoch_ = 0;
    int64_t last_layer_ = -1;
    std::atomic<uint64_t> hits_{0}, misses_{0}, bytes_{0}, evictions_{0}, no_room_{0}, refused_{0}, ns_{0};
};

/// The run's whitelist (null: none, the default).  Set once at start, before any expert is loaded or routed.
const ExpertWhitelist* active_expert_whitelist();
void set_active_expert_whitelist(const ExpertWhitelist* wl);
/// The MTP draft layer's own whitelist (one layer), or null: the draft layer routes over all of its experts.
const ExpertWhitelist* active_draft_whitelist();
void set_active_draft_whitelist(const ExpertWhitelist* wl);
/// --expert-whitelist-stats: the counters, or null.
WhitelistStats* whitelist_stats();
void set_whitelist_stats(WhitelistStats* s);

/// A path asked for an expert the whitelist does not allow.  The first one is kept (layer, expert, where) and
/// every later call returns false too; `whitelist_violation` reports it.  Never loads anything.
bool whitelist_note_violation(int64_t layer, int64_t expert, const char* where);
bool whitelist_violated();
/// "expert whitelist: <where> asked for expert E of layer L, which the whitelist does not allow"; "" when none.
std::string whitelist_violation();
/// Tests only: forget the latched violation.
void whitelist_violation_reset();

/// The one check every dispatch runs on a layer's routed ids: true when each is routable; otherwise notes the
/// first violation and returns false.  False too once any violation is latched.  No whitelist and none latched: true.
bool whitelist_check_ids(int64_t layer, const int32_t* ids, int64_t n, const char* where);

}  // namespace strata::core
