// src/core/expert_whitelist.cpp - the expert whitelist manifest: parse, check, report (expert_whitelist.hpp).
#include "strata/core/expert_whitelist.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>

namespace strata::core {

// ================================ A SMALL, STRICT JSON READER ================================
//
// Enough for the two files this feature reads (the manifest and the pack's index sidecar): RFC 8259 values, a
// duplicate key is an error (a manifest that says a thing twice is not one we can trust either way), numbers keep
// whether they were written as integers.
namespace {

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0.0;
    bool integer = false;   ///< written without a fraction or an exponent, and it fits int64
    int64_t i = 0;
    std::string s;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* get(const char* key) const {
        if (kind != Obj) return nullptr;
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

class JsonReader {
public:
    explicit JsonReader(const std::string& t) : t_(t) {}
    bool parse(Json& out, std::string& err) {
        ws();
        if (!value(out, 0)) { err = msg(); return false; }
        ws();
        if (p_ != t_.size()) { why_ = "text after the JSON value"; err = msg(); return false; }
        return true;
    }

private:
    std::string msg() const {
        size_t line = 1;
        for (size_t k = 0; k < p_ && k < t_.size(); ++k) line += t_[k] == '\n';
        return "JSON line " + std::to_string(line) + ": " + why_;
    }
    void ws() {
        while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
    }
    bool fail(const char* w) { why_ = w; return false; }
    bool lit(const char* w) {
        const size_t n = std::strlen(w);
        if (t_.compare(p_, n, w) != 0) return false;
        p_ += n;
        return true;
    }
    bool value(Json& v, int depth) {
        if (depth > 64) return fail("nested too deep");
        if (p_ >= t_.size()) return fail("unexpected end");
        const char c = t_[p_];
        if (c == '{') return object(v, depth);
        if (c == '[') return array(v, depth);
        if (c == '"') { v.kind = Json::Str; return string(v.s); }
        if (c == 't') { if (!lit("true")) return fail("bad literal"); v.kind = Json::Bool; v.b = true; return true; }
        if (c == 'f') { if (!lit("false")) return fail("bad literal"); v.kind = Json::Bool; v.b = false; return true; }
        if (c == 'n') { if (!lit("null")) return fail("bad literal"); v.kind = Json::Null; return true; }
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return fail("unexpected character");
    }
    bool object(Json& v, int depth) {
        v.kind = Json::Obj;
        ++p_;
        ws();
        if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
        for (;;) {
            ws();
            if (p_ >= t_.size() || t_[p_] != '"') return fail("expected a key");
            std::string key;
            if (!string(key)) return false;
            for (const auto& kv : v.obj)
                if (kv.first == key) { why_ = "duplicate key \"" + key + "\""; return false; }
            ws();
            if (p_ >= t_.size() || t_[p_] != ':') return fail("expected ':'");
            ++p_;
            ws();
            Json child;
            if (!value(child, depth + 1)) return false;
            v.obj.emplace_back(std::move(key), std::move(child));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == '}') { ++p_; return true; }
            return fail("expected ',' or '}'");
        }
    }
    bool array(Json& v, int depth) {
        v.kind = Json::Arr;
        ++p_;
        ws();
        if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
        for (;;) {
            ws();
            Json child;
            if (!value(child, depth + 1)) return false;
            v.arr.push_back(std::move(child));
            ws();
            if (p_ < t_.size() && t_[p_] == ',') { ++p_; continue; }
            if (p_ < t_.size() && t_[p_] == ']') { ++p_; return true; }
            return fail("expected ',' or ']'");
        }
    }
    static void utf8(std::string& o, uint32_t cp) {
        if (cp < 0x80) o += (char) cp;
        else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F));
        } else {
            o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F));
            o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F));
        }
    }
    bool hex4(uint32_t& cp) {
        if (p_ + 4 > t_.size()) return fail("short \\u escape");
        cp = 0;
        for (int k = 0; k < 4; ++k) {
            const char h = t_[p_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= (uint32_t) (h - '0');
            else if (h >= 'a' && h <= 'f') cp |= (uint32_t) (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F') cp |= (uint32_t) (h - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }
    bool string(std::string& o) {
        ++p_;   // the opening quote
        for (;;) {
            if (p_ >= t_.size()) return fail("unterminated string");
            const char c = t_[p_++];
            if (c == '"') return true;
            if ((unsigned char) c < 0x20) return fail("control character in a string");
            if (c != '\\') { o += c; continue; }
            if (p_ >= t_.size()) return fail("unterminated escape");
            const char x = t_[p_++];
            switch (x) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        uint32_t lo = 0;
                        if (p_ + 2 > t_.size() || t_[p_] != '\\' || t_[p_ + 1] != 'u') return fail("lone surrogate");
                        p_ += 2;
                        if (!hex4(lo) || lo < 0xDC00 || lo > 0xDFFF) return fail("bad surrogate pair");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("lone surrogate");
                    }
                    utf8(o, cp);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }
    bool number(Json& v) {
        const size_t s = p_;
        if (t_[p_] == '-') ++p_;
        if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') return fail("bad number");
        if (t_[p_] == '0') ++p_;
        else while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        bool frac = false;
        if (p_ < t_.size() && t_[p_] == '.') {
            frac = true;
            ++p_;
            if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') return fail("bad number");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
            frac = true;
            ++p_;
            if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
            if (p_ >= t_.size() || t_[p_] < '0' || t_[p_] > '9') return fail("bad number");
            while (p_ < t_.size() && t_[p_] >= '0' && t_[p_] <= '9') ++p_;
        }
        const std::string text = t_.substr(s, p_ - s);
        v.kind = Json::Num;
        v.num = std::strtod(text.c_str(), nullptr);
        v.integer = false;
        if (!frac && text.size() <= 19) {   // fits int64 for sure below 19 digits; 19 digits checked by the parse
            errno = 0;
            char* end = nullptr;
            const long long ll = std::strtoll(text.c_str(), &end, 10);
            if (errno == 0 && end != nullptr && *end == '\0') { v.integer = true; v.i = (int64_t) ll; }
        }
        return true;
    }
    const std::string& t_;
    size_t p_ = 0;
    std::string why_;
};

bool read_text(const std::string& path, std::string& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::ostringstream ss;
    ss << f.rdbuf();
    if (!f.good() && !f.eof()) { err = "cannot read " + path; return false; }
    out = ss.str();
    return true;
}

std::string hex16(uint64_t h) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) h);
    return b;
}

bool int_of(const Json* v, int64_t& out) {
    if (v == nullptr || v->kind != Json::Num || !v->integer) return false;
    out = v->i;
    return true;
}

std::mutex g_violation_mu;
std::atomic<bool> g_violated{false};
int64_t g_violation_layer = -1, g_violation_expert = -1;
std::string g_violation_where;
std::atomic<const ExpertWhitelist*> g_active{nullptr};
std::atomic<const ExpertWhitelist*> g_draft{nullptr};
std::atomic<WhitelistStats*> g_stats{nullptr};

/// The latched violation as one sentence (g_violation_mu held).
std::string whitelist_violation_text() {
    if (g_violation_expert < 0)
        return "expert whitelist: " + g_violation_where + " (layer " + std::to_string(g_violation_layer) +
               "): the run cannot keep to the whitelist";
    return "expert whitelist: " + g_violation_where + " asked for expert " + std::to_string(g_violation_expert) +
           " of layer " + std::to_string(g_violation_layer) + ", which the whitelist does not allow";
}

}  // namespace

// ================================ MODE, FINGERPRINT ================================

bool parse_whitelist_mode(const std::string& s, WhitelistMode& out) {
    if (s == "strict") { out = WhitelistMode::Strict; return true; }
    if (s == "substitute") { out = WhitelistMode::Substitute; return true; }
    if (s == "tiered") { out = WhitelistMode::Tiered; return true; }
    return false;
}

const char* whitelist_mode_name(WhitelistMode m) {
    switch (m) {
        case WhitelistMode::Strict: return "strict";
        case WhitelistMode::Substitute: return "substitute";
        case WhitelistMode::Tiered: return "tiered";
    }
    return "?";
}

uint64_t whitelist_fnv1a64(const void* data, size_t n, uint64_t h) {
    const auto* p = (const unsigned char*) data;
    for (size_t k = 0; k < n; ++k) {
        h ^= p[k];
        h *= 1099511628211ull;
    }
    return h;
}

std::string whitelist_fingerprint_text(const WhitelistModelInfo& m) {
    // One line per field, decimal numbers, no spaces: the Python side (tools/expert_whitelist.py) writes the same.
    std::string t = "strata-expert-whitelist-fingerprint-v1\n";
    t += "num_layers=" + std::to_string(m.n_layers) + "\n";
    t += "top_k=" + std::to_string(m.top_k) + "\n";
    t += "hidden=" + std::to_string(m.hidden) + "\n";
    t += "experts_per_layer=";
    for (size_t l = 0; l < m.experts_per_layer.size(); ++l) t += (l ? "," : "") + std::to_string(m.experts_per_layer[l]);
    t += "\npack_index=";
    for (size_t l = 0; l < m.layer_offset.size(); ++l) {
        t += (l ? ";" : "") + std::to_string(l) + ":" + std::to_string(m.layer_offset[l]) + ":" +
             std::to_string(l < m.blob_bytes.size() ? m.blob_bytes[l] : 0);
    }
    t += "\n";
    return t;
}

std::string whitelist_fingerprint(const WhitelistModelInfo& m) {
    const std::string t = whitelist_fingerprint_text(m);
    return "fnv1a64:" + hex16(whitelist_fnv1a64(t.data(), t.size()));
}

// ================================ THE WHITELIST ================================

int64_t ExpertWhitelist::count(int64_t l, uint8_t t) const {
    if (!moe_layer(l)) return 0;
    return (int64_t) std::count(tier[(size_t) l].begin(), tier[(size_t) l].end(), t);
}

int64_t ExpertWhitelist::routable_count(int64_t l) const {
    if (!moe_layer(l)) return 0;
    int64_t n = 0;
    for (int32_t e = 0; e < experts_per_layer[(size_t) l]; ++e) n += routable(l, e);
    return n;
}

int64_t ExpertWhitelist::total(uint8_t t) const {
    int64_t n = 0;
    for (int64_t l = 0; l < n_layers; ++l) n += count(l, t);
    return n;
}

int64_t ExpertWhitelist::substitutes() const {
    int64_t n = 0;
    for (const auto& row : substitute) n += (int64_t) std::count_if(row.begin(), row.end(), [](int32_t s) { return s >= 0; });
    return n;
}

int64_t ExpertWhitelist::substitutes_below_threshold() const {
    int64_t listed = 0;
    for (const auto& row : substitute_listed)
        listed += (int64_t) std::count_if(row.begin(), row.end(), [](int32_t s) { return s >= 0; });
    return listed - substitutes();
}

void ExpertWhitelist::apply_substitute_threshold(bool has, double threshold) {
    has_substitute_threshold = has;
    substitute_threshold = has ? threshold : 0.0;
    substitute = substitute_listed;
    for (size_t l = 0; l < substitute.size(); ++l)
        for (size_t e = 0; e < substitute[l].size(); ++e) {
            const float sim = substitute_similarity[l][e];
            // a listed replacement without a similarity is the manifest author's choice: always used
            if (has && substitute[l][e] >= 0 && sim == sim && (double) sim < threshold) substitute[l][e] = -1;
        }
}

std::vector<uint8_t> ExpertWhitelist::routable_row(int64_t l) const {
    std::vector<uint8_t> r;
    if (!moe_layer(l)) return r;
    r.resize((size_t) experts_per_layer[(size_t) l]);
    for (int32_t e = 0; e < experts_per_layer[(size_t) l]; ++e) r[(size_t) e] = routable(l, e) ? 1 : 0;
    return r;
}

namespace {

/// Reads one id list ("allowed", "on_demand") of layer `l` into `tier` as `t`.
bool read_ids(const Json* list, const char* name, int64_t l, int32_t n_expert, uint8_t t, std::vector<uint8_t>& tier,
              std::string& err) {
    if (list == nullptr) return true;
    if (list->kind != Json::Arr) { err = "layer " + std::to_string(l) + ": \"" + name + "\" must be a list of expert ids"; return false; }
    for (const Json& v : list->arr) {
        int64_t e = 0;
        if (!int_of(&v, e)) { err = "layer " + std::to_string(l) + ": \"" + name + "\" holds a value that is not an integer expert id"; return false; }
        if (e < 0 || e >= n_expert) {
            err = "layer " + std::to_string(l) + ": expert id " + std::to_string(e) + " in \"" + name +
                  "\" is out of range (the layer has experts 0.." + std::to_string(n_expert - 1) + ")";
            return false;
        }
        uint8_t& slot = tier[(size_t) e];
        if (slot == t) {
            err = "layer " + std::to_string(l) + ": expert " + std::to_string(e) + " is listed twice in \"" + name + "\"";
            return false;
        }
        if (slot != kWlBlocked) {
            err = "layer " + std::to_string(l) + ": expert " + std::to_string(e) + " is both \"allowed\" and \"on_demand\"";
            return false;
        }
        slot = t;
    }
    return true;
}

}  // namespace

bool parse_expert_whitelist(const std::string& json, const WhitelistModelInfo& model, const WhitelistMode* mode_override,
                            ExpertWhitelist& out, std::string& err) {
    out = ExpertWhitelist{};
    Json root;
    {
        JsonReader r(json);
        if (!r.parse(root, err)) return false;
    }
    if (root.kind != Json::Obj) { err = "the manifest is not a JSON object"; return false; }
    if (const Json* kind = root.get("kind"); kind != nullptr &&
        (kind->kind != Json::Str || kind->s != "strata-expert-whitelist")) {
        err = "\"kind\" is not \"strata-expert-whitelist\": this is not an expert whitelist manifest";
        return false;
    }
    int64_t version = 0;
    if (!int_of(root.get("format_version"), version)) { err = "\"format_version\" is missing or not an integer"; return false; }
    if (version != 1) {
        err = "unknown format_version " + std::to_string(version) + " (this engine reads format_version 1)";
        return false;
    }
    out.format_version = (int) version;

    // ---- the model: the fingerprint first, it is the one that says "wrong model" most plainly
    const Json* fp = root.get("model_fingerprint");
    if (fp == nullptr || fp->kind != Json::Str || fp->s.empty()) { err = "\"model_fingerprint\" is missing"; return false; }
    const std::string want = whitelist_fingerprint(model);
    if (fp->s != want) {
        err = "the whitelist was made for another model or pack (its model_fingerprint is " + fp->s + ", this model's is " +
              want + ")";
        return false;
    }
    out.fingerprint = fp->s;
    int64_t nl = 0, k = 0;
    if (!int_of(root.get("num_layers"), nl)) { err = "\"num_layers\" is missing or not an integer"; return false; }
    if (nl != model.n_layers) {
        err = "\"num_layers\" is " + std::to_string(nl) + ", the model has " + std::to_string(model.n_layers);
        return false;
    }
    if (!int_of(root.get("top_k"), k)) { err = "\"top_k\" is missing or not an integer"; return false; }
    if (k != model.top_k) {
        err = "\"top_k\" is " + std::to_string(k) + ", the model routes " + std::to_string(model.top_k);
        return false;
    }
    out.n_layers = nl;
    out.top_k = k;
    out.experts_per_layer = model.experts_per_layer;
    if (const Json* epl = root.get("experts_per_layer"); epl == nullptr) {
        err = "\"experts_per_layer\" is missing";
        return false;
    } else if (epl->kind == Json::Num) {
        int64_t n = 0;
        if (!int_of(epl, n)) { err = "\"experts_per_layer\" is not an integer"; return false; }
        for (int64_t l = 0; l < nl; ++l)
            if (model.experts_per_layer[(size_t) l] > 0 && model.experts_per_layer[(size_t) l] != n) {
                err = "\"experts_per_layer\" is " + std::to_string(n) + ", layer " + std::to_string(l) + " of the model has " +
                      std::to_string(model.experts_per_layer[(size_t) l]);
                return false;
            }
    } else if (epl->kind == Json::Arr) {
        if ((int64_t) epl->arr.size() != nl) { err = "\"experts_per_layer\" lists " + std::to_string(epl->arr.size()) + " layers, the model has " + std::to_string(nl); return false; }
        for (int64_t l = 0; l < nl; ++l) {
            int64_t n = 0;
            if (!int_of(&epl->arr[(size_t) l], n) || n != model.experts_per_layer[(size_t) l]) {
                err = "\"experts_per_layer\"[" + std::to_string(l) + "] does not match the model (" +
                      std::to_string(model.experts_per_layer[(size_t) l]) + ")";
                return false;
            }
        }
    } else {
        err = "\"experts_per_layer\" must be an integer or a list";
        return false;
    }

    // ---- the mode
    if (const Json* m = root.get("mode"); m != nullptr) {
        if (m->kind != Json::Str || !parse_whitelist_mode(m->s, out.manifest_mode)) {
            err = "unknown \"mode\" (expected strict, substitute or tiered)";
            return false;
        }
    }
    out.mode = mode_override != nullptr ? *mode_override : out.manifest_mode;

    // ---- the layers
    out.tier.assign((size_t) nl, {});
    for (int64_t l = 0; l < nl; ++l)
        if (out.experts_per_layer[(size_t) l] > 0) out.tier[(size_t) l].assign((size_t) out.experts_per_layer[(size_t) l], kWlBlocked);
    std::vector<uint8_t> seen((size_t) nl, 0);
    bool any_sub = false;
    std::vector<std::vector<int32_t>> sub((size_t) nl);
    std::vector<std::vector<float>> sim((size_t) nl);
    bool has_threshold = false;
    double threshold = 0.0;
    if (const Json* t = root.get("substitute_threshold"); t != nullptr && t->kind != Json::Null) {
        if (t->kind != Json::Num || !std::isfinite(t->num) || t->num < -1.0 || t->num > 1.0) {
            err = "\"substitute_threshold\" must be a number from -1 to 1 (the similarity a replacement needs)";
            return false;
        }
        has_threshold = true;
        threshold = t->num;
    }
    const Json* layers = root.get("layers");
    if (layers == nullptr || layers->kind != Json::Arr) { err = "\"layers\" is missing or not a list"; return false; }
    for (const Json& L : layers->arr) {
        if (L.kind != Json::Obj) { err = "an entry of \"layers\" is not an object"; return false; }
        int64_t l = 0;
        if (!int_of(L.get("layer"), l)) { err = "an entry of \"layers\" has no integer \"layer\""; return false; }
        if (l < 0 || l >= nl) {
            err = "layer id " + std::to_string(l) + " is out of range (the model has layers 0.." + std::to_string(nl - 1) + ")";
            return false;
        }
        if (seen[(size_t) l]) { err = "layer " + std::to_string(l) + " is listed twice"; return false; }
        seen[(size_t) l] = 1;
        const int32_t ne = out.experts_per_layer[(size_t) l];
        if (ne <= 0) { err = "layer " + std::to_string(l) + " has no routed experts and cannot be filtered"; return false; }
        if (L.get("allowed") == nullptr) { err = "layer " + std::to_string(l) + " has no \"allowed\" list"; return false; }
        if (!read_ids(L.get("allowed"), "allowed", l, ne, kWlResident, out.tier[(size_t) l], err)) return false;
        if (!read_ids(L.get("on_demand"), "on_demand", l, ne, kWlOnDemand, out.tier[(size_t) l], err)) return false;
        if (const Json* S = L.get("substitutes"); S != nullptr) {
            if (S->kind != Json::Arr) { err = "layer " + std::to_string(l) + ": \"substitutes\" must be a list"; return false; }
            std::vector<int32_t>& row = sub[(size_t) l];
            row.assign((size_t) ne, -1);
            sim[(size_t) l].assign((size_t) ne, std::numeric_limits<float>::quiet_NaN());
            for (const Json& s : S->arr) {
                int64_t e = 0, w = 0;
                if (s.kind != Json::Obj || !int_of(s.get("expert"), e) || !int_of(s.get("with"), w)) {
                    err = "layer " + std::to_string(l) + ": a substitute needs integer \"expert\" and \"with\"";
                    return false;
                }
                const Json* sv = s.get("similarity");
                if (sv != nullptr && (sv->kind != Json::Num || !std::isfinite(sv->num) || sv->num < -1.0 || sv->num > 1.0)) {
                    err = "layer " + std::to_string(l) + ": the substitute of expert " + std::to_string(e) +
                          " has a \"similarity\" that is not a number from -1 to 1";
                    return false;
                }
                if (e < 0 || e >= ne || w < 0 || w >= ne) {
                    err = "layer " + std::to_string(l) + ": substitute " + std::to_string(e) + " -> " + std::to_string(w) +
                          " is out of range (the layer has experts 0.." + std::to_string(ne - 1) + ")";
                    return false;
                }
                if (row[(size_t) e] >= 0) { err = "layer " + std::to_string(l) + ": expert " + std::to_string(e) + " has two substitutes"; return false; }
                if (out.tier[(size_t) l][(size_t) e] == kWlResident) {
                    err = "layer " + std::to_string(l) + ": expert " + std::to_string(e) + " is allowed and cannot have a substitute";
                    return false;
                }
                if (out.tier[(size_t) l][(size_t) w] != kWlResident) {
                    err = "layer " + std::to_string(l) + ": the substitute of expert " + std::to_string(e) + " is expert " +
                          std::to_string(w) + ", which is not in \"allowed\"";
                    return false;
                }
                row[(size_t) e] = (int32_t) w;
                if (sv != nullptr) sim[(size_t) l][(size_t) e] = (float) sv->num;
                any_sub = true;
            }
        }
        if (const Json* sc = L.get("scores"); sc != nullptr && sc->kind != Json::Obj && sc->kind != Json::Arr) {
            err = "layer " + std::to_string(l) + ": \"scores\" must be an object or a list";
            return false;
        }
    }
    for (int64_t l = 0; l < nl; ++l)
        if (out.experts_per_layer[(size_t) l] > 0 && !seen[(size_t) l]) {
            err = "layer " + std::to_string(l) + " is an MoE layer and is missing from \"layers\"";
            return false;
        }
    if (any_sub) {
        for (int64_t l = 0; l < nl; ++l)
            if (sub[(size_t) l].empty() && out.experts_per_layer[(size_t) l] > 0) {
                sub[(size_t) l].assign((size_t) out.experts_per_layer[(size_t) l], -1);
                sim[(size_t) l].assign((size_t) out.experts_per_layer[(size_t) l], std::numeric_limits<float>::quiet_NaN());
            }
        out.substitute_listed = std::move(sub);
        out.substitute_similarity = std::move(sim);
    }
    out.apply_substitute_threshold(has_threshold, threshold);
    // ---- every layer can still route top_k: never lower it, never route a blocked expert to fill it
    for (int64_t l = 0; l < nl; ++l) {
        if (out.experts_per_layer[(size_t) l] <= 0) continue;
        const int64_t n = out.routable_count(l);
        if (n < k) {
            err = "layer " + std::to_string(l) + " allows " + std::to_string(n) + " experts in " + whitelist_mode_name(out.mode) +
                  " mode, fewer than the model's top_k " + std::to_string(k) + " (top_k is not lowered)";
            return false;
        }
    }

    // ---- what made it (reported, not checked)
    if (const Json* g = root.get("generation"); g != nullptr && g->kind == Json::Obj) {
        if (const Json* p = g->get("profile"); p && p->kind == Json::Str) out.profile = p->s;
        if (const Json* c = g->get("created"); c && c->kind == Json::Str) out.created = c->s;
        int64_t ct = 0;
        if (int_of(g->get("calibration_tokens"), ct)) out.calibration_tokens = ct;
        if (const Json* s = g->get("scoring"); s && s->kind == Json::Obj) {
            if (const Json* mth = s->get("method"); mth && mth->kind == Json::Str) out.scoring = mth->s;
        } else if (s && s->kind == Json::Str) {
            out.scoring = s->s;
        }
    }
    return true;
}

bool load_expert_whitelist(const std::string& path, const WhitelistModelInfo& model, const WhitelistMode* mode_override,
                           ExpertWhitelist& out, std::string& err) {
    std::string text;
    if (!read_text(path, text, err)) return false;
    if (!parse_expert_whitelist(text, model, mode_override, out, err)) {
        err = path + ": " + err;
        return false;
    }
    out.path = path;
    return true;
}

bool check_expert_index_sidecar(const std::string& pack_dir, const WhitelistModelInfo& model, bool& present,
                                std::string& err) {
    present = false;
    const std::string path = pack_dir + "/experts.index.json";
    std::ifstream probe(path, std::ios::binary);
    if (!probe) return true;   // an older pack: the offsets are computed from the layout, as before
    probe.close();
    present = true;
    std::string text;
    if (!read_text(path, text, err)) return false;
    Json root;
    {
        JsonReader r(text);
        if (!r.parse(root, err)) { err = path + ": " + err; return false; }
    }
    int64_t version = 0, nl = 0;
    if (root.kind != Json::Obj || !int_of(root.get("format_version"), version) || version != 1) {
        err = path + ": not an expert index of format_version 1";
        return false;
    }
    if (!int_of(root.get("num_layers"), nl) || nl != model.n_layers) {
        err = path + ": \"num_layers\" does not match the pack's layout (" + std::to_string(model.n_layers) + ")";
        return false;
    }
    const Json* layers = root.get("layers");
    if (layers == nullptr || layers->kind != Json::Arr || (int64_t) layers->arr.size() != nl) {
        err = path + ": \"layers\" must list every layer";
        return false;
    }
    for (int64_t l = 0; l < nl; ++l) {
        const Json& L = layers->arr[(size_t) l];
        int64_t layer = -1, ne = -1, off = -1, bb = -1;
        if (L.kind != Json::Obj || !int_of(L.get("layer"), layer) || !int_of(L.get("n_expert"), ne) ||
            !int_of(L.get("offset"), off) || !int_of(L.get("blob_bytes"), bb) || layer != l) {
            err = path + ": layer " + std::to_string(l) + "'s entry is incomplete or out of order";
            return false;
        }
        if (ne != model.experts_per_layer[(size_t) l] || (uint64_t) off != model.layer_offset[(size_t) l] ||
            (uint64_t) bb != model.blob_bytes[(size_t) l]) {
            err = path + ": layer " + std::to_string(l) + " says " + std::to_string(ne) + " experts at offset " +
                  std::to_string(off) + ", " + std::to_string(bb) + " B each; the pack's layout has " +
                  std::to_string(model.experts_per_layer[(size_t) l]) + " at " + std::to_string(model.layer_offset[(size_t) l]) +
                  ", " + std::to_string(model.blob_bytes[(size_t) l]) + " B (the index is stale: run tools/iq_pack.py index again)";
            return false;
        }
    }
    return true;
}

// ================================ REPORT, FILTER ================================

void whitelist_bytes(const ExpertWhitelist& wl, const std::vector<uint64_t>& blob_bytes, uint64_t& resident,
                     uint64_t& on_demand) {
    resident = on_demand = 0;
    for (int64_t l = 0; l < wl.n_layers; ++l) {
        const uint64_t b = (size_t) l < blob_bytes.size() ? blob_bytes[(size_t) l] : 0;
        resident += (uint64_t) wl.count(l, kWlResident) * b;
        if (wl.mode == WhitelistMode::Tiered) on_demand += (uint64_t) wl.count(l, kWlOnDemand) * b;
    }
}

std::vector<std::string> whitelist_report(const ExpertWhitelist& wl) {
    std::vector<std::string> lines;
    char b[512];
    std::snprintf(b, sizeof b, "expert whitelist %s: profile %s, mode %s%s, made %s", wl.path.c_str(),
                  wl.profile.empty() ? "(none given)" : wl.profile.c_str(), whitelist_mode_name(wl.mode),
                  wl.mode != wl.manifest_mode ? " (set on the command line)" : "",
                  wl.created.empty() ? "(date not given)" : wl.created.c_str());
    lines.emplace_back(b);
    int64_t moe = 0, all = 0;
    for (int64_t l = 0; l < wl.n_layers; ++l)
        if (wl.moe_layer(l)) { ++moe; all += wl.experts_per_layer[(size_t) l]; }
    const int64_t res = wl.total(kWlResident), od = wl.total(kWlOnDemand);
    std::snprintf(b, sizeof b, "expert whitelist: %lld of %lld routed experts allowed (%.1f%%) in %lld MoE layers, top-%lld%s",
                  (long long) res, (long long) all, all > 0 ? 100.0 * (double) res / (double) all : 0.0, (long long) moe,
                  (long long) wl.top_k,
                  wl.mode == WhitelistMode::Tiered ? "" : (od > 0 ? "; the on_demand tier is blocked in this mode" : ""));
    lines.emplace_back(b);
    if (wl.mode == WhitelistMode::Tiered) {
        std::snprintf(b, sizeof b, "expert whitelist: tiered: %lld more on_demand (read from the drive when routed)", (long long) od);
        lines.emplace_back(b);
    }
    if (wl.mode == WhitelistMode::Substitute) {
        std::snprintf(b, sizeof b, "expert whitelist: substitute: %lld blocked experts have a replacement, the rest fall back to "
                                   "the best allowed one", (long long) wl.substitutes());
        lines.emplace_back(b);
        if (wl.has_substitute_threshold) {
            std::snprintf(b, sizeof b, "expert whitelist: substitute: similarity threshold %.4f; %lld listed replacements are "
                                       "below it and not used", wl.substitute_threshold, (long long) wl.substitutes_below_threshold());
            lines.emplace_back(b);
        }
    }
    // the per-layer counts, eight layers to a line
    std::string row;
    for (int64_t l = 0; l < wl.n_layers; ++l) {
        if (!wl.moe_layer(l)) continue;
        std::snprintf(b, sizeof b, "%s%lld:%lld", row.empty() ? "" : " ", (long long) l, (long long) wl.routable_count(l));
        row += b;
        if ((l + 1) % 12 == 0 || l + 1 == wl.n_layers) {
            lines.emplace_back("expert whitelist: per layer (layer:experts) " + row);
            row.clear();
        }
    }
    if (!row.empty()) lines.emplace_back("expert whitelist: per layer (layer:experts) " + row);
    return lines;
}

int64_t whitelist_filter_pairs(std::vector<std::pair<int32_t, int32_t>>& pairs, const ExpertWhitelist& wl) {
    const size_t before = pairs.size();
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [&](const std::pair<int32_t, int32_t>& p) { return !wl.resident(p.first, p.second); }),
                pairs.end());
    return (int64_t) (before - pairs.size());
}

// ================================ SLOTS, SELECTIVE LOADING ================================

ExpertSlotMap ExpertSlotMap::build(const ExpertWhitelist& wl, bool with_on_demand) {
    ExpertSlotMap m;
    m.n_layers = wl.n_layers;
    for (int32_t n : wl.experts_per_layer) m.stride = std::max<int64_t>(m.stride, n);
    m.slot_of.assign((size_t) (m.n_layers * m.stride), -1);
    for (int64_t l = 0; l < wl.n_layers; ++l) {
        if (!wl.moe_layer(l)) continue;
        for (int32_t e = 0; e < wl.experts_per_layer[(size_t) l]; ++e) {
            const uint8_t t = wl.tier_of(l, e);
            if (t == kWlResident || (with_on_demand && t == kWlOnDemand && wl.mode == WhitelistMode::Tiered)) {
                m.slot_of[(size_t) (l * m.stride + e)] = (int32_t) m.id_of.size();
                m.id_of.emplace_back((int32_t) l, e);
            }
        }
    }
    return m;
}

bool ExpertSlotMap::check(std::string& err) const {
    if ((int64_t) slot_of.size() != n_layers * stride) { err = "slot map: table size"; return false; }
    int64_t mapped = 0;
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < stride; ++e) {
            const int32_t s = slot_of[(size_t) (l * stride + e)];
            if (s < 0) continue;
            ++mapped;
            if (s >= (int32_t) id_of.size() || id_of[(size_t) s].first != l || id_of[(size_t) s].second != e) {
                err = "slot map: expert " + std::to_string(e) + " of layer " + std::to_string(l) + " maps to slot " +
                      std::to_string(s) + ", which does not map back";
                return false;
            }
        }
    if (mapped != (int64_t) id_of.size()) { err = "slot map: a slot has no original id"; return false; }
    return true;
}

bool plan_compact_experts(const ExpertWhitelist& wl, const std::vector<uint64_t>& layer_offset,
                          const std::vector<uint64_t>& blob_bytes, bool with_on_demand, CompactExpertPlan& plan,
                          std::string& err) {
    plan = CompactExpertPlan{};
    if ((int64_t) layer_offset.size() < wl.n_layers || (int64_t) blob_bytes.size() < wl.n_layers) {
        err = "compact experts: the pack layout has fewer layers than the whitelist";
        return false;
    }
    plan.map = ExpertSlotMap::build(wl, with_on_demand);
    if (!plan.map.check(err)) return false;
    const size_t n = plan.map.id_of.size();
    plan.src_off.resize(n);
    plan.dst_off.resize(n);
    plan.bytes.resize(n);
    for (size_t s = 0; s < n; ++s) {
        const auto [l, e] = plan.map.id_of[s];
        plan.bytes[s] = blob_bytes[(size_t) l];
        plan.src_off[s] = layer_offset[(size_t) l] + (uint64_t) e * blob_bytes[(size_t) l];
        plan.dst_off[s] = plan.total;
        plan.total += plan.bytes[s];
    }
    return true;
}

bool load_compact_experts(const CompactExpertPlan& plan, BlobReader& reader, uint8_t* dst, std::string& err) {
    const size_t n = plan.src_off.size();
    for (size_t s = 0; s < n;) {
        // a run of slots whose blobs are neighbours in the pack AND in the store: one read
        size_t t = s + 1;
        uint64_t len = plan.bytes[s];
        while (t < n && plan.src_off[t] == plan.src_off[s] + len && plan.dst_off[t] == plan.dst_off[s] + len) {
            len += plan.bytes[t];
            ++t;
        }
        if (!reader.read(plan.src_off[s], len, dst + plan.dst_off[s])) {
            err = "compact experts: reading " + std::to_string(len) + " B at offset " + std::to_string(plan.src_off[s]) +
                  " (layer " + std::to_string(plan.map.id_of[s].first) + ", expert " + std::to_string(plan.map.id_of[s].second) +
                  ") failed";
            return false;
        }
        s = t;
    }
    return true;
}

// ================================ STATS ================================

void WhitelistStats::reset(int64_t n_layers, int64_t n_expert) {
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    const size_t n = (size_t) (n_layers * n_expert);
    sel_.reset(n ? new std::atomic<uint64_t>[n] : nullptr);
    exe_.reset(n ? new std::atomic<uint64_t>[n] : nullptr);
    for (size_t k = 0; k < n; ++k) {
        sel_[k].store(0, std::memory_order_relaxed);
        exe_[k].store(0, std::memory_order_relaxed);
    }
    od_hit_ = od_miss_ = od_bytes_ = od_evict_ = od_full_ = od_ns_ = tokens_ = 0;
}

uint64_t WhitelistStats::selected_outside(const ExpertWhitelist& wl) const {
    uint64_t n = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e)
            if (!wl.routable(l, e)) n += selected_at(l, e);
    return n;
}

uint64_t WhitelistStats::executed_outside(const ExpertWhitelist& wl) const {
    uint64_t n = 0;
    for (int64_t l = 0; l < n_layers_; ++l)
        for (int64_t e = 0; e < n_expert_; ++e)
            if (!wl.routable(l, e)) n += executed_at(l, e);
    return n;
}

std::vector<std::string> WhitelistStats::report(const ExpertWhitelist* wl, double seconds) const {
    std::vector<std::string> lines;
    char b[256];
    uint64_t sel_all = 0, exe_all = 0, sel_out = 0, exe_out = 0;
    for (int64_t l = 0; l < n_layers_; ++l) {
        int64_t ds = 0, de = 0, os = 0, oe = 0;
        uint64_t ns = 0, nx = 0;
        for (int64_t e = 0; e < n_expert_; ++e) {
            const uint64_t s = selected_at(l, e), x = executed_at(l, e);
            ns += s;
            nx += x;
            ds += s > 0;
            de += x > 0;
            if (wl != nullptr && !wl->routable(l, e)) { os += s > 0; oe += x > 0; sel_out += s; exe_out += x; }
        }
        sel_all += ns;
        exe_all += nx;
        std::snprintf(b, sizeof b, "expert whitelist stats: layer %lld: %llu selected (%lld distinct ids), %llu executed "
                                   "(%lld distinct), outside the whitelist %lld / %lld", (long long) l,
                      (unsigned long long) ns, (long long) ds, (unsigned long long) nx, (long long) de, (long long) os,
                      (long long) oe);
        lines.emplace_back(b);
    }
    std::snprintf(b, sizeof b, "expert whitelist stats: total %llu selected, %llu executed; outside the whitelist %llu selected, "
                               "%llu executed", (unsigned long long) sel_all, (unsigned long long) exe_all,
                  (unsigned long long) sel_out, (unsigned long long) exe_out);
    lines.emplace_back(b);
    const uint64_t hit = od_hit_.load(), miss = od_miss_.load(), bytes = od_bytes_.load(), tok = tokens_.load();
    const uint64_t evict = od_evict_.load(), full = od_full_.load(), ns = od_ns_.load();
    if (hit + miss > 0) {
        std::snprintf(b, sizeof b, "expert whitelist stats: on_demand %llu hits, %llu misses, %.1f MiB read (%.3f MiB a token)",
                      (unsigned long long) hit, (unsigned long long) miss, (double) bytes / 1048576.0,
                      tok > 0 ? (double) bytes / 1048576.0 / (double) tok : 0.0);
        lines.emplace_back(b);
    }
    if (evict + full + ns > 0) {
        std::snprintf(b, sizeof b, "expert whitelist stats: on_demand cache: %llu evictions, %llu misses without room (read past "
                                   "the cache), reading took %.3f s (summed over threads)", (unsigned long long) evict,
                      (unsigned long long) full, (double) ns * 1e-9);
        lines.emplace_back(b);
    }
    if (seconds > 0.0 && tok > 0) {
        std::snprintf(b, sizeof b, "expert whitelist stats: %llu tokens in %.2f s (%.2f tokens/s)", (unsigned long long) tok,
                      seconds, (double) tok / seconds);
        lines.emplace_back(b);
        const double load_s = (double) ns * 1e-9;
        if (load_s > 0.0 && load_s < seconds) {
            // the reads' share of the run: an upper bound on what they cost (summed over threads, some overlapped)
            std::snprintf(b, sizeof b, "expert whitelist stats: on_demand reads took %.1f%% of the run; without them at most "
                                       "%.2f tokens/s (a calculation, not a measurement)", 100.0 * load_s / seconds,
                          (double) tok / (seconds - load_s));
            lines.emplace_back(b);
        }
    }
    return lines;
}

// ================================ THE ON_DEMAND CACHE ================================

bool OnDemandCache::init(const ExpertWhitelist& wl, const std::vector<uint64_t>& blob_bytes, uint64_t budget,
                         uint64_t pin_epochs, std::string& err) {
    std::lock_guard<std::mutex> lk(mu_);
    wl_ = nullptr;
    slots_.clear();
    free_.clear();
    index_.clear();
    used_ = peak_ = seq_ = epoch_ = 0;
    last_layer_ = -1;
    if (wl.mode != WhitelistMode::Tiered || wl.total(kWlOnDemand) == 0) {
        err = "the on_demand cache is for a tiered whitelist with on_demand experts";
        return false;
    }
    uint64_t largest = 0;
    for (int64_t l = 0; l < wl.n_layers; ++l)
        if (wl.count(l, kWlOnDemand) > 0) {
            if ((size_t) l >= blob_bytes.size() || blob_bytes[(size_t) l] == 0) {
                err = "the on_demand cache: no blob size for layer " + std::to_string(l);
                return false;
            }
            largest = std::max(largest, blob_bytes[(size_t) l]);
        }
    if (budget < largest) {
        err = "the on_demand cache of " + std::to_string(budget >> 10) + " KiB cannot hold one on_demand expert (" +
              std::to_string(largest >> 10) + " KiB)";
        return false;
    }
    blob_bytes_ = blob_bytes;
    stride_ = 0;
    for (int32_t n : wl.experts_per_layer) stride_ = std::max<int64_t>(stride_, n);
    budget_ = budget;
    pin_epochs_ = pin_epochs;
    wl_ = &wl;
    return true;
}

void OnDemandCache::begin_layer(int64_t layer) {
    std::lock_guard<std::mutex> lk(mu_);
    if (layer != last_layer_) {
        ++epoch_;
        last_layer_ = layer;
    }
}

bool OnDemandCache::holds(int64_t layer, int64_t expert) const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = index_.find(layer * stride_ + expert);
    return it != index_.end() && !slots_[it->second].filling;
}

uint64_t OnDemandCache::used() const { std::lock_guard<std::mutex> lk(mu_); return used_; }
uint64_t OnDemandCache::peak() const { std::lock_guard<std::mutex> lk(mu_); return peak_; }
int64_t OnDemandCache::entries() const { std::lock_guard<std::mutex> lk(mu_); return (int64_t) index_.size(); }

const uint8_t* OnDemandCache::get(int64_t layer, int64_t expert, Reader& reader, Got* got, std::string* err) {
    return fetch(layer, expert, reader, nullptr, got, err);
}

bool OnDemandCache::copy(int64_t layer, int64_t expert, Reader& reader, uint8_t* dst, Got* got, std::string* err) {
    return dst != nullptr && fetch(layer, expert, reader, dst, got, err) != nullptr;
}

/// `get` (dst null: a pointer, pinned for the epochs) and `copy` (dst: the blob copied, the entry held while it is).
const uint8_t* OnDemandCache::fetch(int64_t layer, int64_t expert, Reader& reader, uint8_t* dst, Got* got,
                                    std::string* err) {
    auto answer = [&](Got g, const char* why) -> const uint8_t* {
        if (got != nullptr) *got = g;
        if (err != nullptr && why != nullptr) *err = why;
        return nullptr;
    };
    if (wl_ == nullptr) return answer(Got::Refused, "the on_demand cache is not set up");
    if (!wl_->on_demand(layer, expert)) {
        refused_.fetch_add(1, std::memory_order_relaxed);
        // a blocked expert is the run's hard error and is never read; a resident one has its own place
        if (!wl_->routable(layer, expert)) {
            whitelist_note_violation(layer, expert, "the on_demand cache");
            return answer(Got::Refused, "the whitelist does not allow this expert");
        }
        return answer(Got::Refused, "a resident expert is not the on_demand cache's");
    }
    const bool pointer = dst == nullptr;
    WhitelistStats* ws = whitelist_stats();
    const int64_t key = layer * stride_ + expert;
    const uint64_t bytes = blob_bytes_[(size_t) layer];
    size_t v = 0;
    uint8_t* buf = nullptr;
    bool hit = false;
    // a copy out of an entry, outside the lock (its `copying` count keeps it), then the count goes
    auto copy_out = [&](size_t slot, uint8_t* from) {
        std::memcpy(dst, from, (size_t) bytes);
        std::lock_guard<std::mutex> lk(mu_);
        --slots_[slot].copying;
    };
    {
        std::unique_lock<std::mutex> lk(mu_);
        const uint64_t seq = ++seq_;
        auto it = index_.find(key);
        if (it != index_.end() && slots_[it->second].filling) {
            // another thread reads it: wait for that.  Gone afterwards (that read failed, or the entry was evicted as soon
            // as it was read): a miss of our own
            cv_.wait(lk, [&] { const auto j = index_.find(key); return j == index_.end() || !slots_[j->second].filling; });
            it = index_.find(key);
        }
        if (it != index_.end()) {
            v = it->second;
            Entry& s = slots_[v];
            s.used_seq = seq;
            if (pointer) {
                s.epoch = epoch_;
                s.pinned = true;
            } else {
                ++s.copying;
            }
            hits_.fetch_add(1, std::memory_order_relaxed);
            if (ws != nullptr) ws->on_demand_hit();
            if (got != nullptr) *got = Got::Hit;
            hit = true;
            buf = s.buf.get();
            if (pointer) return buf;
        } else {
            // a miss: make room, least recently used first, never an entry being read or copied, nor one whose pointer
            // was handed out during the pinned epochs
            uint64_t evicted = 0;
            while (used_ + bytes > budget_) {
                size_t victim = slots_.size();
                for (size_t i = 0; i < slots_.size(); ++i) {
                    const Entry& s = slots_[i];
                    if (s.key < 0 || s.filling || s.copying > 0 || (s.pinned && s.epoch + pin_epochs_ > epoch_)) continue;
                    if (victim == slots_.size() || s.used_seq < slots_[victim].used_seq) victim = i;
                }
                if (victim == slots_.size()) {
                    no_room_.fetch_add(1, std::memory_order_relaxed);
                    if (ws != nullptr) ws->on_demand_no_room();
                    return answer(Got::NoRoom, "every entry of the on_demand cache is in use");
                }
                Entry& s = slots_[victim];
                index_.erase(s.key);
                used_ -= s.bytes;
                s.key = -1;
                free_.push_back(victim);
                ++evicted;
            }
            evictions_.fetch_add(evicted, std::memory_order_relaxed);
            if (ws != nullptr && evicted > 0) ws->on_demand_evicted(evicted);
            // a free slot, reusing its buffer when it has the right size
            if (!free_.empty()) {
                v = free_.back();
                free_.pop_back();
            } else {
                v = slots_.size();
                slots_.emplace_back();
            }
            Entry& s = slots_[v];
            if (s.bytes != bytes || !s.buf) {
                s.buf.reset(new (std::nothrow) uint8_t[(size_t) bytes]);
                if (!s.buf) {
                    s.bytes = 0;
                    free_.push_back(v);
                    return answer(Got::Failed, "out of memory for an on_demand expert");
                }
                s.bytes = bytes;
            }
            s.key = key;
            s.used_seq = seq;
            s.pinned = pointer;
            s.epoch = pointer ? epoch_ : 0;
            s.copying = pointer ? 0 : 1;
            s.filling = true;
            index_[key] = v;
            used_ += bytes;
            peak_ = std::max(peak_, used_);
            buf = s.buf.get();   // the buffer stays where it is when slots_ grows (another thread's miss)
        }
    }
    if (!hit) {
        // the miss's read, outside the lock (another thread asking for this expert waits on cv_)
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = reader.read(layer, expert, buf, bytes);
        const uint64_t ns = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(mu_);
            Entry& s = slots_[v];
            s.filling = false;
            if (!ok) {
                index_.erase(key);
                used_ -= s.bytes;
                s.key = -1;
                s.copying = 0;
                free_.push_back(v);
            }
        }
        cv_.notify_all();
        if (!ok) return answer(Got::Failed, "reading the on_demand expert from the pack failed");
        misses_.fetch_add(1, std::memory_order_relaxed);
        bytes_.fetch_add(bytes, std::memory_order_relaxed);
        ns_.fetch_add(ns, std::memory_order_relaxed);
        if (ws != nullptr) {
            ws->on_demand_miss(bytes);
            ws->on_demand_load_ns(ns);
        }
        if (got != nullptr) *got = Got::Miss;
    }
    if (!pointer) copy_out(v, buf);
    return buf;
}

// ================================ THE RUN'S STATE ================================

const ExpertWhitelist* active_expert_whitelist() { return g_active.load(std::memory_order_acquire); }
void set_active_expert_whitelist(const ExpertWhitelist* wl) { g_active.store(wl && wl->active() ? wl : nullptr, std::memory_order_release); }
const ExpertWhitelist* active_draft_whitelist() { return g_draft.load(std::memory_order_acquire); }
void set_active_draft_whitelist(const ExpertWhitelist* wl) { g_draft.store(wl && wl->active() ? wl : nullptr, std::memory_order_release); }
WhitelistStats* whitelist_stats() { return g_stats.load(std::memory_order_acquire); }
void set_whitelist_stats(WhitelistStats* s) { g_stats.store(s, std::memory_order_release); }

bool whitelist_note_violation(int64_t layer, int64_t expert, const char* where) {
    std::lock_guard<std::mutex> lock(g_violation_mu);
    if (!g_violated.load(std::memory_order_relaxed)) {
        g_violation_layer = layer;
        g_violation_expert = expert;
        g_violation_where = where != nullptr ? where : "a path";
        g_violated.store(true, std::memory_order_release);
        std::fprintf(stderr, "strata: %s\n", whitelist_violation_text().c_str());   // once: the first one decides
    }
    return false;
}

bool whitelist_violated() { return g_violated.load(std::memory_order_acquire); }

std::string whitelist_violation() {
    std::lock_guard<std::mutex> lock(g_violation_mu);
    if (!g_violated.load(std::memory_order_relaxed)) return {};
    return whitelist_violation_text();
}

void whitelist_violation_reset() {
    std::lock_guard<std::mutex> lock(g_violation_mu);
    g_violated.store(false, std::memory_order_release);
    g_violation_layer = g_violation_expert = -1;
    g_violation_where.clear();
}

bool whitelist_check_ids(int64_t layer, const int32_t* ids, int64_t n, const char* where) {
    if (whitelist_violated()) return false;   // anything latched before (a GPU's router, the draft layer) fails the run here
    const ExpertWhitelist* wl = active_expert_whitelist();
    if (wl == nullptr) return true;
    for (int64_t i = 0; i < n; ++i)
        if (!wl->routable(layer, ids[i])) return whitelist_note_violation(layer, ids[i], where);
    return true;
}

}  // namespace strata::core
