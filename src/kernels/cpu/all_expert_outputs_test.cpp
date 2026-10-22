// src/kernels/cpu/all_expert_outputs_test.cpp - the expert whitelist's similarity data on the CPU (all_expert_outputs,
// pool.hpp; docs/EXPERT_WHITELIST.md 7.3) against a reference (CPU only, synthetic data).
//
// Every expert of a layer on several rows through the pool, as the engine computes the sampled rows' outputs, against
// each expert on each row alone through the single-token sequence (native: native_quant_act, native_gu_rows,
// native_quant_h, native_down_rows; the canonical Q2_0 pack: s2_expert_vnni_q).  100 synthetic experts, each with its
// own weights, and 11 rows: the groups of kMaxSplitMulti (96) experts and of MAXT (8) rows both split.  The outputs
// must be the reference's bit for bit, each in its [row][expert] place.
//
//   all_expert_outputs_test    native Q4_0 experts (any CPU with AVX); the canonical pack's only with AVX-512
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

int g_fail = 0;
#define CHECK(c)                                                                         \
    do {                                                                                 \
        if (!(c)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);            \
            ++g_fail;                                                                    \
        }                                                                                \
    } while (0)

constexpr int E = 100, N = 11;

/// The expert input rows: a different pattern per row.
std::vector<float> rows() {
    std::vector<float> x((size_t) N * cpu::H);
    for (int t = 0; t < N; ++t)
        for (int i = 0; i < cpu::H; ++i) x[(size_t) t * cpu::H + i] = (1.0f + 0.1f * (float) t) * std::sin((float) (i + 13 * t) * 0.031f);
    return x;
}

/// Compares the pool's [row][expert] outputs with `ref(t, e, out)`; returns the mismatching (row, expert) pairs.
template <class Ref>
int compare(const std::vector<float>& y, Ref&& ref, const char* what) {
    std::vector<float> one(cpu::H);
    int bad = 0;
    double worst = 0.0;
    for (int t = 0; t < N; ++t)
        for (int e = 0; e < E; ++e) {
            ref(t, e, one.data());
            const float* got = y.data() + ((size_t) t * E + e) * cpu::H;
            if (std::memcmp(got, one.data(), one.size() * sizeof(float)) != 0) {
                double num = 0.0, den = 0.0;
                for (int i = 0; i < cpu::H; ++i) {
                    num += (double) (got[i] - one[i]) * (got[i] - one[i]);
                    den += (double) one[i] * one[i];
                }
                worst = std::max(worst, std::sqrt(num / (den > 0 ? den : 1.0)));
                ++bad;
            }
        }
    std::printf("all_expert_outputs_test: %s: %d experts x %d rows, %d differ from the reference%s", what, E, N, bad,
                bad ? "" : " (bit for bit)\n");
    if (bad) std::printf(" (worst relative error %.3e)\n", worst);
    return bad;
}

/// The experts must differ from one another, or a mix-up of their places could not be seen.
void check_distinct(const std::vector<float>& y) {
    int same = 0;
    for (int t = 0; t < N; ++t)
        for (int e = 0; e + 1 < E; ++e)
            same += std::memcmp(y.data() + ((size_t) t * E + e) * cpu::H, y.data() + ((size_t) t * E + e + 1) * cpu::H,
                                cpu::H * sizeof(float)) == 0;
    CHECK(same == 0);
}

void test_native() {
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cpu::H, cpu::FF, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        ++g_fail;
        return;
    }
    // each expert its own gate, up and down weights
    std::vector<std::vector<uint8_t>> blobs((size_t) E, std::vector<uint8_t>(f.bytes));
    std::vector<float> w((size_t) cpu::H * cpu::FF);
    for (int e = 0; e < E; ++e) {
        const float ph[3] = {0.7f * (float) e, 1.3f + 0.5f * (float) e, 2.9f + 0.3f * (float) e};
        const size_t off[3] = {0, f.up_off, f.down_off};
        for (int m = 0; m < 3; ++m) {
            for (size_t i = 0; i < w.size(); ++i) w[i] = 0.02f * std::sin((float) i * (0.17f + 0.001f * (float) m) + ph[m]);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, w.data(), blobs[(size_t) e].data() + off[m], 0, m < 2 ? cpu::FF : cpu::H,
                                m < 2 ? cpu::H : cpu::FF, nullptr);
        }
    }
    const std::vector<float> x = rows();
    std::vector<float> y((size_t) N * E * cpu::H, std::nanf(""));
    cpu::ExpertPool pool(3, /*pin=*/false);
    CHECK(cpu::all_expert_outputs(pool, &f, E, [&](int64_t e) { return blobs[(size_t) e].data(); }, x.data(), N, y.data(), err));
    // the reference: one expert, one row, the single-token sequence
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
    std::vector<float> ff(cpu::FF);
    auto ref = [&](int t, int e, float* out) {
        cpu::native_quant_act(f, x.data() + (size_t) t * cpu::H, act.data());
        const void* a[1] = {act.data()};
        float* fp[1] = {ff.data()};
        cpu::native_gu_rows(f, blobs[(size_t) e].data(), a, 1, fp, 0, cpu::FF);
        cpu::native_quant_h(f, ff.data(), hq.data());
        const void* hp[1] = {hq.data()};
        float* op[1] = {out};
        cpu::native_down_rows(f, blobs[(size_t) e].data(), hp, 1, op, 0, cpu::H);
    };
    CHECK(compare(y, ref, "native Q4_0") == 0);
    check_distinct(y);
    for (float v : y) CHECK(std::isfinite(v));
    // an expert without a blob stops it with its number; no rows: nothing to do
    CHECK(!cpu::all_expert_outputs(pool, &f, E, [&](int64_t e) { return e == 97 ? nullptr : blobs[(size_t) e].data(); },
                                   x.data(), N, y.data(), err) &&
          err.find("expert 97") != std::string::npos);
    std::vector<float> none(1, 7.0f);
    CHECK(cpu::all_expert_outputs(pool, &f, E, [&](int64_t e) { return blobs[(size_t) e].data(); }, x.data(), 0, none.data(), err) &&
          none[0] == 7.0f);
}

void test_canonical() {
    if (const cpu::CpuFeatures feat = cpu::cpu_features(); !feat.usable()) {
        std::printf("all_expert_outputs_test: the canonical Q2_0 kernels need %s: not tested on this CPU\n", feat.reason());
        return;
    }
    std::mt19937 rng(5);
    std::vector<uint8_t> blobs((size_t) E * cpu::BLOB);
    for (int e = 0; e < E; ++e) {   // random codes, fp16 scales ~0.004-0.016 (expert_multi_test's synthetic blobs)
        uint8_t* b = &blobs[(size_t) e * cpu::BLOB];
        for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
        for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
            const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
            std::memcpy(b + i, &h, 2);
        }
    }
    const std::vector<float> x = rows();
    std::vector<float> y((size_t) N * E * cpu::H, std::nanf(""));
    cpu::ExpertPool pool(3, /*pin=*/false);
    std::string err;
    CHECK(cpu::all_expert_outputs(pool, nullptr, E, [&](int64_t e) { return &blobs[(size_t) e * cpu::BLOB]; }, x.data(), N,
                                  y.data(), err));
    static cpu::ActQ a;
    static cpu::ExpertScratch ws;
    auto ref = [&](int t, int e, float* out) {
        cpu::act_quant_q8_1(x.data() + (size_t) t * cpu::H, cpu::H, a);
        cpu::s2_expert_vnni_q(&blobs[(size_t) e * cpu::BLOB], a, out, ws);
    };
    CHECK(compare(y, ref, "canonical Q2_0") == 0);
    check_distinct(y);
}

}  // namespace

int main() {
    // the activation quantizers are the AVX ones at least (act_quant_any): a CPU without AVX cannot run them
    if (!cpu::cpu_avx1_ok()) {
        std::printf("all_expert_outputs_test: no AVX on this CPU: SKIPPED\n");
        return 77;
    }
    test_native();
    test_canonical();
    if (g_fail) {
        std::fprintf(stderr, "all_expert_outputs_test: %d check(s) FAILED\n", g_fail);
        return 1;
    }
    std::puts("all_expert_outputs_test: passed");
    return 0;
}
