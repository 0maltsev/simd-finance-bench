#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace alpha {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Jensen's Alpha: R_a - [R_f + β(R_m - R_f)])
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> asset_returns,
                     std::span<const double> market_returns,
                     double risk_free_rate) {
    const size_t n = asset_returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Compute Beta (same as beta::scalar)
    double sum_a = 0.0, sum_m = 0.0;
    double sum_am = 0.0, sum_m2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_a  += asset_returns[i];
        sum_m  += market_returns[i];
        sum_am += asset_returns[i] * market_returns[i];
        sum_m2 += market_returns[i] * market_returns[i];
    }

    double covariance = n * sum_am - sum_a * sum_m;
    double variance   = n * sum_m2 - sum_m * sum_m;

    constexpr double eps = 1e-12;
    double beta = (variance > eps) ? covariance / variance : 0.0;

    // Stage 2: Compute Alpha
    double mean_asset  = sum_a / static_cast<double>(n);
    double mean_market = sum_m / static_cast<double>(n);

    // Jensen's Alpha formula
    double alpha = mean_asset - (risk_free_rate + beta * (mean_market - risk_free_rate));

    return alpha;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> asset_returns,
                   std::span<const double> market_returns,
                   double risk_free_rate) {
    using V = std::experimental::simd<double>;
    const size_t n = asset_returns.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // ── Stage 1: Compute Beta (4 independent accumulators) ──
    V sum_a{0.0}, sum_m{0.0};
    V sum_am{0.0}, sum_m2{0.0};

    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V va(&asset_returns[i],  std::experimental::element_aligned);
        V vm(&market_returns[i], std::experimental::element_aligned);

        sum_a  += va;
        sum_m  += vm;
        sum_am += va * vm;
        sum_m2 += vm * vm;
    }

    // Manual reduction via copy_to
    alignas(64) double buf[8];

    sum_a.copy_to(buf, std::experimental::element_aligned);
    double s_sum_a = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_a += buf[k];

    sum_m.copy_to(buf, std::experimental::element_aligned);
    double s_sum_m = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_m += buf[k];

    sum_am.copy_to(buf, std::experimental::element_aligned);
    double s_sum_am = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_am += buf[k];

    sum_m2.copy_to(buf, std::experimental::element_aligned);
    double s_sum_m2 = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_m2 += buf[k];

    // Scalar tail
    for (; i < n; ++i) {
        s_sum_a  += asset_returns[i];
        s_sum_m  += market_returns[i];
        s_sum_am += asset_returns[i] * market_returns[i];
        s_sum_m2 += market_returns[i] * market_returns[i];
    }

    // Compute Beta
    double covariance = n * s_sum_am - s_sum_a * s_sum_m;
    double variance   = n * s_sum_m2 - s_sum_m * s_sum_m;

    constexpr double eps = 1e-12;
    double beta = (variance > eps) ? covariance / variance : 0.0;

    // ── Stage 2: Compute Alpha (scalar final calculation) ──
    double mean_asset  = s_sum_a / static_cast<double>(n);
    double mean_market = s_sum_m / static_cast<double>(n);

    double alpha = mean_asset - (risk_free_rate + beta * (mean_market - risk_free_rate));

    return alpha;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> asset_returns,
                     std::span<const double> market_returns,
                     double risk_free_rate) {
    const size_t n = asset_returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Compute Beta
    __m512d sum_a  = _mm512_setzero_pd();
    __m512d sum_m  = _mm512_setzero_pd();
    __m512d sum_am = _mm512_setzero_pd();
    __m512d sum_m2 = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&asset_returns[i]);
        __m512d vm = _mm512_loadu_pd(&market_returns[i]);

        sum_a  = _mm512_add_pd(sum_a, va);
        sum_m  = _mm512_add_pd(sum_m, vm);
        sum_am = _mm512_fmadd_pd(va, vm, sum_am);
        sum_m2 = _mm512_fmadd_pd(vm, vm, sum_m2);
    }

    double s_sum_a  = _mm512_reduce_add_pd(sum_a);
    double s_sum_m  = _mm512_reduce_add_pd(sum_m);
    double s_sum_am = _mm512_reduce_add_pd(sum_am);
    double s_sum_m2 = _mm512_reduce_add_pd(sum_m2);

    for (; i < n; ++i) {
        s_sum_a  += asset_returns[i];
        s_sum_m  += market_returns[i];
        s_sum_am += asset_returns[i] * market_returns[i];
        s_sum_m2 += market_returns[i] * market_returns[i];
    }

    double covariance = n * s_sum_am - s_sum_a * s_sum_m;
    double variance   = n * s_sum_m2 - s_sum_m * s_sum_m;

    constexpr double eps = 1e-12;
    double beta = (variance > eps) ? covariance / variance : 0.0;

    // Stage 2: Compute Alpha
    double mean_asset  = s_sum_a / static_cast<double>(n);
    double mean_market = s_sum_m / static_cast<double>(n);

    double alpha = mean_asset - (risk_free_rate + beta * (mean_market - risk_free_rate));

    return alpha;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> asset_returns,
                   std::span<const double> market_returns,
                   double risk_free_rate) {
    const size_t n = asset_returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Compute Beta
    __m256d sum_a  = _mm256_setzero_pd();
    __m256d sum_m  = _mm256_setzero_pd();
    __m256d sum_am = _mm256_setzero_pd();
    __m256d sum_m2 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&asset_returns[i]);
        __m256d vm = _mm256_loadu_pd(&market_returns[i]);

        sum_a  = _mm256_add_pd(sum_a, va);
        sum_m  = _mm256_add_pd(sum_m, vm);
        sum_am = _mm256_fmadd_pd(va, vm, sum_am);
        sum_m2 = _mm256_fmadd_pd(vm, vm, sum_m2);
    }

    // Manual reduction for AVX2
    auto hadd4 = [](__m256d v) -> double {
        v = _mm256_hadd_pd(v, v);
        v = _mm256_hadd_pd(v, v);
        return _mm_cvtsd_f64(_mm256_extractf128_pd(v, 0)) +
               _mm_cvtsd_f64(_mm256_extractf128_pd(v, 1));
    };

    double s_sum_a  = hadd4(sum_a);
    double s_sum_m  = hadd4(sum_m);
    double s_sum_am = hadd4(sum_am);
    double s_sum_m2 = hadd4(sum_m2);

    for (; i < n; ++i) {
        s_sum_a  += asset_returns[i];
        s_sum_m  += market_returns[i];
        s_sum_am += asset_returns[i] * market_returns[i];
        s_sum_m2 += market_returns[i] * market_returns[i];
    }

    double covariance = n * s_sum_am - s_sum_a * s_sum_m;
    double variance   = n * s_sum_m2 - s_sum_m * s_sum_m;

    constexpr double eps = 1e-12;
    double beta = (variance > eps) ? covariance / variance : 0.0;

    // Stage 2: Compute Alpha
    double mean_asset  = s_sum_a / static_cast<double>(n);
    double mean_market = s_sum_m / static_cast<double>(n);

    double alpha = mean_asset - (risk_free_rate + beta * (mean_market - risk_free_rate));

    return alpha;
}
#endif

} // namespace alpha