#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sharpe_ratio {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (two-pass algorithm)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns, double risk_free_rate) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Pass 1: compute mean
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += returns[i];
    }
    double mean = sum / static_cast<double>(n);

    // Pass 2: compute standard deviation
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = returns[i] - mean;
        sum_sq += diff * diff;
    }
    double std_dev = std::sqrt(sum_sq / static_cast<double>(n - 1));

    // Sharpe ratio: (mean - risk_free) / std_dev
    constexpr double eps = 1e-12;
    return (std_dev > eps) ? (mean - risk_free_rate) / std_dev : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns, double risk_free_rate) {
    using V = std::experimental::simd<double>;
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // ── Pass 1: vectorized mean computation ──
    V sum{0.0};
    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V r(&returns[i], std::experimental::element_aligned);
        sum += r;
    }

    // Manual reduction via copy_to (hsum not available in GCC 14 libstdc++)
    alignas(64) double buf[8]; // 8 >= max simd_size for double on x86
    sum.copy_to(buf, std::experimental::element_aligned);

    double s_sum = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum += buf[k];
    }
    for (; i < n; ++i) {
        s_sum += returns[i];
    }

    const double mean = s_sum / static_cast<double>(n);

    // ── Pass 2: vectorized variance computation ──
    V sum_sq{0.0};
    V v_mean(mean);
    i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V r(&returns[i], std::experimental::element_aligned);
        V diff = r - v_mean;
        sum_sq += diff * diff;
    }

    // Manual reduction
    sum_sq.copy_to(buf, std::experimental::element_aligned);

    double s_sum_sq = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum_sq += buf[k];
    }
    for (; i < n; ++i) {
        double diff = returns[i] - mean;
        s_sum_sq += diff * diff;
    }

    // Final calculations (scalar sqrt + division)
    double std_dev = std::sqrt(s_sum_sq / static_cast<double>(n - 1));

    constexpr double eps = 1e-12;
    return (std_dev > eps) ? (mean - risk_free_rate) / std_dev : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns, double risk_free_rate) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Pass 1: mean
    __m512d sum = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        sum = _mm512_add_pd(sum, _mm512_loadu_pd(&returns[i]));
    }
    double s_sum = _mm512_reduce_add_pd(sum);
    for (; i < n; ++i) { s_sum += returns[i]; }

    const double mean = s_sum / static_cast<double>(n);

    // Pass 2: variance with FMA
    __m512d sum_sq = _mm512_setzero_pd();
    __m512d v_mean = _mm512_set1_pd(mean);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d r = _mm512_loadu_pd(&returns[i]);
        __m512d diff = _mm512_sub_pd(r, v_mean);
        // ✅ FMA: diff*diff + acc in one instruction
        sum_sq = _mm512_fmadd_pd(diff, diff, sum_sq);
    }
    double s_sum_sq = _mm512_reduce_add_pd(sum_sq);
    for (; i < n; ++i) {
        double diff = returns[i] - mean;
        s_sum_sq += diff * diff;
    }

    // Final calculations (scalar sqrt - no vectorized sqrt in intrinsics for single value)
    double std_dev = std::sqrt(s_sum_sq / static_cast<double>(n - 1));

    constexpr double eps = 1e-12;
    return (std_dev > eps) ? (mean - risk_free_rate) / std_dev : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns, double risk_free_rate) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Pass 1: mean
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        sum = _mm256_add_pd(sum, _mm256_loadu_pd(&returns[i]));
    }
    // Manual reduction for AVX2
    sum = _mm256_hadd_pd(sum, sum); sum = _mm256_hadd_pd(sum, sum);
    double s_sum = _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 1));
    for (; i < n; ++i) { s_sum += returns[i]; }

    const double mean = s_sum / static_cast<double>(n);

    // Pass 2: variance
    __m256d sum_sq = _mm256_setzero_pd();
    __m256d v_mean = _mm256_set1_pd(mean);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d r = _mm256_loadu_pd(&returns[i]);
        __m256d diff = _mm256_sub_pd(r, v_mean);
        sum_sq = _mm256_fmadd_pd(diff, diff, sum_sq);
    }
    // Manual reduction
    sum_sq = _mm256_hadd_pd(sum_sq, sum_sq); sum_sq = _mm256_hadd_pd(sum_sq, sum_sq);
    double s_sum_sq = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_sq, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_sq, 1));
    for (; i < n; ++i) {
        double diff = returns[i] - mean;
        s_sum_sq += diff * diff;
    }

    double std_dev = std::sqrt(s_sum_sq / static_cast<double>(n - 1));

    constexpr double eps = 1e-12;
    return (std_dev > eps) ? (mean - risk_free_rate) / std_dev : 0.0;
}
#endif

} // namespace sharpe_ratio