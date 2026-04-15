#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace kyle_lambda {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (two-pass algorithm)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> dp, std::span<const double> dq) {
    const size_t n = dp.size();
    if (n < 2) return 0.0;

    // Pass 1: compute means
    double mean_dp = 0.0, mean_dq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        mean_dp += dp[i];
        mean_dq += dq[i];
    }
    mean_dp /= static_cast<double>(n);
    mean_dq /= static_cast<double>(n);

    // Pass 2: compute covariance and variance
    double cov = 0.0, var = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double ddp = dp[i] - mean_dp;
        double ddq = dq[i] - mean_dq;
        cov += ddp * ddq;
        var += ddq * ddq;
    }

    // Kyle's lambda: price impact per unit volume
    constexpr double eps = 1e-12;
    return (var > eps) ? cov / var : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> dp, std::span<const double> dq) {
    using V = std::experimental::simd<double>;
    const size_t n = dp.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // ── Pass 1: vectorized mean computation ──
    V sum_dp{0.0}, sum_dq{0.0};
    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V p(&dp[i], std::experimental::element_aligned);
        V q(&dq[i], std::experimental::element_aligned);
        sum_dp += p;
        sum_dq += q;
    }

    // Manual reduction via copy_to (hsum not available in GCC 14 libstdc++)
    alignas(64) double buf_p[8], buf_q[8];
    sum_dp.copy_to(buf_p, std::experimental::element_aligned);
    sum_dq.copy_to(buf_q, std::experimental::element_aligned);

    double s_sum_dp = 0.0, s_sum_dq = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum_dp += buf_p[k];
        s_sum_dq += buf_q[k];
    }
    for (; i < n; ++i) {
        s_sum_dp += dp[i];
        s_sum_dq += dq[i];
    }

    const double mean_dp = s_sum_dp / static_cast<double>(n);
    const double mean_dq = s_sum_dq / static_cast<double>(n);

    // ── Pass 2: vectorized covariance/variance ──
    V sum_cov{0.0}, sum_var{0.0};
    V v_mean_dp(mean_dp), v_mean_dq(mean_dq);
    i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V p(&dp[i], std::experimental::element_aligned);
        V q(&dq[i], std::experimental::element_aligned);

        V ddp = p - v_mean_dp;
        V ddq = q - v_mean_dq;

        // ✅ FMA-style accumulation (compiler may fuse mul+add)
        sum_cov += ddp * ddq;
        sum_var += ddq * ddq;
    }

    // Manual reduction
    sum_cov.copy_to(buf_p, std::experimental::element_aligned);
    sum_var.copy_to(buf_q, std::experimental::element_aligned);

    double s_cov = 0.0, s_var = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_cov += buf_p[k];
        s_var += buf_q[k];
    }
    for (; i < n; ++i) {
        double ddp = dp[i] - mean_dp;
        double ddq = dq[i] - mean_dq;
        s_cov += ddp * ddq;
        s_var += ddq * ddq;
    }

    // Final division (the bottleneck: no vectorized reciprocal in std::simd)
    constexpr double eps = 1e-12;
    return (s_var > eps) ? s_cov / s_var : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> dp, std::span<const double> dq) {
    const size_t n = dp.size();
    if (n < 2) return 0.0;

    // Pass 1: means
    __m512d sum_p = _mm512_setzero_pd();
    __m512d sum_q = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        sum_p = _mm512_add_pd(sum_p, _mm512_loadu_pd(&dp[i]));
        sum_q = _mm512_add_pd(sum_q, _mm512_loadu_pd(&dq[i]));
    }
    double s_sum_p = _mm512_reduce_add_pd(sum_p);
    double s_sum_q = _mm512_reduce_add_pd(sum_q);
    for (; i < n; ++i) { s_sum_p += dp[i]; s_sum_q += dq[i]; }

    const double mean_p = s_sum_p / static_cast<double>(n);
    const double mean_q = s_sum_q / static_cast<double>(n);

    // Pass 2: cov/var with FMA
    __m512d sum_cov = _mm512_setzero_pd();
    __m512d sum_var = _mm512_setzero_pd();
    __m512d v_mean_p = _mm512_set1_pd(mean_p);
    __m512d v_mean_q = _mm512_set1_pd(mean_q);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d p = _mm512_loadu_pd(&dp[i]);
        __m512d q = _mm512_loadu_pd(&dq[i]);
        __m512d ddp = _mm512_sub_pd(p, v_mean_p);
        __m512d ddq = _mm512_sub_pd(q, v_mean_q);
        // ✅ FMA: ddp*ddq + acc in one instruction, lower latency
        sum_cov = _mm512_fmadd_pd(ddp, ddq, sum_cov);
        sum_var = _mm512_fmadd_pd(ddq, ddq, sum_var);
    }
    double s_cov = _mm512_reduce_add_pd(sum_cov);
    double s_var = _mm512_reduce_add_pd(sum_var);
    for (; i < n; ++i) {
        double ddp = dp[i] - mean_p;
        double ddq = dq[i] - mean_q;
        s_cov += ddp * ddq;
        s_var += ddq * ddq;
    }

    constexpr double eps = 1e-12;
    return (s_var > eps) ? s_cov / s_var : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> dp, std::span<const double> dq) {
    const size_t n = dp.size();
    if (n < 2) return 0.0;

    // Pass 1: means
    __m256d sum_p = _mm256_setzero_pd();
    __m256d sum_q = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        sum_p = _mm256_add_pd(sum_p, _mm256_loadu_pd(&dp[i]));
        sum_q = _mm256_add_pd(sum_q, _mm256_loadu_pd(&dq[i]));
    }
    // Manual reduction for AVX2
    sum_p = _mm256_hadd_pd(sum_p, sum_p); sum_p = _mm256_hadd_pd(sum_p, sum_p);
    sum_q = _mm256_hadd_pd(sum_q, sum_q); sum_q = _mm256_hadd_pd(sum_q, sum_q);
    double s_sum_p = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_p, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_p, 1));
    double s_sum_q = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_q, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_q, 1));
    for (; i < n; ++i) { s_sum_p += dp[i]; s_sum_q += dq[i]; }

    const double mean_p = s_sum_p / static_cast<double>(n);
    const double mean_q = s_sum_q / static_cast<double>(n);

    // Pass 2: cov/var
    __m256d sum_cov = _mm256_setzero_pd();
    __m256d sum_var = _mm256_setzero_pd();
    __m256d v_mean_p = _mm256_set1_pd(mean_p);
    __m256d v_mean_q = _mm256_set1_pd(mean_q);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d p = _mm256_loadu_pd(&dp[i]);
        __m256d q = _mm256_loadu_pd(&dq[i]);
        __m256d ddp = _mm256_sub_pd(p, v_mean_p);
        __m256d ddq = _mm256_sub_pd(q, v_mean_q);
        sum_cov = _mm256_fmadd_pd(ddp, ddq, sum_cov);
        sum_var = _mm256_fmadd_pd(ddq, ddq, sum_var);
    }
    // Manual reduction
    sum_cov = _mm256_hadd_pd(sum_cov, sum_cov); sum_cov = _mm256_hadd_pd(sum_cov, sum_cov);
    sum_var = _mm256_hadd_pd(sum_var, sum_var); sum_var = _mm256_hadd_pd(sum_var, sum_var);
    double s_cov = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_cov, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_cov, 1));
    double s_var = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_var, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_var, 1));
    for (; i < n; ++i) {
        double ddp = dp[i] - mean_p;
        double ddq = dq[i] - mean_q;
        s_cov += ddp * ddq;
        s_var += ddq * ddq;
    }

    constexpr double eps = 1e-12;
    return (s_var > eps) ? s_cov / s_var : 0.0;
}
#endif

} // namespace kyle_lambda