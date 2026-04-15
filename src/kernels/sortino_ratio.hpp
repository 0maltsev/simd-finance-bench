#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace sortino_ratio {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (two-pass algorithm)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns, double target_return) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Pass 1: compute mean
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += returns[i];
    }
    double mean = sum / static_cast<double>(n);

    // Pass 2: compute downside deviation (only negative returns below target)
    double sum_sq_down = 0.0;
    size_t down_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (returns[i] < target_return) {
            double diff = returns[i] - target_return;
            sum_sq_down += diff * diff;
            down_count++;
        }
    }

    // Downside deviation (annualized)
    constexpr double eps = 1e-12;
    double downside_dev = (down_count > 0)
        ? std::sqrt(sum_sq_down / static_cast<double>(n))
        : eps;

    // Sortino ratio: (mean - target) / downside_dev
    return (downside_dev > eps) ? (mean - target_return) / downside_dev : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns, double target_return) {
    using V = std::experimental::simd<double>;
    using M = typename V::mask_type;
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

    // Manual reduction via copy_to
    alignas(64) double buf[8];
    sum.copy_to(buf, std::experimental::element_aligned);

    double s_sum = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum += buf[k];
    }
    for (; i < n; ++i) {
        s_sum += returns[i];
    }

    const double mean = s_sum / static_cast<double>(n);

    // ── Pass 2: vectorized downside deviation ──
    V sum_sq_down{0.0};
    V v_target(target_return);
    i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V r(&returns[i], std::experimental::element_aligned);

        // ✅ Mask: only returns below target count
        M below_target = r < v_target;

        // Compute squared downside for all lanes
        V diff = r - v_target;
        V sq = diff * diff;

        // ✅ P1928 masked accumulation: only add where below_target
        std::experimental::where(below_target, sum_sq_down) = sum_sq_down + sq;
    }

    // Manual reduction
    sum_sq_down.copy_to(buf, std::experimental::element_aligned);

    double s_sum_sq_down = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum_sq_down += buf[k];
    }
    for (; i < n; ++i) {
        if (returns[i] < target_return) {
            double diff = returns[i] - target_return;
            s_sum_sq_down += diff * diff;
        }
    }

    // Final calculations (scalar sqrt + division)
    constexpr double eps = 1e-12;
    double downside_dev = std::sqrt(s_sum_sq_down / static_cast<double>(n));

    return (downside_dev > eps) ? (mean - target_return) / downside_dev : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns, double target_return) {
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

    // Pass 2: downside deviation with mask-add
    __m512d sum_sq_down = _mm512_setzero_pd();
    __m512d v_target = _mm512_set1_pd(target_return);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d r = _mm512_loadu_pd(&returns[i]);
        __mmask8 below = _mm512_cmp_pd_mask(r, v_target, _CMP_LT_OS);
        __m512d diff = _mm512_sub_pd(r, v_target);
        __m512d sq = _mm512_mul_pd(diff, diff);

        // ✅ AVX-512 native mask-add: only accumulate where below target
        sum_sq_down = _mm512_mask_add_pd(sum_sq_down, below, sum_sq_down, sq);
    }
    double s_sum_sq_down = _mm512_reduce_add_pd(sum_sq_down);
    for (; i < n; ++i) {
        if (returns[i] < target_return) {
            double diff = returns[i] - target_return;
            s_sum_sq_down += diff * diff;
        }
    }

    constexpr double eps = 1e-12;
    double downside_dev = std::sqrt(s_sum_sq_down / static_cast<double>(n));

    return (downside_dev > eps) ? (mean - target_return) / downside_dev : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns, double target_return) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Pass 1: mean
    __m256d sum = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        sum = _mm256_add_pd(sum, _mm256_loadu_pd(&returns[i]));
    }
    sum = _mm256_hadd_pd(sum, sum); sum = _mm256_hadd_pd(sum, sum);
    double s_sum = _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 1));
    for (; i < n; ++i) { s_sum += returns[i]; }

    const double mean = s_sum / static_cast<double>(n);

    // Pass 2: downside deviation with blend
    __m256d sum_sq_down = _mm256_setzero_pd();
    __m256d v_target = _mm256_set1_pd(target_return);
    __m256d v_zero = _mm256_setzero_pd();
    i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d r = _mm256_loadu_pd(&returns[i]);
        __m256d cmp = _mm256_cmp_pd(r, v_target, _CMP_LT_OS);
        __m256d diff = _mm256_sub_pd(r, v_target);
        __m256d sq = _mm256_mul_pd(diff, diff);

        // ✅ AVX2 blend: zero out non-downside lanes before accumulation
        __m256d masked_sq = _mm256_blendv_pd(v_zero, sq, cmp);
        sum_sq_down = _mm256_add_pd(sum_sq_down, masked_sq);
    }
    sum_sq_down = _mm256_hadd_pd(sum_sq_down, sum_sq_down); sum_sq_down = _mm256_hadd_pd(sum_sq_down, sum_sq_down);
    double s_sum_sq_down = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_sq_down, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum_sq_down, 1));
    for (; i < n; ++i) {
        if (returns[i] < target_return) {
            double diff = returns[i] - target_return;
            s_sum_sq_down += diff * diff;
        }
    }

    constexpr double eps = 1e-12;
    double downside_dev = std::sqrt(s_sum_sq_down / static_cast<double>(n));

    return (downside_dev > eps) ? (mean - target_return) / downside_dev : 0.0;
}
#endif

} // namespace sortino_ratio