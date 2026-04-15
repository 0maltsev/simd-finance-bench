#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <vector>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cvar {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Expected Shortfall / Conditional VaR)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Copy returns for sorting (partial_sort modifies the array)
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    // Find the k-th element (e.g., 5% for 95% confidence)
    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    // Partial sort to get all elements <= k-th in first k positions
    std::partial_sort(sorted_returns.begin(),
                      sorted_returns.begin() + k + 1,
                      sorted_returns.end());

    // Sum all returns in the tail (0 to k)
    double tail_sum = 0.0;
    for (size_t i = 0; i <= k; ++i) {
        tail_sum += sorted_returns[i];
    }

    // CVaR is the negative average of tail losses
    return -tail_sum / static_cast<double>(k + 1);
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns, double confidence_level) {
    using V = std::experimental::simd<double>;
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Selection (same as VaR - cannot vectorize nth_element)
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::partial_sort(sorted_returns.begin(),
                      sorted_returns.begin() + k + 1,
                      sorted_returns.end());

    // Stage 2: Tail reduction (CAN be vectorized!)
    const size_t vec_len = V::size();
    V tail_sum{0.0};
    size_t i = 0;

    for (; i + vec_len <= k + 1; i += vec_len) {
        V tail(&sorted_returns[i], std::experimental::element_aligned);
        tail_sum += tail;
    }

    // Manual reduction
    alignas(64) double buf[8];
    tail_sum.copy_to(buf, std::experimental::element_aligned);
    double s_tail_sum = 0.0;
    for (size_t j = 0; j < vec_len; ++j) {
        s_tail_sum += buf[j];
    }

    // Scalar tail
    for (; i <= k; ++i) {
        s_tail_sum += sorted_returns[i];
    }

    return -s_tail_sum / static_cast<double>(k + 1);
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Selection (scalar)
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::partial_sort(sorted_returns.begin(),
                      sorted_returns.begin() + k + 1,
                      sorted_returns.end());

    // Stage 2: Tail reduction (vectorized)
    __m512d tail_sum = _mm512_setzero_pd();
    size_t i = 0;

    for (; i + 8 <= k + 1; i += 8) {
        __m512d tail = _mm512_loadu_pd(&sorted_returns[i]);
        tail_sum = _mm512_add_pd(tail_sum, tail);
    }

    double s_tail_sum = _mm512_reduce_add_pd(tail_sum);

    for (; i <= k; ++i) {
        s_tail_sum += sorted_returns[i];
    }

    return -s_tail_sum / static_cast<double>(k + 1);
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Stage 1: Selection (scalar)
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::partial_sort(sorted_returns.begin(),
                      sorted_returns.begin() + k + 1,
                      sorted_returns.end());

    // Stage 2: Tail reduction (vectorized)
    __m256d tail_sum = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 4 <= k + 1; i += 4) {
        __m256d tail = _mm256_loadu_pd(&sorted_returns[i]);
        tail_sum = _mm256_add_pd(tail_sum, tail);
    }

    // Manual reduction for AVX2
    tail_sum = _mm256_hadd_pd(tail_sum, tail_sum);
    tail_sum = _mm256_hadd_pd(tail_sum, tail_sum);
    double s_tail_sum = _mm_cvtsd_f64(_mm256_extractf128_pd(tail_sum, 0)) +
                        _mm_cvtsd_f64(_mm256_extractf128_pd(tail_sum, 1));

    for (; i <= k; ++i) {
        s_tail_sum += sorted_returns[i];
    }

    return -s_tail_sum / static_cast<double>(k + 1);
}
#endif

} // namespace cvar