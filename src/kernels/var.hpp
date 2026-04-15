#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <vector>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace var {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Historical VaR via nth_element)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Copy returns for sorting (nth_element modifies the array)
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    // Find the percentile index (e.g., 5% for 95% confidence)
    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    // Partial sort to find k-th element (O(n) average)
    std::nth_element(sorted_returns.begin(),
                     sorted_returns.begin() + k,
                     sorted_returns.end());

    // VaR is the negative of the k-th return (loss is positive)
    return -sorted_returns[k];
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns, double confidence_level) {
    using V = std::experimental::simd<double>;
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // ⚠️ VaR requires sorting/selection - SIMD cannot accelerate nth_element
    // However, we can use SIMD for pre-processing (abs, negation, etc.)
    // For this implementation, we use the same scalar approach as baseline
    // to demonstrate the algorithmic barrier.

    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::nth_element(sorted_returns.begin(),
                     sorted_returns.begin() + k,
                     sorted_returns.end());

    return -sorted_returns[k];
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // ⚠️ Same algorithmic barrier - sorting/selection is scalar
    // Intrinsics cannot accelerate nth_element meaningfully
    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::nth_element(sorted_returns.begin(),
                     sorted_returns.begin() + k,
                     sorted_returns.end());

    return -sorted_returns[k];
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns, double confidence_level) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    std::vector<double> sorted_returns(returns.begin(), returns.end());

    size_t k = static_cast<size_t>((1.0 - confidence_level) * n);
    k = std::min(k, n - 1);

    std::nth_element(sorted_returns.begin(),
                     sorted_returns.begin() + k,
                     sorted_returns.end());

    return -sorted_returns[k];
}
#endif

} // namespace var