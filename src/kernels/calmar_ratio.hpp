#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace calmar_ratio {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (cumulative returns + max drawdown)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns, double periods_per_year) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    // Compute cumulative returns and track maximum drawdown
    double cum_return = 1.0;
    double peak = 1.0;
    double max_drawdown = 0.0;

    for (size_t i = 0; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);
        if (cum_return > peak) {
            peak = cum_return;
        }
        double drawdown = (peak - cum_return) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    // Annualized return
    double total_return = cum_return - 1.0;
    double annualized_return = total_return * (periods_per_year / static_cast<double>(n));

    // Calmar ratio: annualized_return / max_drawdown
    constexpr double eps = 1e-12;
    return (max_drawdown > eps) ? annualized_return / max_drawdown : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns, double periods_per_year) {
    using V = std::experimental::simd<double>;
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // Block-wise processing: vectorize cum_return calculation,
    // but maintain sequential peak/drawdown tracking across blocks.
    // This reflects real HFT practice where sequential dependencies
    // are chunked for SIMD processing.

    double cum_return = 1.0;
    double peak = 1.0;
    double max_drawdown = 0.0;

    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        // Load returns for this block
        V r(&returns[i], std::experimental::element_aligned);

        // Compute cumulative returns within block (sequential dependency)
        // We must process element-by-element within the block
        alignas(64) double r_buf[8];
        r.copy_to(r_buf, std::experimental::element_aligned);

        for (size_t k = 0; k < vec_len; ++k) {
            cum_return *= (1.0 + r_buf[k]);
            if (cum_return > peak) {
                peak = cum_return;
            }
            double drawdown = (peak - cum_return) / peak;
            if (drawdown > max_drawdown) {
                max_drawdown = drawdown;
            }
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);
        if (cum_return > peak) {
            peak = cum_return;
        }
        double drawdown = (peak - cum_return) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    // Annualized return
    double total_return = cum_return - 1.0;
    double annualized_return = total_return * (periods_per_year / static_cast<double>(n));

    constexpr double eps = 1e-12;
    return (max_drawdown > eps) ? annualized_return / max_drawdown : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns, double periods_per_year) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    double cum_return = 1.0;
    double peak = 1.0;
    double max_drawdown = 0.0;

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d r = _mm512_loadu_pd(&returns[i]);
        __m512d one = _mm512_set1_pd(1.0);
        __m512d cum_vec = _mm512_set1_pd(cum_return);

        // Compute cumulative returns within block
        // Note: Still sequential within block due to dependency
        alignas(64) double r_buf[8];
        _mm512_storeu_pd(r_buf, r);

        for (int k = 0; k < 8; ++k) {
            cum_return *= (1.0 + r_buf[k]);
            if (cum_return > peak) {
                peak = cum_return;
            }
            double drawdown = (peak - cum_return) / peak;
            if (drawdown > max_drawdown) {
                max_drawdown = drawdown;
            }
        }
    }

    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);
        if (cum_return > peak) {
            peak = cum_return;
        }
        double drawdown = (peak - cum_return) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    double total_return = cum_return - 1.0;
    double annualized_return = total_return * (periods_per_year / static_cast<double>(n));

    constexpr double eps = 1e-12;
    return (max_drawdown > eps) ? annualized_return / max_drawdown : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns, double periods_per_year) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    double cum_return = 1.0;
    double peak = 1.0;
    double max_drawdown = 0.0;

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d r = _mm256_loadu_pd(&returns[i]);
        alignas(32) double r_buf[4];
        _mm256_storeu_pd(r_buf, r);

        for (int k = 0; k < 4; ++k) {
            cum_return *= (1.0 + r_buf[k]);
            if (cum_return > peak) {
                peak = cum_return;
            }
            double drawdown = (peak - cum_return) / peak;
            if (drawdown > max_drawdown) {
                max_drawdown = drawdown;
            }
        }
    }

    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);
        if (cum_return > peak) {
            peak = cum_return;
        }
        double drawdown = (peak - cum_return) / peak;
        if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
        }
    }

    double total_return = cum_return - 1.0;
    double annualized_return = total_return * (periods_per_year / static_cast<double>(n));

    constexpr double eps = 1e-12;
    return (max_drawdown > eps) ? annualized_return / max_drawdown : 0.0;
}
#endif

} // namespace calmar_ratio