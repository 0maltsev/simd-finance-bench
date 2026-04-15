#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace max_drawdown {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (cumulative returns + peak tracking)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> returns) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    double cum_return = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;

    for (size_t i = 0; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);

        // Update peak (sequential dependency)
        if (cum_return > peak) {
            peak = cum_return;
        }

        // Calculate and track drawdown
        double dd = (peak - cum_return) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> returns) {
    using V = std::experimental::simd<double>;
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // State variables (maintained across blocks)
    double cum_return = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;

    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        // Load returns for this block
        V r(&returns[i], std::experimental::element_aligned);

        // ✅ Must process sequentially within block due to dependency chain
        // This is an algorithmic barrier, not an API limitation
        alignas(64) double r_buf[8];
        r.copy_to(r_buf, std::experimental::element_aligned);

        for (size_t k = 0; k < vec_len; ++k) {
            cum_return *= (1.0 + r_buf[k]);

            if (cum_return > peak) {
                peak = cum_return;
            }

            double dd = (peak - cum_return) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);

        if (cum_return > peak) {
            peak = cum_return;
        }

        double dd = (peak - cum_return) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> returns) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    double cum_return = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d r = _mm512_loadu_pd(&returns[i]);
        alignas(64) double r_buf[8];
        _mm512_storeu_pd(r_buf, r);

        // Sequential processing within block (same as std::simd)
        for (int k = 0; k < 8; ++k) {
            cum_return *= (1.0 + r_buf[k]);

            if (cum_return > peak) {
                peak = cum_return;
            }

            double dd = (peak - cum_return) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }

    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);

        if (cum_return > peak) {
            peak = cum_return;
        }

        double dd = (peak - cum_return) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> returns) {
    const size_t n = returns.size();
    if (n < 2) return 0.0;

    double cum_return = 1.0;
    double peak = 1.0;
    double max_dd = 0.0;

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

            double dd = (peak - cum_return) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }

    for (; i < n; ++i) {
        cum_return *= (1.0 + returns[i]);

        if (cum_return > peak) {
            peak = cum_return;
        }

        double dd = (peak - cum_return) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }

    return max_dd;
}
#endif

} // namespace max_drawdown