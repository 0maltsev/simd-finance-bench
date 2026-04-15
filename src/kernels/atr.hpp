#pragma once
#include <span>
#include <cmath>
#include <algorithm>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace atr {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Wilder's smoothing)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> high, std::span<const double> low,
                     std::span<const double> close, double period) {
    const size_t n = high.size();
    if (n < 2) return 0.0;

    double beta  = (period - 1.0) / period;
    double alpha = 1.0 / period;
    double atr = std::abs(high[0] - low[0]); // Initial TR

    for (size_t i = 1; i < n; ++i) {
        double tr = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i-1]),
            std::abs(low[i] - close[i-1])
        });
        atr = atr * beta + tr * alpha;
    }
    return atr;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> high, std::span<const double> low,
                   std::span<const double> close, double period) {
    using V = std::experimental::simd<double>;
    const size_t n = high.size();
    if (n < 2) return 0.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;
    const double v_beta = beta, v_alpha = alpha;

    double atr = std::abs(high[0] - low[0]);
    size_t i = 1;
    const size_t vec_len = V::size();

    // Block-wise processing: vectorize TR calculation,
    // but maintain sequential ATR dependency chain inside each block.
    // This reflects real HFT practice where IIR filters are chunked.
    for (; i + vec_len <= n; i += vec_len) {
        V h(&high[i],   std::experimental::element_aligned);
        V l(&low[i],    std::experimental::element_aligned);
        V c(&close[i-1], std::experimental::element_aligned);

        V d1 = h - l;
        V d2 = h - c;
        V d3 = l - c;

        // ✅ P1928 abs() via max(x, -x)
        V abs_d2 = std::experimental::max(d2, -d2);
        V abs_d3 = std::experimental::max(d3, -d3);

        // Nested max for 3-way comparison
        V tr = std::experimental::max(std::experimental::max(d1, abs_d2), abs_d3);

        // Sequential smoothing within the block (dependency barrier)
        alignas(64) double tr_buf[8]; // 8 >= max simd_size for double
        tr.copy_to(tr_buf, std::experimental::element_aligned);

        for (size_t k = 0; k < vec_len; ++k) {
            atr = atr * v_beta + tr_buf[k] * v_alpha;
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        double tr = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i-1]),
            std::abs(low[i] - close[i-1])
        });
        atr = atr * beta + tr * alpha;
    }
    return atr;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> high, std::span<const double> low,
                     std::span<const double> close, double period) {
    const size_t n = high.size();
    if (n < 2) return 0.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;
    double atr = std::abs(high[0] - low[0]);

    __m512d v_beta = _mm512_set1_pd(beta);
    __m512d v_alpha = _mm512_set1_pd(alpha);
    __m512d v_zero = _mm512_setzero_pd();

    size_t i = 1;
    for (; i + 8 <= n; i += 8) {
        __m512d h = _mm512_loadu_pd(&high[i]);
        __m512d l = _mm512_loadu_pd(&low[i]);
        __m512d c = _mm512_loadu_pd(&close[i-1]);

        __m512d d1 = _mm512_sub_pd(h, l);
        __m512d d2 = _mm512_sub_pd(h, c);
        __m512d d3 = _mm512_sub_pd(l, c);

        __m512d abs_d2 = _mm512_max_pd(d2, _mm512_sub_pd(v_zero, d2));
        __m512d abs_d3 = _mm512_max_pd(d3, _mm512_sub_pd(v_zero, d3));
        __m512d tr = _mm512_max_pd(_mm512_max_pd(d1, abs_d2), abs_d3);

        alignas(64) double tr_buf[8];
        _mm512_storeu_pd(tr_buf, tr);
        for (int k = 0; k < 8; ++k) atr = atr * beta + tr_buf[k] * alpha;
    }

    for (; i < n; ++i) {
        double tr = std::max({high[i]-low[i], std::abs(high[i]-close[i-1]), std::abs(low[i]-close[i-1])});
        atr = atr * beta + tr * alpha;
    }
    return atr;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> high, std::span<const double> low,
                   std::span<const double> close, double period) {
    const size_t n = high.size();
    if (n < 2) return 0.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;
    double atr = std::abs(high[0] - low[0]);

    __m256d v_beta = _mm256_set1_pd(beta);
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_zero = _mm256_setzero_pd();

    size_t i = 1;
    for (; i + 4 <= n; i += 4) {
        __m256d h = _mm256_loadu_pd(&high[i]);
        __m256d l = _mm256_loadu_pd(&low[i]);
        __m256d c = _mm256_loadu_pd(&close[i-1]);

        __m256d d1 = _mm256_sub_pd(h, l);
        __m256d d2 = _mm256_sub_pd(h, c);
        __m256d d3 = _mm256_sub_pd(l, c);

        __m256d abs_d2 = _mm256_max_pd(d2, _mm256_sub_pd(v_zero, d2));
        __m256d abs_d3 = _mm256_max_pd(d3, _mm256_sub_pd(v_zero, d3));
        __m256d tr = _mm256_max_pd(_mm256_max_pd(d1, abs_d2), abs_d3);

        alignas(32) double tr_buf[4];
        _mm256_storeu_pd(tr_buf, tr);
        for (int k = 0; k < 4; ++k) atr = atr * beta + tr_buf[k] * alpha;
    }

    for (; i < n; ++i) {
        double tr = std::max({high[i]-low[i], std::abs(high[i]-close[i-1]), std::abs(low[i]-close[i-1])});
        atr = atr * beta + tr * alpha;
    }
    return atr;
}
#endif

} // namespace atr