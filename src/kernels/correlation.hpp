#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace correlation {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Pearson correlation coefficient)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> x, std::span<const double> y) {
    const size_t n = x.size();
    if (n < 2) return 0.0;

    // Single-pass algorithm: compute all sums in one pass
    double sum_x = 0.0, sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_x  += x[i];
        sum_y  += y[i];
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }

    // Pearson correlation formula
    double numerator   = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) *
                                   (n * sum_y2 - sum_y * sum_y));

    constexpr double eps = 1e-12;
    return (denominator > eps) ? numerator / denominator : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> x, std::span<const double> y) {
    using V = std::experimental::simd<double>;
    const size_t n = x.size();
    if (n < 2) return 0.0;

    const size_t vec_len = V::size();

    // 5 independent accumulators (embarrassingly parallel)
    V sum_x{0.0}, sum_y{0.0};
    V sum_xy{0.0}, sum_x2{0.0}, sum_y2{0.0};

    size_t i = 0;
    for (; i + vec_len <= n; i += vec_len) {
        V vx(&x[i], std::experimental::element_aligned);
        V vy(&y[i], std::experimental::element_aligned);

        // ✅ All 5 accumulators are independent - compiler can ILP
        sum_x  += vx;
        sum_y  += vy;
        sum_xy += vx * vy;
        sum_x2 += vx * vx;
        sum_y2 += vy * vy;
    }

    // Manual reduction via copy_to (hsum not available in GCC 14)
    alignas(64) double buf[8];

    sum_x.copy_to(buf, std::experimental::element_aligned);
    double s_sum_x = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_x += buf[k];

    sum_y.copy_to(buf, std::experimental::element_aligned);
    double s_sum_y = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_y += buf[k];

    sum_xy.copy_to(buf, std::experimental::element_aligned);
    double s_sum_xy = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_xy += buf[k];

    sum_x2.copy_to(buf, std::experimental::element_aligned);
    double s_sum_x2 = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_x2 += buf[k];

    sum_y2.copy_to(buf, std::experimental::element_aligned);
    double s_sum_y2 = 0.0;
    for (size_t k = 0; k < vec_len; ++k) s_sum_y2 += buf[k];

    // Scalar tail
    for (; i < n; ++i) {
        s_sum_x  += x[i];
        s_sum_y  += y[i];
        s_sum_xy += x[i] * y[i];
        s_sum_x2 += x[i] * x[i];
        s_sum_y2 += y[i] * y[i];
    }

    // Final calculation (scalar sqrt + division)
    double numerator   = n * s_sum_xy - s_sum_x * s_sum_y;
    double denominator = std::sqrt((n * s_sum_x2 - s_sum_x * s_sum_x) *
                                   (n * s_sum_y2 - s_sum_y * s_sum_y));

    constexpr double eps = 1e-12;
    return (denominator > eps) ? numerator / denominator : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> x, std::span<const double> y) {
    const size_t n = x.size();
    if (n < 2) return 0.0;

    __m512d sum_x = _mm512_setzero_pd();
    __m512d sum_y = _mm512_setzero_pd();
    __m512d sum_xy = _mm512_setzero_pd();
    __m512d sum_x2 = _mm512_setzero_pd();
    __m512d sum_y2 = _mm512_setzero_pd();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d vx = _mm512_loadu_pd(&x[i]);
        __m512d vy = _mm512_loadu_pd(&y[i]);

        // ✅ FMA for product accumulations
        sum_x  = _mm512_add_pd(sum_x, vx);
        sum_y  = _mm512_add_pd(sum_y, vy);
        sum_xy = _mm512_fmadd_pd(vx, vy, sum_xy);
        sum_x2 = _mm512_fmadd_pd(vx, vx, sum_x2);
        sum_y2 = _mm512_fmadd_pd(vy, vy, sum_y2);
    }

    double s_sum_x  = _mm512_reduce_add_pd(sum_x);
    double s_sum_y  = _mm512_reduce_add_pd(sum_y);
    double s_sum_xy = _mm512_reduce_add_pd(sum_xy);
    double s_sum_x2 = _mm512_reduce_add_pd(sum_x2);
    double s_sum_y2 = _mm512_reduce_add_pd(sum_y2);

    for (; i < n; ++i) {
        s_sum_x  += x[i];
        s_sum_y  += y[i];
        s_sum_xy += x[i] * y[i];
        s_sum_x2 += x[i] * x[i];
        s_sum_y2 += y[i] * y[i];
    }

    double numerator   = n * s_sum_xy - s_sum_x * s_sum_y;
    double denominator = std::sqrt((n * s_sum_x2 - s_sum_x * s_sum_x) *
                                   (n * s_sum_y2 - s_sum_y * s_sum_y));

    constexpr double eps = 1e-12;
    return (denominator > eps) ? numerator / denominator : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> x, std::span<const double> y) {
    const size_t n = x.size();
    if (n < 2) return 0.0;

    __m256d sum_x = _mm256_setzero_pd();
    __m256d sum_y = _mm256_setzero_pd();
    __m256d sum_xy = _mm256_setzero_pd();
    __m256d sum_x2 = _mm256_setzero_pd();
    __m256d sum_y2 = _mm256_setzero_pd();

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d vx = _mm256_loadu_pd(&x[i]);
        __m256d vy = _mm256_loadu_pd(&y[i]);

        sum_x  = _mm256_add_pd(sum_x, vx);
        sum_y  = _mm256_add_pd(sum_y, vy);
        sum_xy = _mm256_fmadd_pd(vx, vy, sum_xy);
        sum_x2 = _mm256_fmadd_pd(vx, vx, sum_x2);
        sum_y2 = _mm256_fmadd_pd(vy, vy, sum_y2);
    }

    // Manual reduction for AVX2
    auto hadd4 = [](__m256d v) -> double {
        v = _mm256_hadd_pd(v, v);
        v = _mm256_hadd_pd(v, v);
        return _mm_cvtsd_f64(_mm256_extractf128_pd(v, 0)) +
               _mm_cvtsd_f64(_mm256_extractf128_pd(v, 1));
    };

    double s_sum_x  = hadd4(sum_x);
    double s_sum_y  = hadd4(sum_y);
    double s_sum_xy = hadd4(sum_xy);
    double s_sum_x2 = hadd4(sum_x2);
    double s_sum_y2 = hadd4(sum_y2);

    for (; i < n; ++i) {
        s_sum_x  += x[i];
        s_sum_y  += y[i];
        s_sum_xy += x[i] * y[i];
        s_sum_x2 += x[i] * x[i];
        s_sum_y2 += y[i] * y[i];
    }

    double numerator   = n * s_sum_xy - s_sum_x * s_sum_y;
    double denominator = std::sqrt((n * s_sum_x2 - s_sum_x * s_sum_x) *
                                   (n * s_sum_y2 - s_sum_y * s_sum_y));

    constexpr double eps = 1e-12;
    return (denominator > eps) ? numerator / denominator : 0.0;
}
#endif

} // namespace correlation