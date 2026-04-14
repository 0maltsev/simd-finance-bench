#pragma once
#include <span>
#include <cmath>
#include <numeric>
#include <experimental/simd>

namespace vwap {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> prices, std::span<const double> vols) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < prices.size(); ++i) {
        num += prices[i] * vols[i];
        den += vols[i];
    }
    constexpr double eps = 1e-9;
    return (den > eps) ? num / den : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental)
// ─────────────────────────────────────────────────────────────
    inline double simd(std::span<const double> prices, std::span<const double> vols) {
    using V = std::experimental::simd<double>;
    V num{0.0}, den{0.0};
    size_t i = 0;
    const size_t vec_len = V::size();

    for (; i + vec_len <= prices.size(); i += vec_len) {
        V p(&prices[i], std::experimental::element_aligned);
        V v(&vols[i],   std::experimental::element_aligned);
        num = num + (p * v);
        den = den + v;
    }

    alignas(64) double buf_num[V::size()];
    alignas(64) double buf_den[V::size()];
    num.copy_to(buf_num, std::experimental::element_aligned);
    den.copy_to(buf_den, std::experimental::element_aligned);

    double s_num = 0.0, s_den = 0.0;
    for (size_t k = 0; k < V::size(); ++k) {
        s_num += buf_num[k];
        s_den += buf_den[k];
    }

    // Хвост
    for (; i < prices.size(); ++i) {
        s_num += prices[i] * vols[i];
        s_den += vols[i];
    }

    constexpr double eps = 1e-9;
    return (s_den > eps) ? s_num / s_den : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX-512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__) && defined(__AVX512DQ__)
inline double avx512(std::span<const double> prices, std::span<const double> vols) {
    __m512d num = _mm512_setzero_pd();
    __m512d den = _mm512_setzero_pd();
    size_t i = 0;
    for (; i + 8 <= prices.size(); i += 8) {
        __m512d p = _mm512_loadu_pd(&prices[i]);
        __m512d v = _mm512_loadu_pd(&vols[i]);
        num = _mm512_fmadd_pd(p, v, num);
        den = _mm512_add_pd(den, v);
    }
    double s_num = _mm512_reduce_add_pd(num);
    double s_den = _mm512_reduce_add_pd(den);
    for (; i < prices.size(); ++i) {
        s_num += prices[i] * vols[i];
        s_den += vols[i];
    }
    constexpr double eps = 1e-9;
    return (s_den > eps) ? s_num / s_den : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> prices, std::span<const double> vols) {
    __m256d num = _mm256_setzero_pd();
    __m256d den = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= prices.size(); i += 4) {
        __m256d p = _mm256_loadu_pd(&prices[i]);
        __m256d v = _mm256_loadu_pd(&vols[i]);
        num = _mm256_fmadd_pd(p, v, num);
        den = _mm256_add_pd(den, v);
    }
    num = _mm256_hadd_pd(num, num); num = _mm256_hadd_pd(num, num);
    den = _mm256_hadd_pd(den, den); den = _mm256_hadd_pd(den, den);
    double s_num = _mm_cvtsd_f64(_mm256_extractf128_pd(num, 0)) +
                   _mm_cvtsd_f64(_mm256_extractf128_pd(num, 1));
    double s_den = _mm_cvtsd_f64(_mm256_extractf128_pd(den, 0)) +
                   _mm_cvtsd_f64(_mm256_extractf128_pd(den, 1));
    for (; i < prices.size(); ++i) {
        s_num += prices[i] * vols[i];
        s_den += vols[i];
    }
    constexpr double eps = 1e-9;
    return (s_den > eps) ? s_num / s_den : 0.0;
}
#endif

} // namespace vwap