#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace ewma {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> data, double alpha) {
    double s = data[0];
    const double alpha1 = 1.0 - alpha;
    for (size_t i = 1; i < data.size(); ++i) {
        s = alpha * data[i] + alpha1 * s;
    }
    return s;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental)
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> data, double alpha) {
    using V = std::experimental::simd<double>;
    const double alpha1 = 1.0 - alpha;
    V v_alpha(alpha);
    V v_alpha1(alpha1);

    double s = data[0];
    size_t i = 1;
    const size_t vec_len = V::size();

    for (; i + vec_len <= data.size(); i += vec_len) {
        V x(&data[i], std::experimental::element_aligned);
        V v_s(s); // Broadcast state to all lanes

        // Core EWMA step
        V new_s = v_alpha * x + v_alpha1 * v_s;

        // Mask: skip stale/invalid ticks
        auto valid = x > 0.0;

        std::experimental::where(valid, v_s) = new_s;

        // Propagate last lane to scalar state
        s = v_s[vec_len - 1];
    }

    // Scalar tail
    const double alpha1_scalar = 1.0 - alpha;
    for (; i < data.size(); ++i) {
        if (data[i] > 0.0) s = alpha * data[i] + alpha1_scalar * s;
    }
    return s;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX2 / AVX-512)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> data, double alpha) {
    const double alpha1 = 1.0 - alpha;
    __m512d v_alpha = _mm512_set1_pd(alpha);
    __m512d v_alpha1 = _mm512_set1_pd(alpha1);
    double s = data[0];
    size_t i = 1;
    for (; i + 8 <= data.size(); i += 8) {
        __m512d x = _mm512_loadu_pd(&data[i]);
        __m512d v_s = _mm512_set1_pd(s);
        __m512d new_s = _mm512_fmadd_pd(v_alpha, x, _mm512_mul_pd(v_alpha1, v_s));
        __mmask8 valid = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_GT_OS);
        __m512d blended = _mm512_mask_blend_pd(valid, v_s, new_s);
        alignas(64) double buf[8];
        _mm512_storeu_pd(buf, blended);
        s = buf[7];
    }
    const double a1 = 1.0 - alpha;
    for (; i < data.size(); ++i) { if (data[i] > 0.0) s = alpha * data[i] + a1 * s; }
    return s;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> data, double alpha) {
    const double alpha1 = 1.0 - alpha;
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_alpha1 = _mm256_set1_pd(alpha1);
    double s = data[0];
    size_t i = 1;
    for (; i + 4 <= data.size(); i += 4) {
        __m256d x = _mm256_loadu_pd(&data[i]);
        __m256d v_s = _mm256_set1_pd(s);
        __m256d new_s = _mm256_fmadd_pd(v_alpha, x, _mm256_mul_pd(v_alpha1, v_s));
        __m256d cmp = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GT_OS);
        __m256d blended = _mm256_blendv_pd(v_s, new_s, cmp);
        alignas(32) double buf[4];
        _mm256_storeu_pd(buf, blended);
        s = buf[3];
    }
    const double a1 = 1.0 - alpha;
    for (; i < data.size(); ++i) { if (data[i] > 0.0) s = alpha * data[i] + a1 * s; }
    return s;
}
#endif

} // namespace ewma