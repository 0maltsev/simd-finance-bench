#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace book_imbalance {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> bid_vol, std::span<const double> ask_vol, double threshold) {
    double sum_imb = 0.0;
    double count = 0.0;
    for (size_t i = 0; i < bid_vol.size(); ++i) {
        double den = bid_vol[i] + ask_vol[i];
        if (den > 1e-9) {
            double diff = bid_vol[i] - ask_vol[i];
            if (std::abs(diff) < threshold) {
                sum_imb += diff / den;
                count += 1.0;
            }
        }
    }
    return count > 0.0 ? sum_imb / count : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> bid_vol, std::span<const double> ask_vol, double threshold) {
    using V = std::experimental::simd<double>;
    using M = typename V::mask_type;

    V sum{0.0};
    V count{0.0};
    size_t i = 0;
    const size_t vec_len = V::size();
    V v_thresh(threshold);

    for (; i + vec_len <= bid_vol.size(); i += vec_len) {
        V b(&bid_vol[i], std::experimental::element_aligned);
        V a(&ask_vol[i], std::experimental::element_aligned);

        // 1. Mask creation (branch-free predicates)
        M valid = (b + a) > V{1e-9};

        // ✅ Fix #1: abs() via element-wise max (ternary ?: doesn't work with masks)
        V diff = b - a;
        V abs_diff = std::experimental::max(diff, -diff);
        M in_range = abs_diff < v_thresh;

        M cond = valid & in_range;

        // 2. Core computation (executed for all lanes)
        V den = b + a;
        V imb = diff / den;

        // 3. P1928 masked accumulation (proxy assignment)
        std::experimental::where(cond, sum) = sum + imb;
        std::experimental::where(cond, count) = count + V{1.0};
    }

    // ✅ Fix #2: Manual reduction via copy_to (hsum not available in GCC 14 libstdc++)
    alignas(64) double buf_sum[8];  // 8 >= max simd_size for double on x86
    alignas(64) double buf_cnt[8];
    sum.copy_to(buf_sum, std::experimental::element_aligned);
    count.copy_to(buf_cnt, std::experimental::element_aligned);

    double s_sum = 0.0, s_count = 0.0;
    for (size_t k = 0; k < vec_len; ++k) {
        s_sum += buf_sum[k];
        s_count += buf_cnt[k];
    }

    // Scalar tail
    const double thresh = threshold;
    for (; i < bid_vol.size(); ++i) {
        double den = bid_vol[i] + ask_vol[i];
        if (den > 1e-9) {
            double diff = bid_vol[i] - ask_vol[i];
            if (std::abs(diff) < thresh) {
                s_sum += diff / den;
                s_count += 1.0;
            }
        }
    }
    return s_count > 0.0 ? s_sum / s_count : 0.0;
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> bid_vol, std::span<const double> ask_vol, double threshold) {
    __m512d sum = _mm512_setzero_pd();
    __m512d count = _mm512_setzero_pd();
    __m512d v_thresh = _mm512_set1_pd(threshold);
    size_t i = 0;
    for (; i + 8 <= bid_vol.size(); i += 8) {
        __m512d b = _mm512_loadu_pd(&bid_vol[i]);
        __m512d a = _mm512_loadu_pd(&ask_vol[i]);

        __mmask8 valid = _mm512_cmp_pd_mask(b + a, _mm512_set1_pd(1e-9), _CMP_GT_OS);
        __m512d diff = b - a;
        __m512d abs_diff = _mm512_max_pd(diff, _mm512_sub_pd(_mm512_setzero_pd(), diff));
        __mmask8 in_range = _mm512_cmp_pd_mask(abs_diff, v_thresh, _CMP_LT_OS);
        __mmask8 cond = valid & in_range;

        __m512d den = b + a;
        __m512d imb = diff / den;

        sum = _mm512_mask_add_pd(sum, cond, sum, imb);
        count = _mm512_mask_add_pd(count, cond, count, _mm512_set1_pd(1.0));
    }
    double s_sum = _mm512_reduce_add_pd(sum);
    double s_count = _mm512_reduce_add_pd(count);

    const double thresh = threshold;
    for (; i < bid_vol.size(); ++i) {
        double den = bid_vol[i] + ask_vol[i];
        if (den > 1e-9 && std::abs(bid_vol[i] - ask_vol[i]) < thresh) {
            s_sum += (bid_vol[i] - ask_vol[i]) / den;
            s_count += 1.0;
        }
    }
    return s_count > 0.0 ? s_sum / s_count : 0.0;
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> bid_vol, std::span<const double> ask_vol, double threshold) {
    __m256d sum = _mm256_setzero_pd();
    __m256d count = _mm256_setzero_pd();
    __m256d v_thresh = _mm256_set1_pd(threshold);
    __m256d zero = _mm256_setzero_pd();
    size_t i = 0;
    for (; i + 4 <= bid_vol.size(); i += 4) {
        __m256d b = _mm256_loadu_pd(&bid_vol[i]);
        __m256d a = _mm256_loadu_pd(&ask_vol[i]);

        __m256d den = _mm256_add_pd(b, a);
        __m256d valid = _mm256_cmp_pd(den, _mm256_set1_pd(1e-9), _CMP_GT_OS);

        __m256d diff = _mm256_sub_pd(b, a);
        __m256d abs_diff = _mm256_max_pd(diff, _mm256_sub_pd(zero, diff));
        __m256d in_range = _mm256_cmp_pd(abs_diff, v_thresh, _CMP_LT_OS);
        __m256d cond = _mm256_and_pd(valid, in_range);

        __m256d imb = _mm256_div_pd(diff, den);
        __m256d add_sum = _mm256_add_pd(sum, imb);
        __m256d add_cnt = _mm256_add_pd(count, _mm256_set1_pd(1.0));

        sum = _mm256_blendv_pd(sum, add_sum, cond);
        count = _mm256_blendv_pd(count, add_cnt, cond);
    }

    // Manual reduction for AVX2
    sum = _mm256_hadd_pd(sum, sum); sum = _mm256_hadd_pd(sum, sum);
    double s_sum = _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(sum, 1));

    count = _mm256_hadd_pd(count, count); count = _mm256_hadd_pd(count, count);
    double s_count = _mm_cvtsd_f64(_mm256_extractf128_pd(count, 0)) + _mm_cvtsd_f64(_mm256_extractf128_pd(count, 1));

    const double thresh = threshold;
    for (; i < bid_vol.size(); ++i) {
        double den = bid_vol[i] + ask_vol[i];
        if (den > 1e-9 && std::abs(bid_vol[i] - ask_vol[i]) < thresh) {
            s_sum += (bid_vol[i] - ask_vol[i]) / den;
            s_count += 1.0;
        }
    }
    return s_count > 0.0 ? s_sum / s_count : 0.0;
}
#endif

} // namespace book_imbalance