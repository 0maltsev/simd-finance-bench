#pragma once
#include <span>
#include <cmath>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace rsi {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (Wilder's smoothing)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> close, double period) {
    const size_t n = close.size();
    if (n < 2) return 50.0; // Neutral RSI

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;

    double avg_gain = 0.0;
    double avg_loss = 0.0;

    // First pass: initialize with simple averages
    for (size_t i = 1; i <= static_cast<size_t>(period) && i < n; ++i) {
        double change = close[i] - close[i-1];
        if (change > 0.0) avg_gain += change;
        else              avg_loss += std::abs(change);
    }
    avg_gain /= period;
    avg_loss /= period;

    // Second pass: Wilder's smoothing (IIR filter)
    for (size_t i = static_cast<size_t>(period) + 1; i < n; ++i) {
        double change = close[i] - close[i-1];
        double gain = (change > 0.0) ? change : 0.0;
        double loss = (change <= 0.0) ? std::abs(change) : 0.0;

        avg_gain = avg_gain * beta + gain * alpha;
        avg_loss = avg_loss * beta + loss * alpha;
    }

    // RSI calculation
    constexpr double eps = 1e-12;
    if (avg_loss < eps) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> close, double period) {
    using V = std::experimental::simd<double>;
    const size_t n = close.size();
    if (n < 2) return 50.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;
    const size_t vec_len = V::size();

    double avg_gain = 0.0;
    double avg_loss = 0.0;

    // ── Pass 1: Initialize (scalar, same as baseline) ──
    size_t i = 1;
    for (; i <= static_cast<size_t>(period) && i < n; ++i) {
        double change = close[i] - close[i-1];
        if (change > 0.0) avg_gain += change;
        else              avg_loss += std::abs(change);
    }
    avg_gain /= period;
    avg_loss /= period;

    // ── Pass 2: Wilder's smoothing with block-wise SIMD ──
    V v_beta(beta), v_alpha(alpha);
    V v_zero(0.0);

    for (; i + vec_len <= n; i += vec_len) {
        V c_curr(&close[i],   std::experimental::element_aligned);
        V c_prev(&close[i-1], std::experimental::element_aligned);

        V change = c_curr - c_prev;

        // ✅ Masked gain/loss extraction (P1928 where)
        V gain = change;
        std::experimental::where(change <= v_zero, gain) = v_zero;

        V loss = -change;
        std::experimental::where(change > v_zero, loss) = v_zero;

        // ✅ abs() via max(x, -x)
        loss = std::experimental::max(loss, -loss);

        // Sequential smoothing within block (dependency barrier)
        alignas(64) double gain_buf[8], loss_buf[8];
        gain.copy_to(gain_buf, std::experimental::element_aligned);
        loss.copy_to(loss_buf, std::experimental::element_aligned);

        for (size_t k = 0; k < vec_len; ++k) {
            avg_gain = avg_gain * beta + gain_buf[k] * alpha;
            avg_loss = avg_loss * beta + loss_buf[k] * alpha;
        }
    }

    // ── Scalar tail ──
    for (; i < n; ++i) {
        double change = close[i] - close[i-1];
        double gain = (change > 0.0) ? change : 0.0;
        double loss = (change <= 0.0) ? std::abs(change) : 0.0;
        avg_gain = avg_gain * beta + gain * alpha;
        avg_loss = avg_loss * beta + loss * alpha;
    }

    // ── Final RSI calculation (scalar division) ──
    constexpr double eps = 1e-12;
    if (avg_loss < eps) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> close, double period) {
    const size_t n = close.size();
    if (n < 2) return 50.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;

    double avg_gain = 0.0, avg_loss = 0.0;

    // Pass 1: Initialize
    size_t i = 1;
    for (; i <= static_cast<size_t>(period) && i < n; ++i) {
        double change = close[i] - close[i-1];
        if (change > 0.0) avg_gain += change;
        else              avg_loss += std::abs(change);
    }
    avg_gain /= period;
    avg_loss /= period;

    // Pass 2: Wilder's smoothing
    __m512d v_beta = _mm512_set1_pd(beta);
    __m512d v_alpha = _mm512_set1_pd(alpha);
    __m512d v_zero = _mm512_setzero_pd();

    for (; i + 8 <= n; i += 8) {
        __m512d c_curr = _mm512_loadu_pd(&close[i]);
        __m512d c_prev = _mm512_loadu_pd(&close[i-1]);
        __m512d change = _mm512_sub_pd(c_curr, c_prev);

        // Masked gain/loss
        __mmask8 pos_mask = _mm512_cmp_pd_mask(change, v_zero, _CMP_GT_OS);
        __m512d gain = _mm512_mask_blend_pd(pos_mask, v_zero, change);
        __m512d loss = _mm512_mask_blend_pd(pos_mask, change, v_zero);
        loss = _mm512_max_pd(loss, _mm512_sub_pd(v_zero, loss)); // abs

        alignas(64) double gain_buf[8], loss_buf[8];
        _mm512_storeu_pd(gain_buf, gain);
        _mm512_storeu_pd(loss_buf, loss);

        for (int k = 0; k < 8; ++k) {
            avg_gain = avg_gain * beta + gain_buf[k] * alpha;
            avg_loss = avg_loss * beta + loss_buf[k] * alpha;
        }
    }

    for (; i < n; ++i) {
        double change = close[i] - close[i-1];
        double gain = (change > 0.0) ? change : 0.0;
        double loss = (change <= 0.0) ? std::abs(change) : 0.0;
        avg_gain = avg_gain * beta + gain * alpha;
        avg_loss = avg_loss * beta + loss * alpha;
    }

    constexpr double eps = 1e-12;
    if (avg_loss < eps) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> close, double period) {
    const size_t n = close.size();
    if (n < 2) return 50.0;

    const double beta  = (period - 1.0) / period;
    const double alpha = 1.0 / period;

    double avg_gain = 0.0, avg_loss = 0.0;

    // Pass 1: Initialize
    size_t i = 1;
    for (; i <= static_cast<size_t>(period) && i < n; ++i) {
        double change = close[i] - close[i-1];
        if (change > 0.0) avg_gain += change;
        else              avg_loss += std::abs(change);
    }
    avg_gain /= period;
    avg_loss /= period;

    // Pass 2: Wilder's smoothing
    __m256d v_beta = _mm256_set1_pd(beta);
    __m256d v_alpha = _mm256_set1_pd(alpha);
    __m256d v_zero = _mm256_setzero_pd();

    for (; i + 4 <= n; i += 4) {
        __m256d c_curr = _mm256_loadu_pd(&close[i]);
        __m256d c_prev = _mm256_loadu_pd(&close[i-1]);
        __m256d change = _mm256_sub_pd(c_curr, c_prev);

        __m256d cmp = _mm256_cmp_pd(change, v_zero, _CMP_GT_OS);
        __m256d gain = _mm256_blendv_pd(v_zero, change, cmp);
        __m256d loss = _mm256_blendv_pd(change, v_zero, cmp);
        loss = _mm256_max_pd(loss, _mm256_sub_pd(v_zero, loss)); // abs

        alignas(32) double gain_buf[4], loss_buf[4];
        _mm256_storeu_pd(gain_buf, gain);
        _mm256_storeu_pd(loss_buf, loss);

        for (int k = 0; k < 4; ++k) {
            avg_gain = avg_gain * beta + gain_buf[k] * alpha;
            avg_loss = avg_loss * beta + loss_buf[k] * alpha;
        }
    }

    for (; i < n; ++i) {
        double change = close[i] - close[i-1];
        double gain = (change > 0.0) ? change : 0.0;
        double loss = (change <= 0.0) ? std::abs(change) : 0.0;
        avg_gain = avg_gain * beta + gain * alpha;
        avg_loss = avg_loss * beta + loss * alpha;
    }

    constexpr double eps = 1e-12;
    if (avg_loss < eps) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - (100.0 / (1.0 + rs));
}
#endif

} // namespace rsi