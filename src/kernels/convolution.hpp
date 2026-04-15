#pragma once
#include <span>
#include <cmath>
#include <vector>
#include <experimental/simd>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace convolution {

// ─────────────────────────────────────────────────────────────
// 1. Scalar baseline (FIR filter with fixed kernel)
// ─────────────────────────────────────────────────────────────
inline double scalar(std::span<const double> input, std::span<const double> kernel) {
    const size_t n = input.size();
    const size_t k = kernel.size();
    if (n < k || k == 0) return 0.0;

    // Compute sum of convolved output (for benchmarking purposes)
    double sum = 0.0;
    for (size_t i = k - 1; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < k; ++j) {
            acc += input[i - j] * kernel[j];
        }
        sum += acc;
    }
    return sum / static_cast<double>(n - k + 1);
}

// ─────────────────────────────────────────────────────────────
// 2. std::simd (experimental) - P1928 + GCC 14 compatible
// ─────────────────────────────────────────────────────────────
inline double simd(std::span<const double> input, std::span<const double> kernel) {
    using V = std::experimental::simd<double>;
    const size_t n = input.size();
    const size_t k = kernel.size();
    if (n < k || k == 0) return 0.0;

    const size_t vec_len = V::size();
    double sum = 0.0;

    // Process output elements in vectorized fashion
    // Each output element is independent (no loop-carried dependency)
    size_t i = k - 1;
    for (; i + vec_len <= n; i += vec_len) {
        V acc{0.0};

        // Sliding window: load overlapping input segments
        for (size_t j = 0; j < k; ++j) {
            V k_val(kernel[j]);
            V input_window(&input[i - j - vec_len + 1],
                          std::experimental::element_aligned);
            acc += k_val * input_window;
        }

        // Manual reduction of accumulator vector
        alignas(64) double buf[8];
        acc.copy_to(buf, std::experimental::element_aligned);
        for (size_t l = 0; l < vec_len; ++l) {
            sum += buf[l];
        }
    }

    // Scalar tail
    for (; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < k; ++j) {
            acc += input[i - j] * kernel[j];
        }
        sum += acc;
    }

    return sum / static_cast<double>(n - k + 1);
}

// ─────────────────────────────────────────────────────────────
// 3. Intrinsics (AVX512 → AVX2 fallback)
// ─────────────────────────────────────────────────────────────
#if defined(__AVX512F__)
inline double avx512(std::span<const double> input, std::span<const double> kernel) {
    const size_t n = input.size();
    const size_t k = kernel.size();
    if (n < k || k == 0) return 0.0;

    __m512d sum_vec = _mm512_setzero_pd();
    size_t i = k - 1;

    for (; i + 8 <= n; i += 8) {
        __m512d acc = _mm512_setzero_pd();

        for (size_t j = 0; j < k; ++j) {
            __m512d k_val = _mm512_set1_pd(kernel[j]);
            __m512d input_window = _mm512_loadu_pd(&input[i - j - 7]);
            acc = _mm512_fmadd_pd(k_val, input_window, acc);
        }

        sum_vec = _mm512_add_pd(sum_vec, acc);
    }

    double sum = _mm512_reduce_add_pd(sum_vec);

    for (; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < k; ++j) {
            acc += input[i - j] * kernel[j];
        }
        sum += acc;
    }

    return sum / static_cast<double>(n - k + 1);
}
#elif defined(__AVX2__)
inline double avx2(std::span<const double> input, std::span<const double> kernel) {
    const size_t n = input.size();
    const size_t k = kernel.size();
    if (n < k || k == 0) return 0.0;

    __m256d sum_vec = _mm256_setzero_pd();
    size_t i = k - 1;

    for (; i + 4 <= n; i += 4) {
        __m256d acc = _mm256_setzero_pd();

        for (size_t j = 0; j < k; ++j) {
            __m256d k_val = _mm256_set1_pd(kernel[j]);
            __m256d input_window = _mm256_loadu_pd(&input[i - j - 3]);
            acc = _mm256_fmadd_pd(k_val, input_window, acc);
        }

        sum_vec = _mm256_add_pd(sum_vec, acc);
    }

    // Manual reduction for AVX2
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_pd(sum_vec, sum_vec);
    double sum = _mm_cvtsd_f64(_mm256_extractf128_pd(sum_vec, 0)) +
                 _mm_cvtsd_f64(_mm256_extractf128_pd(sum_vec, 1));

    for (; i < n; ++i) {
        double acc = 0.0;
        for (size_t j = 0; j < k; ++j) {
            acc += input[i - j] * kernel[j];
        }
        sum += acc;
    }

    return sum / static_cast<double>(n - k + 1);
}
#endif

} // namespace convolution