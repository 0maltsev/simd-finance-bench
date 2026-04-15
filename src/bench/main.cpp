#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <span>
#include "kernels/vwap.hpp"
#include "kernels/ewma.hpp"
#include "kernels/book_imbalance.hpp"

// ─────────────────────────────────────────────────────────────
// Data Generation Helpers (deterministic, seed=42)
// ─────────────────────────────────────────────────────────────
static std::vector<double> GenerateData(size_t n, double min = 100.0, double max = 200.0) {
    std::vector<double> data(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(min, max);
    for (auto& x : data) x = dist(rng);
    return data;
}

static void PrepareVWAPData(std::vector<double>& prices, std::vector<double>& vols, size_t n) {
    prices = GenerateData(n, 100.0, 200.0);
    vols   = GenerateData(n, 1.0, 100.0);
}

// ─────────────────────────────────────────────────────────────
// VWAP Benchmarks
// ─────────────────────────────────────────────────────────────
static void BM_VWAP_Scalar(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareVWAPData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::scalar(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

static void BM_VWAP_STD_SIMD(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareVWAPData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::simd(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

#if defined(__AVX512F__)
static void BM_VWAP_AVX512(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareVWAPData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::avx512(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}
#elif defined(__AVX2__)
static void BM_VWAP_AVX2(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareVWAPData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::avx2(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}
#endif

// ─────────────────────────────────────────────────────────────
// EWMA Benchmarks
// ─────────────────────────────────────────────────────────────
static void BM_EWMA_Scalar(benchmark::State& state) {
    std::vector<double> data = GenerateData(state.range(0));
    double alpha = 0.1;
    for (auto _ : state) {
        auto res = ewma::scalar(data, alpha);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}

static void BM_EWMA_STD_SIMD(benchmark::State& state) {
    std::vector<double> data = GenerateData(state.range(0));
    double alpha = 0.1;
    for (auto _ : state) {
        auto res = ewma::simd(data, alpha);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}

#if defined(__AVX512F__)
static void BM_EWMA_AVX512(benchmark::State& state) {
    std::vector<double> data = GenerateData(state.range(0));
    double alpha = 0.1;
    for (auto _ : state) {
        auto res = ewma::avx512(data, alpha);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}
#elif defined(__AVX2__)
static void BM_EWMA_AVX2(benchmark::State& state) {
    std::vector<double> data = GenerateData(state.range(0));
    double alpha = 0.1;
    for (auto _ : state) {
        auto res = ewma::avx2(data, alpha);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double));
}
#endif

// ─────────────────────────────────────────────────────────────
// Book Imbalance Benchmarks
// ─────────────────────────────────────────────────────────────
static void BM_BOOK_IMB_Scalar(benchmark::State& state) {
    std::vector<double> bid = GenerateData(state.range(0), 1000.0, 5000.0);
    std::vector<double> ask = GenerateData(state.range(0), 1000.0, 5000.0);
    double threshold = 500.0;
    for (auto _ : state) {
        auto res = book_imbalance::scalar(bid, ask, threshold);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

static void BM_BOOK_IMB_STD_SIMD(benchmark::State& state) {
    std::vector<double> bid = GenerateData(state.range(0), 1000.0, 5000.0);
    std::vector<double> ask = GenerateData(state.range(0), 1000.0, 5000.0);
    double threshold = 500.0;
    for (auto _ : state) {
        auto res = book_imbalance::simd(bid, ask, threshold);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

#if defined(__AVX512F__)
static void BM_BOOK_IMB_AVX512(benchmark::State& state) {
    std::vector<double> bid = GenerateData(state.range(0), 1000.0, 5000.0);
    std::vector<double> ask = GenerateData(state.range(0), 1000.0, 5000.0);
    double threshold = 500.0;
    for (auto _ : state) {
        auto res = book_imbalance::avx512(bid, ask, threshold);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}
#elif defined(__AVX2__)
static void BM_BOOK_IMB_AVX2(benchmark::State& state) {
    std::vector<double> bid = GenerateData(state.range(0), 1000.0, 5000.0);
    std::vector<double> ask = GenerateData(state.range(0), 1000.0, 5000.0);
    double threshold = 500.0;
    for (auto _ : state) {
        auto res = book_imbalance::avx2(bid, ask, threshold);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}
#endif

// ─────────────────────────────────────────────────────────────
// Benchmark Registration
// ─────────────────────────────────────────────────────────────
BENCHMARK(BM_VWAP_Scalar)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
BENCHMARK(BM_VWAP_STD_SIMD)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#if defined(__AVX512F__)
BENCHMARK(BM_VWAP_AVX512)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#elif defined(__AVX2__)
BENCHMARK(BM_VWAP_AVX2)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#endif

BENCHMARK(BM_EWMA_Scalar)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
BENCHMARK(BM_EWMA_STD_SIMD)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#if defined(__AVX512F__)
BENCHMARK(BM_EWMA_AVX512)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#elif defined(__AVX2__)
BENCHMARK(BM_EWMA_AVX2)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#endif

BENCHMARK(BM_BOOK_IMB_Scalar)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
BENCHMARK(BM_BOOK_IMB_STD_SIMD)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#if defined(__AVX512F__)
BENCHMARK(BM_BOOK_IMB_AVX512)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#elif defined(__AVX2__)
BENCHMARK(BM_BOOK_IMB_AVX2)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#endif

BENCHMARK_MAIN();