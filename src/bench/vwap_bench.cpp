#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "kernels/vwap.hpp"

static void BM_VWAP(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<double> prices(n), vols(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(100.0, 200.0);
    for (size_t i = 0; i < n; ++i) {
        prices[i] = dist(rng);
        vols[i] = std::abs(dist(rng) * 10.0);
    }

    for (auto _ : state) {
        auto res = vwap::scalar({prices.data(), n}, {vols.data(), n});
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(n * state.iterations());
}

static void BM_VWAP_SIMD(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<double> prices(n), vols(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(100.0, 200.0);
    for (size_t i = 0; i < n; ++i) {
        prices[i] = dist(rng);
        vols[i] = std::abs(dist(rng) * 10.0);
    }

    for (auto _ : state) {
        auto res = vwap::simd({prices.data(), n}, {vols.data(), n});
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(n * state.iterations());
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
static void BM_VWAP_AVX512(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<double> prices(n), vols(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(100.0, 200.0);
    for (size_t i = 0; i < n; ++i) {
        prices[i] = dist(rng);
        vols[i] = std::abs(dist(rng) * 10.0);
    }

    for (auto _ : state) {
        auto res = vwap::avx512({prices.data(), n}, {vols.data(), n});
        benchmark::DoNotOptimize(res);
    }
    state.SetItemsProcessed(n * state.iterations());
}
#endif

BENCHMARK(BM_VWAP)->Arg(1 << 14)->Arg(1 << 18)->Arg(1 << 22);
BENCHMARK(BM_VWAP_SIMD)->Arg(1 << 14)->Arg(1 << 18)->Arg(1 << 22);
#if defined(__AVX512F__) && defined(__AVX512DQ__)
BENCHMARK(BM_VWAP_AVX512)->Arg(1 << 14)->Arg(1 << 18)->Arg(1 << 22);
#endif

BENCHMARK_MAIN();