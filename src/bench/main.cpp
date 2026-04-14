#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include <span>
#include "kernels/vwap.hpp"

static void PrepareData(std::vector<double>& prices, std::vector<double>& vols, size_t n) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(100.0, 200.0);
    prices.resize(n);
    vols.resize(n);
    for (size_t i = 0; i < n; ++i) {
        prices[i] = dist(rng);
        vols[i] = std::abs(dist(rng) * 10.0);
    }
}

static void BM_VWAP_Scalar(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::scalar(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

static void BM_VWAP_STD_SIMD(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::simd(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}

#if defined(__AVX512F__) && defined(__AVX512DQ__)
static void BM_VWAP_AVX512(benchmark::State& state) {
    std::vector<double> prices, vols;
    PrepareData(prices, vols, state.range(0));
    for (auto _ : state) {
        auto res = vwap::avx512(prices, vols);
        benchmark::DoNotOptimize(res);
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(double) * 2);
}
#endif

BENCHMARK(BM_VWAP_Scalar)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
BENCHMARK(BM_VWAP_STD_SIMD)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#if defined(__AVX512F__) && defined(__AVX512DQ__)
BENCHMARK(BM_VWAP_AVX512)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22);
#endif

BENCHMARK_MAIN();