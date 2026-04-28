# SIMD Finance Bench

Benchmark suite for quantitative finance kernels comparing `scalar`, `std::simd` (C++23 `<experimental/simd>`), and hand-written AVX intrinsics on modern x86 CPUs.

This repository accompanies the paper **"Bridging Abstraction and Performance: C++23 std::simd Against AVX-512 Intrinsics in High-Frequency Trading"** by **Oleg Maltsev**.

## Motivation

High-frequency trading (HFT) systems are latency- and throughput-critical. Core quantitative kernels (pricing, signal extraction, risk features, microstructure metrics) run over large vectors and sliding windows where SIMD efficiency directly affects:

- per-tick processing latency,
- strategy capacity under bursty market load,
- hardware utilization and cost/performance.

`std::simd` promises portable SIMD abstraction in standard C++, but HFT workloads often rely on architecture-specific intrinsics to squeeze out tail latency. This project quantifies that tradeoff.

## Research Goal

The benchmark answers one question: **how close can C++23 `std::simd` get to optimized AVX implementations on finance kernels?**

We compare three implementations per kernel:

1. `scalar` baseline
2. `std::simd` (`<experimental/simd>`)
3. AVX intrinsics (`AVX-512` primary, `AVX2` fallback where applicable)

The target is gap analysis between high-level SIMD abstraction and low-level ISA control.

## Key Findings

- **Reduction-heavy kernels** (e.g., VWAP, Beta, Correlation): near parity, typically ~`1-3%` gap.
- **Masked/conditional kernels** (e.g., Sortino, Book Imbalance): larger penalties for `std::simd`.
- **Division-heavy kernels** (Kyle's Lambda): mixed behavior, sensitive to compiler lowering and division throughput.
- **Sliding-window/IIR kernels** (EWMA, RSI, ATR): largest gaps, up to ~`2.85x` in unfavorable cases.
- Two dominant causes:
  - mask lowering differences (`blend` sequences vs efficient `k`-register usage),
  - lack of efficient horizontal reduction primitives in current `std::simd` workflows.

## CPU and Toolchain Requirements

- x86-64 CPU with at least `AVX2` support
- `AVX-512F`/`AVX-512DQ` recommended for full paper-equivalent runs
- C++23 compiler:
  - **Primary:** GCC 14+
  - **Secondary:** Clang (partially supported for this setup)
- CMake `>= 3.20`
- Google Benchmark
- LLVM tools (for `llvm-mca` static pipeline analysis)

Check CPU flags:

```bash
lscpu | rg "avx2|avx512"
```

## Project Structure

Logical layout used in the paper:

- `kernels/` - finance kernels by computational pattern
- `benchmarks/` - Google Benchmark harness
- `simd/` - `std::simd` implementations
- `intrinsics/` - AVX2/AVX-512 implementations

Current repository layout:

- `src/kernels/` - contains scalar + `std::simd` + intrinsic variants per kernel
- `src/bench/main.cpp` - benchmark registration and dataset generation
- `results/` - benchmark JSON outputs and MCA text outputs
- `paper/` - figures and tables used in the manuscript

## Implemented Kernels

### 1) Reduction-heavy

- `vwap`
- `beta`
- `correlation`
- `sharpe_ratio`
- `alpha`

### 2) Masked / Conditional

- `sortino_ratio`
- `book_imbalance`
- `max_drawdown`
- `var`
- `cvar`

### 3) Division-heavy

- `kyle_lambda`
- `calmar_ratio`

### 4) Sliding Window / IIR

- `ewma`
- `rsi`
- `atr`
- `convolution` (windowed filter-style workload)

## Methodology

### Benchmarking (Google Benchmark)

- Deterministic synthetic inputs (fixed RNG seed).
- Multiple problem sizes (`2^16`, `2^20`, `2^22`).
- Per-kernel comparison across scalar / `std::simd` / AVX.
- Throughput and timing collected with Google Benchmark runners.

### Static Analysis (`llvm-mca`)

- MCA is used to inspect generated instruction streams for:
  - uOps pressure,
  - port utilization,
  - dependency chains and bottlenecks.
- This complements runtime metrics and explains observed gaps (especially for mask-heavy and reduction-heavy paths).

## Build Instructions

### 1) Configure

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 2) Build

```bash
cmake --build build -j
```

The project compiles with high-performance flags in `CMakeLists.txt`:

- `-O3`
- `-march=native`
- plus compiler-specific flags (e.g., Clang `-fexperimental-library`)

If you need explicit AVX target selection, you can additionally pass:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

## Usage

Run full benchmark suite:

```bash
./build/simd_bench
```

Run with reproducible benchmark settings and export JSON:

```bash
./build/simd_bench \
  --benchmark_min_time=0.5s \
  --benchmark_repetitions=3 \
  --benchmark_out_format=json \
  --benchmark_out=results/main_results.json
```

Filter specific kernels:

```bash
./build/simd_bench --benchmark_filter="VWAP|EWMA|KYLE"
```

## Example Output

```text
---------------------------------------------------------------------------
Benchmark                              Time             CPU   Iterations
---------------------------------------------------------------------------
BM_VWAP_Scalar/1048576             162000 ns       161500 ns         4300
BM_VWAP_STD_SIMD/1048576            26500 ns        26400 ns        26000
BM_VWAP_AVX512/1048576              25800 ns        25700 ns        27200

BM_EWMA_Scalar/1048576             311000 ns       310200 ns         2200
BM_EWMA_STD_SIMD/1048576            61200 ns        60900 ns        11500
BM_EWMA_AVX512/1048576              33600 ns        33400 ns        20900
```

## Performance Summary

| Kernel Class | Typical `std::simd` vs AVX-512 | Main Reason |
|---|---:|---|
| Reduction-heavy | `~1.01x - 1.03x` | Good vectorization, moderate reduction overhead |
| Masked / conditional | `~1.2x - 2.0x+` | Mask lowering quality (`blend` vs mask registers) |
| Division-heavy | `~1.1x - 1.6x` | Division latency + compiler scheduling |
| Sliding window / IIR | up to `~2.85x` | Dependency chains + non-trivial recurrence patterns |

Interpretation: `std::simd` is often close for regular reductions, but architecture-tuned intrinsics still dominate irregular mask-heavy and recurrent kernels.

## Code Examples

### `std::simd` kernel (VWAP core loop)

```cpp
using V = std::experimental::simd<double>;
V num{0.0}, den{0.0};
for (size_t i = 0; i + V::size() <= n; i += V::size()) {
    V p(&prices[i], std::experimental::element_aligned);
    V v(&vols[i], std::experimental::element_aligned);
    num = num + (p * v);
    den = den + v;
}
```

### AVX-512 intrinsic kernel (VWAP core loop)

```cpp
__m512d num = _mm512_setzero_pd();
__m512d den = _mm512_setzero_pd();
for (size_t i = 0; i + 8 <= n; i += 8) {
    __m512d p = _mm512_loadu_pd(&prices[i]);
    __m512d v = _mm512_loadu_pd(&vols[i]);
    num = _mm512_fmadd_pd(p, v, num);
    den = _mm512_add_pd(den, v);
}
double s_num = _mm512_reduce_add_pd(num);
double s_den = _mm512_reduce_add_pd(den);
```

### Short difference

- `std::simd`: portable and expressive, but codegen quality depends heavily on compiler lowering.
- AVX intrinsics: explicit ISA control and often better peak performance, but less portable and harder to maintain.

## Limitations

- Current `std::simd` implementation is from `<experimental/simd>`, not a fully stabilized standard library API.
- Results are hardware- and compiler-dependent; absolute timings do not generalize across all CPUs.
- Dataset is synthetic; real market microstructure feeds may expose additional branch and cache effects.
- Clang path is only partially characterized compared to GCC 14.

## Future Work

- Evaluate C++26-era SIMD facilities as implementations mature.
- Investigate improved mask lowering and horizontal reduction support in standard SIMD backends.
- Add perf-counter integration (`perf`, TopDown, PEBS where available).
- Extend kernels with more microstructure and options-pricing workloads.
- Add cross-platform CI matrix for GCC/Clang and AVX2/AVX-512 targets.

## References

- Maltsev, O. *Bridging Abstraction and Performance: C++23 std::simd Against AVX-512 Intrinsics in High-Frequency Trading*.
- C++ Parallelism TS / `std::experimental::simd` references.
- [Google Benchmark](https://github.com/google/benchmark)
- [LLVM-MCA Documentation](https://llvm.org/docs/CommandGuide/llvm-mca.html)