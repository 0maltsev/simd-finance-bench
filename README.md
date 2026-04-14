# simd-finance-bench

# 1. Очистка + создание директории
rm -rf build && mkdir build && cd build

# 2. Конфигурация (указываем компилятор явно, если стандартный старый)
cmake .. \
-DCMAKE_CXX_COMPILER=$(which clang++-18 || which clang++) \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_FLAGS="-O3 -march=native -fexperimental-library -fno-vectorize"

# 3. Сборка (используем все ядра)
cmake --build . -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# 4. Запуск бенчмарков
./simd_bench \
--benchmark_min_time=0.5s \
--benchmark_repetitions=3 \
--benchmark_enable_random_interleaving=false \
--benchmark_out_format=json \
--benchmark_out=results_vwap.json