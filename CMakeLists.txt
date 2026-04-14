cmake_minimum_required(VERSION 3.20)
project(simd_finance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(benchmark REQUIRED)

add_executable(simd_bench src/bench/main.cpp)
target_include_directories(simd_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(simd_bench PRIVATE benchmark::benchmark)

# Флаги для GCC 14 + experimental SIMD
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC не требует -fexperimental-library, TS v2 включается автоматически через -std=c++23
    target_compile_options(simd_bench PRIVATE
            -O3 -fno-semantic-interposition -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(simd_bench PRIVATE
            -O3 -march=native -fexperimental-library -fno-vectorize)
endif()