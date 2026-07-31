# CMakeLists.txt for the clean-slate GPU backend (gpu/)
#
# This builds a standalone test/benchmark executable for the new GPU
# backend, independent of the legacy cuda/ code.  The backend targets
# NVIDIA Blackwell (GB200, compute capability 10.0) but also builds for
# older architectures (sm_70+) for development and CI.

# Guard: only include when CUDA is available and USE_CUDA_V2 is requested
if(NOT USE_CUDA_V2)
    return()
endif()

enable_language(CUDA)

# Require CUDA 11.8+ (for cudaMallocAsync, stream capture, C++17 device code)
find_package(CUDAToolkit 11.8 REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Architecture flags: target Blackwell (sm_100) plus common dev architectures.
# NOTE: CUDA 13.x dropped sm_70 (Volta) support. Adjust this list based on
# the installed CUDA toolkit version. Override from the command line with
# -DCMAKE_CUDA_ARCHITECTURES=<list> (e.g. "121" for GB10 / DGX Spark).
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 90 100)
endif()

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math --expt-relaxed-constexpr")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -lineinfo")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# Collect GPU backend sources
file(GLOB GPU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)
file(GLOB GPU_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# Build the GPU backend as a static library
add_library(tomogpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})

target_include_directories(tomogpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tomogpu PUBLIC
    CUDA::cudart
)

# Enable language extensions for CUDA
set_target_properties(tomogpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# --- Tests ---
enable_testing()

# Standalone correctness test (does not require MPI)
add_executable(test_gpu_correctness
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_correctness.cu
)
target_link_libraries(test_gpu_correctness PRIVATE tomogpu CUDA::cudart)
add_test(NAME gpu_correctness COMMAND test_gpu_correctness)

# Memory management test
add_executable(test_gpu_memory
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_memory.cu
)
target_link_libraries(test_gpu_memory PRIVATE tomogpu CUDA::cudart)
add_test(NAME gpu_memory COMMAND test_gpu_memory)

# --- Benchmarks ---
add_executable(bench_gpu_sweep
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_gpu_sweep.cu
)
target_link_libraries(bench_gpu_sweep PRIVATE tomogpu CUDA::cudart)

# CPU vs GPU comprehensive benchmark
add_executable(bench_cpu_vs_gpu
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_cpu_vs_gpu.cu
)
target_link_libraries(bench_cpu_vs_gpu PRIVATE tomogpu CUDA::cudart)

# Print configuration
message(STATUS "=== GPU Backend (clean-slate v2) ===")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  Sources: ${GPU_SOURCES}")
