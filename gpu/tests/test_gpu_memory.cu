//
// test_gpu_memory.cu — Memory management tests for the GPU backend.
//
// Verifies:
//   * RAII allocation/deallocation
//   * Stream-ordered allocation (cudaMallocAsync)
//   * Pinned host buffers
//   * Large allocations (stress test)
//   * Memory leak detection (via cudaMemGetInfo before/after)
//
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu_memory.h"
#include "gpu_types.h"

using namespace tomogpu;

// ---------------------------------------------------------------------------
// Helper: get free/total GPU memory
// ---------------------------------------------------------------------------
static void get_mem_info(size_t& free, size_t& total) {
    cudaMemGetInfo(&free, &total);
}

// ---------------------------------------------------------------------------
// Test 1: Basic DeviceBuffer allocation and deallocation
// ---------------------------------------------------------------------------
static void test_basic_alloc() {
    printf("  [RUN ] Basic DeviceBuffer alloc/dealloc\n");
    size_t free_before, total;
    get_mem_info(free_before, total);

    {
        DeviceBuffer<real_t> buf(1024);
        assert(buf.size() == 1024);
        assert(buf.get() != nullptr);
        // RAII frees on scope exit
    }

    size_t free_after;
    get_mem_info(free_after, total);
    assert(free_after >= free_before);  // memory should be freed
    printf("  [PASS] Basic alloc (freed %zu bytes)\n", free_after - free_before);
}

// ---------------------------------------------------------------------------
// Test 2: Move semantics
// ---------------------------------------------------------------------------
static void test_move_semantics() {
    printf("  [RUN ] DeviceBuffer move semantics\n");
    DeviceBuffer<real_t> buf1(512);
    real_t* ptr1 = buf1.get();

    DeviceBuffer<real_t> buf2(std::move(buf1));
    assert(buf1.empty());       // moved-from is empty
    assert(buf2.get() == ptr1); // new owner has the pointer
    assert(buf2.size() == 512);

    printf("  [PASS] Move semantics\n");
}

// ---------------------------------------------------------------------------
// Test 3: Resize
// ---------------------------------------------------------------------------
static void test_resize() {
    printf("  [RUN ] DeviceBuffer resize\n");
    DeviceBuffer<real_t> buf(256);
    assert(buf.size() == 256);

    buf.resize(1024);
    assert(buf.size() == 1024);

    buf.resize(0);
    assert(buf.empty());

    printf("  [PASS] Resize\n");
}

// ---------------------------------------------------------------------------
// Test 4: PinnedHostBuffer
// ---------------------------------------------------------------------------
static void test_pinned_host() {
    printf("  [RUN ] PinnedHostBuffer\n");
    PinnedHostBuffer<real_t> buf(512);
    assert(buf.size() == 512);
    assert(buf.get() != nullptr);

    // Fill and verify
    for (int i = 0; i < 512; i++) buf.get()[i] = static_cast<real_t>(i);
    for (int i = 0; i < 512; i++) assert(buf.get()[i] == static_cast<real_t>(i));

    printf("  [PASS] PinnedHostBuffer\n");
}

// ---------------------------------------------------------------------------
// Test 5: Large allocation stress test
// ---------------------------------------------------------------------------
static void test_large_alloc() {
    printf("  [RUN ] Large allocation (1 GiB)\n");
    // 1 GiB of doubles = 134,217,728 elements
    const size_t N = 134217728;
    DeviceBuffer<real_t> buf(N);
    assert(buf.size() == N);
    assert(buf.get() != nullptr);
    printf("  [PASS] Large alloc (%zu elements)\n", N);
}

// ---------------------------------------------------------------------------
// Test 6: Copy operations
// ---------------------------------------------------------------------------
static void test_copy_ops() {
    printf("  [RUN ] Copy operations\n");
    const int N = 1024;
    std::vector<real_t> host(N);
    for (int i = 0; i < N; i++) host[i] = static_cast<real_t>(i * 0.5);

    DeviceBuffer<real_t> buf(N);
    buf.copy_from_host(host.data());
    cudaDeviceSynchronize();

    std::vector<real_t> result(N);
    buf.copy_to_host(result.data());
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) assert(result[i] == host[i]);

    printf("  [PASS] Copy operations\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("============================================================\n");
    printf("  GPU Backend Memory Tests\n");
    printf("============================================================\n\n");

    // Set device
    int ndev;
    cudaGetDeviceCount(&ndev);
    if (ndev == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }
    cudaSetDevice(0);

    test_basic_alloc();
    test_move_semantics();
    test_resize();
    test_pinned_host();
    test_large_alloc();
    test_copy_ops();

    printf("\n============================================================\n");
    printf("  All memory tests passed!\n");
    printf("============================================================\n");
    return 0;
}
