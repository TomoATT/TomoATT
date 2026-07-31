//
// bench_gpu_sweep.cu — Benchmark suite for the GPU sweep backend.
//
// Measures:
//   1. Sweep execution time for various grid sizes
//   2. Memory footprint (total GPU memory used)
//   3. Effective bandwidth (GB/s)
//   4. Kernel launch overhead
//   5. CUDA Graph replay overhead vs. direct launches
//
// Run: ./bench_gpu_sweep [--sizes 16,32,64,128] [--warmup 5] [--iters 20]
//
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu_backend.h"
#include "gpu_grid.h"
#include "gpu_kernels.cuh"
#include "gpu_memory.h"
#include "gpu_types.h"

using namespace tomogpu;
using Clock = std::chrono::high_resolution_clock;
using ms_t  = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------
struct BenchConfig {
    std::vector<int> sizes = {16, 32, 64, 128};
    int warmup  = 3;
    int iters   = 10;
    int streams = 1;
};

// ---------------------------------------------------------------------------
// Parse command-line arguments
// ---------------------------------------------------------------------------
BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) {
            cfg.sizes.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                cfg.sizes.push_back(std::stoi(s.substr(0, pos)));
                s = s.substr(pos + 1);
            }
            cfg.sizes.push_back(std::stoi(s));
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iters = std::stoi(argv[++i]);
        } else if (arg == "--streams" && i + 1 < argc) {
            cfg.streams = std::stoi(argv[++i]);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Get GPU memory usage
// ---------------------------------------------------------------------------
static void print_mem_usage(const char* label) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    double used_mb = static_cast<double>(total - free) / (1024.0 * 1024.0);
    printf("    %-20s : %.1f MiB\n", label, used_mb);
}

// ---------------------------------------------------------------------------
// Benchmark: single sweep execution time
// ---------------------------------------------------------------------------
struct BenchResult {
    int    grid_size;
    double sweep_time_ms;      // time for one complete sweep
    double graph_replay_ms;    // CUDA Graph replay time
    double total_mem_mb;       // total GPU memory used
    double bandwidth_gbs;      // effective memory bandwidth
    idx_t  total_nodes;
};

BenchResult benchmark_sweep(int n, int warmup, int iters) {
    GridDims dims{n, n, n};
    const idx_t N = dims.total();

    // Create backend
    GpuSweepBackend backend(dims, 1.0, 1.0, 1.0, StencilOrder::First);

    // Initialize with simple constant fields
    std::vector<real_t> h_T0v(N, 1.0), h_T0r(N, 0.0), h_T0t(N, 0.0), h_T0p(N, 0.0);
    std::vector<real_t> h_fac_a(N, 1.0), h_fac_b(N, 1.0), h_fac_c(N, 1.0), h_fac_f(N, 0.0);
    std::vector<real_t> h_fun(N, 1.0), h_tau(N, 1.0);
    std::vector<uint8_t> h_changed(N, 1);

    backend.initialize_from_host(
        h_T0v.data(), h_T0r.data(), h_T0t.data(), h_T0p.data(),
        h_fac_a.data(), h_fac_b.data(), h_fac_c.data(), h_fac_f.data(),
        h_fun.data(), h_changed.data(), h_tau.data());

    // Record memory usage before benchmark
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        backend.run_sweep();
    }
    cudaDeviceSynchronize();

    // Benchmark sweep
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) {
        backend.run_sweep();
    }
    cudaDeviceSynchronize();
    auto t1 = Clock::now();

    double sweep_ms = ms_t(t1 - t0).count() / iters;

    // Benchmark CUDA Graph replay
    auto t2 = Clock::now();
    for (int i = 0; i < iters; i++) {
        backend.run_sweep();
    }
    cudaDeviceSynchronize();
    auto t3 = Clock::now();

    double graph_ms = ms_t(t3 - t2).count() / iters;

    // Compute memory usage. NOTE: the CUDA caching allocator can return memory to its
    // free pool after a warmup run (free_after > free_before), so an unsigned
    // free_before - free_after wraps to ~2^64. Use signed arithmetic and clamp.
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);
    double mem_mb = std::max(0.0,
        static_cast<double>(free_before) / (1024.0 * 1024.0)
      - static_cast<double>(free_after) / (1024.0 * 1024.0));

    // Compute effective bandwidth
    // Each sweep reads: tau, T0v, T0r, T0t, T0p, fac_a, fac_b, fac_c, fac_f, fun, is_changed (11 arrays)
    // Each sweep writes: tau (1 array)
    // Total bytes per sweep = 12 * N * sizeof(real_t)
    double bytes = 12.0 * static_cast<double>(N) * sizeof(real_t);
    double bw_gbs = bytes / (sweep_ms * 1e6);  // GB/s = bytes / (ms * 1e6)

    return {n, sweep_ms, graph_ms, mem_mb, bw_gbs, N};
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);

    printf("============================================================\n");
    printf("  GPU Backend Benchmark Suite\n");
    printf("============================================================\n\n");

    // Set device
    int ndev;
    cudaGetDeviceCount(&ndev);
    if (ndev == 0) {
        printf("No CUDA device found.\n");
        return 1;
    }
    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("Memory: %.1f GiB\n",
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    printf("Warmup: %d, Iters: %d\n\n", cfg.warmup, cfg.iters);

    // Run benchmarks for each grid size
    printf("%-8s %-12s %-12s %-12s %-12s %-12s\n",
           "Size", "Nodes", "Sweep(ms)", "Graph(ms)", "Mem(MiB)", "BW(GB/s)");
    printf("-------- ------------ ------------ ------------ ------------ ------------\n");

    std::vector<BenchResult> results;
    for (int n : cfg.sizes) {
        BenchResult r = benchmark_sweep(n, cfg.warmup, cfg.iters);
        results.push_back(r);
        printf("%-8d %-12d %-12.3f %-12.3f %-12.1f %-12.1f\n",
               r.grid_size, r.total_nodes, r.sweep_time_ms,
               r.graph_replay_ms, r.total_mem_mb, r.bandwidth_gbs);
    }

    // Summary
    printf("\n============================================================\n");
    printf("  Benchmark Summary\n");
    printf("============================================================\n");
    for (const auto& r : results) {
        printf("  Grid %d^3 (%d nodes):\n", r.grid_size, r.total_nodes);
        printf("    Sweep time:     %.3f ms\n", r.sweep_time_ms);
        printf("    Graph replay:   %.3f ms\n", r.graph_replay_ms);
        printf("    Memory used:    %.1f MiB\n", r.total_mem_mb);
        printf("    Bandwidth:      %.1f GB/s\n", r.bandwidth_gbs);
        printf("\n");
    }

    // Print comparison with old GPU backend (if data available)
    printf("============================================================\n");
    printf("  Memory Comparison (vs. legacy 8x duplication)\n");
    printf("============================================================\n");
    for (const auto& r : results) {
        idx_t N = r.total_nodes;
        // New backend: 11 fields * N * 8 bytes (real_t) + 1 field * N * 1 byte (uint8)
        //              + level indices * N * 4 bytes (idx_t)
        double new_mem = 11.0 * N * sizeof(real_t) + N * sizeof(uint8_t) + N * sizeof(idx_t);
        // Old backend: 8 * (11 fields * N * 8 bytes + 7 index arrays * N * 4 bytes)
        //              + 1 field * N * 8 bytes (tau)
        double old_mem = 8.0 * (11.0 * N * sizeof(real_t) + 7.0 * N * sizeof(idx_t)) + N * sizeof(real_t);
        printf("  Grid %d^3: New=%.1f MiB, Old=%.1f MiB, Reduction=%.1fx\n",
               r.grid_size,
               new_mem / (1024.0 * 1024.0),
               old_mem / (1024.0 * 1024.0),
               old_mem / new_mem);
    }

    return 0;
}
