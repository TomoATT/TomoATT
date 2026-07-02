//
// bench_cpu_vs_gpu.cu — Comprehensive CPU vs GPU v2 benchmark.
//
// Measures:
//   1. Traveltime field errors (CPU reference vs GPU v2)
//   2. Time to completion (wall-clock)
//   3. RAM consumption (peak RSS + GPU memory)
//
// Two stopping modes:
//   --mode iters    : fixed number of sweeps (default)
//   --mode conv     : iterate until L1 change < conv_tol (convergence-based)
//
// Outputs:
//   - Console table with results
//   - JSON file with structured results (benchmark_results.json)
//   - Raw binary T fields for Python visualization (T_cpu_N.raw, T_gpu_N.raw)
//
// Usage:
//   ./bench_cpu_vs_gpu [--sizes 32,64,128,256] [--iters 20] [--warmup 5]
//                      [--mode conv] [--conv-tol 1e-6] [--max-iters 1000]
//                      [--output-dir .] [--no-save-fields]
//
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
// Utility: peak RSS in MiB
// ---------------------------------------------------------------------------
#ifdef __linux__
#include <sys/resource.h>
static double get_peak_rss_mb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // KB→MiB
}
#else
static double get_peak_rss_mb() { return 0.0; }
#endif

// ---------------------------------------------------------------------------
// CPU reference sweep — scalar (non-SIMD) implementation.
// This is the baseline CPU path: a simple triple-loop with no SIMD
// intrinsics, matching the GPU v2 backend math exactly.
// Returns the L1 change in tau (for convergence checking).
// ---------------------------------------------------------------------------
static double cpu_sweep_once(
    real_t* tau,
    real_t* tau_old,
    const real_t* T0v, const real_t* T0r, const real_t* T0t, const real_t* T0p,
    const real_t* fac_a, const real_t* fac_b, const real_t* fac_c, const real_t* fac_f,
    const real_t* fun, const uint8_t* is_changed,
    int nx, int ny, int nz, real_t dr, real_t dt, real_t dp) {

    const int max_level = nx + ny + nz - 3;
    const real_t half   = static_cast<real_t>(0.5);
    const real_t two    = static_cast<real_t>(2.0);

    // Build level decomposition (host-side, same as GPU)
    std::vector<std::vector<int>> level_nodes(max_level + 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int level = i + j + k;
                if (level <= max_level) {
                    level_nodes[level].push_back(i + j * nx + k * nx * ny);
                }
            }
        }
    }

    // Save tau to tau_old for convergence checking
    std::memcpy(tau_old, tau, sizeof(real_t) * nx * ny * nz);

    for (int dir = 0; dir < 8; ++dir) {
        int sign_r = (dir < 4) ? +1 : -1;
        int sign_t = (dir % 4 < 2) ? +1 : -1;
        int sign_p = (dir % 2 == 0) ? +1 : -1;

        for (int level = 0; level <= max_level; ++level) {
            for (int idx : level_nodes[level]) {
                int k = idx / (nx * ny);
                int j = (idx - k * nx * ny) / nx;
                int i = idx - k * nx * ny - j * nx;

                int phys_i = (sign_p > 0) ? (i - 1) : (nx - i);
                int phys_j = (sign_t > 0) ? (j - 1) : (ny - j);
                int phys_k = (sign_r > 0) ? (k - 1) : (nz - k);

                if (phys_i <= 0 || phys_i >= nx - 1 ||
                    phys_j <= 0 || phys_j >= ny - 1 ||
                    phys_k <= 0 || phys_k >= nz - 1) continue;

                int c = phys_k * nx * ny + phys_j * nx + phys_i;
                if (!is_changed[c]) continue;

                int cm1 = c - 1,      cp1 = c + 1;
                int jm1 = c - nx,     jp1 = c + nx;
                int km1 = c - nx * ny, kp1 = c + nx * ny;

                real_t dp_inv = 1.0 / dp;
                real_t dt_inv = 1.0 / dt;
                real_t dr_inv = 1.0 / dr;

                real_t pp1 = (tau[c] - tau[cm1]) * dp_inv;
                real_t pp2 = (tau[cp1] - tau[c]) * dp_inv;
                real_t pt1 = (tau[c] - tau[jm1]) * dt_inv;
                real_t pt2 = (tau[jp1] - tau[c]) * dt_inv;
                real_t pr1 = (tau[c] - tau[km1]) * dr_inv;
                real_t pr2 = (tau[kp1] - tau[c]) * dr_inv;

                real_t sigr = std::sqrt(fac_a[c]) * T0v[c];
                real_t sigt = std::sqrt(fac_b[c]) * T0v[c];
                real_t sigp = std::sqrt(fac_c[c]) * T0v[c];
                real_t coe  = 1.0 / (sigr / dr + sigt / dt + sigp / dp);

                real_t dr_term = T0r[c] * tau[c] + T0v[c] * (pr1 + pr2) * half;
                real_t dt_term = T0t[c] * tau[c] + T0v[c] * (pt1 + pt2) * half;
                real_t dp_term = T0p[c] * tau[c] + T0v[c] * (pp1 + pp2) * half;

                real_t Htau = std::sqrt(
                    fac_a[c] * dr_term * dr_term +
                    fac_b[c] * dt_term * dt_term +
                    fac_c[c] * dp_term * dp_term -
                    two * fac_f[c] * dt_term * dp_term);

                real_t correction =
                    (sigr * (pr2 - pr1) + sigt * (pt2 - pt1) + sigp * (pp2 - pp1)) * half;

                tau[c] += coe * (fun[c] - Htau + correction);
            }
        }
    }

    // Compute L1 change
    double l1_change = 0.0;
    int total = nx * ny * nz;
    for (int i = 0; i < total; ++i) {
        l1_change += std::abs(tau[i] - tau_old[i]);
    }
    l1_change /= total;
    return l1_change;
}

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------
struct BenchConfig {
    std::vector<int> sizes = {32, 64, 128, 256};
    int warmup = 5;
    int iters  = 20;
    // Convergence-based stopping
    std::string mode = "iters";       // "iters" or "conv"
    double conv_tol = 1e-6;
    int max_iters = 1000;
    // Output
    bool save_fields = true;
    std::string output_dir = ".";
};

static BenchConfig parse_args(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) {
            cfg.sizes.clear();
            std::string s = argv[++i];
            size_t pos;
            while ((pos = s.find(',')) != std::string::npos) {
                cfg.sizes.push_back(std::stoi(s.substr(0, pos)));
                s = s.substr(pos + 1);
            }
            cfg.sizes.push_back(std::stoi(s));
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iters = std::stoi(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            cfg.mode = argv[++i];
        } else if (arg == "--conv-tol" && i + 1 < argc) {
            cfg.conv_tol = std::stod(argv[++i]);
        } else if (arg == "--max-iters" && i + 1 < argc) {
            cfg.max_iters = std::stoi(argv[++i]);
        } else if (arg == "--no-save-fields") {
            cfg.save_fields = false;
        } else if (arg == "--output-dir" && i + 1 < argc) {
            cfg.output_dir = argv[++i];
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Save 3D field to raw binary file (for Python visualization)
// ---------------------------------------------------------------------------
static void save_field_raw(const std::string& filename, const real_t* data, int nx, int ny, int nz) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        fprintf(stderr, "Warning: cannot open %s for writing\n", filename.c_str());
        return;
    }
    int32_t dims[3] = {nx, ny, nz};
    ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(nx) * ny * nz * sizeof(real_t));
    printf("  Saved field: %s (%dx%dx%d, %.1f MiB)\n", filename.c_str(), nx, ny, nz,
           nx * ny * nz * sizeof(real_t) / (1024.0 * 1024.0));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    BenchConfig cfg = parse_args(argc, argv);

    printf("============================================================\n");
    printf("  CPU vs GPU v2 Benchmark Suite\n");
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
    printf("Sizes: ");
    for (int s : cfg.sizes) printf("%d ", s);
    printf("\nWarmup: %d", cfg.warmup);
    if (cfg.mode == "conv") {
        printf(", Mode: convergence (tol=%.2e, max_iters=%d)", cfg.conv_tol, cfg.max_iters);
    } else {
        printf(", Mode: fixed iters (%d)", cfg.iters);
    }
    printf("\n\n");

    // Print header
    printf("%-6s %-10s | %-10s %-10s %-8s | %-12s %-12s | %-10s %-10s %-10s\n",
           "Grid", "Nodes",
           "CPU(ms)", "GPU(ms)", "Speedup",
           "L1_err", "Linf_err",
           "CPU_RSS", "GPU_Mem", "Total_Mem");
    printf("------ ---------- | ---------- ---------- -------- | ------------ ------------ | ---------- ---------- ----------\n");

    struct Result {
        int grid_size;
        idx_t total_nodes;
        double cpu_ms;
        double gpu_ms;
        double speedup;
        double l1_err;
        double linf_err;
        double cpu_rss_mb;
        double gpu_mem_mb;
        double total_mem_mb;
        int cpu_iters_done;
        int gpu_iters_done;
        double cpu_l1_final;
        double gpu_l1_final;
    };
    std::vector<Result> results;

    // Open JSON output file
    std::string json_path = cfg.output_dir + "/benchmark_results.json";
    FILE* json_fp = fopen(json_path.c_str(), "w");
    if (json_fp) {
        fprintf(json_fp, "{\n");
        fprintf(json_fp, "  \"device\": \"%s\",\n", prop.name);
        fprintf(json_fp, "  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
        fprintf(json_fp, "  \"device_memory_gib\": %.1f,\n", static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
        fprintf(json_fp, "  \"mode\": \"%s\",\n", cfg.mode.c_str());
        fprintf(json_fp, "  \"conv_tol\": %.2e,\n", cfg.conv_tol);
        fprintf(json_fp, "  \"max_iters\": %d,\n", cfg.max_iters);
        fprintf(json_fp, "  \"warmup\": %d,\n", cfg.warmup);
        fprintf(json_fp, "  \"iters\": %d,\n", cfg.iters);
        fprintf(json_fp, "  \"results\": [\n");
    }

    for (size_t s_idx = 0; s_idx < cfg.sizes.size(); ++s_idx) {
        int n = cfg.sizes[s_idx];
        GridDims dims{n, n, n};
        const idx_t N = dims.total();

        // Initialize fields on host (constant velocity, single source)
        std::vector<real_t> h_T0v(N, 1.0), h_T0r(N, 0.0), h_T0t(N, 0.0), h_T0p(N, 0.0);
        std::vector<real_t> h_fac_a(N, 1.0), h_fac_b(N, 1.0), h_fac_c(N, 1.0), h_fac_f(N, 0.0);
        std::vector<real_t> h_fun(N, 1.0);
        std::vector<uint8_t> h_changed(N, 1);

        // Set boundary nodes to not changed
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    if (i == 0 || i == n-1 || j == 0 || j == n-1 || k == 0 || k == n-1) {
                        h_changed[k*n*n + j*n + i] = 0;
                    }
                }
            }
        }

        // --- CPU reference ---
        std::vector<real_t> h_tau_cpu(N, 1.0);
        std::vector<real_t> h_tau_cpu_old(N, 1.0);
        double cpu_rss_before = get_peak_rss_mb();
        auto t0 = Clock::now();

        int cpu_iters_done = 0;
        double cpu_l1_final = 0.0;
        int cpu_max = (cfg.mode == "conv") ? cfg.max_iters : cfg.iters;
        for (int it = 0; it < cpu_max; ++it) {
            double l1_change = cpu_sweep_once(h_tau_cpu.data(), h_tau_cpu_old.data(),
                  h_T0v.data(), h_T0r.data(), h_T0t.data(), h_T0p.data(),
                  h_fac_a.data(), h_fac_b.data(), h_fac_c.data(), h_fac_f.data(),
                  h_fun.data(), h_changed.data(),
                  n, n, n, 1.0, 1.0, 1.0);
            cpu_iters_done++;
            cpu_l1_final = l1_change;
            if (cfg.mode == "conv" && l1_change < cfg.conv_tol) break;
        }

        auto t1 = Clock::now();
        double cpu_rss_after = get_peak_rss_mb();
        double cpu_ms = ms_t(t1 - t0).count();
        double cpu_rss = std::max(cpu_rss_before, cpu_rss_after);

        // --- GPU v2 ---
        std::vector<real_t> h_tau_gpu(N, 1.0);
        GpuSweepBackend backend(dims, 1.0, 1.0, 1.0, StencilOrder::First);
        backend.initialize_from_host(
            h_T0v.data(), h_T0r.data(), h_T0t.data(), h_T0p.data(),
            h_fac_a.data(), h_fac_b.data(), h_fac_c.data(), h_fac_f.data(),
            h_fun.data(), h_changed.data(), h_tau_gpu.data());

        // Measure GPU memory after allocation
        size_t free_gpu, total_gpu;
        cudaMemGetInfo(&free_gpu, &total_gpu);
        double gpu_mem_mb = static_cast<double>(total_gpu - free_gpu) / (1024.0 * 1024.0);

        // Warmup
        for (int i = 0; i < cfg.warmup; ++i) {
            backend.run_sweep();
        }
        cudaDeviceSynchronize();

        // Benchmark — fixed iters or convergence-based
        int gpu_iters_done = 0;
        double gpu_l1_final = 0.0;
        auto t2 = Clock::now();

        if (cfg.mode == "conv") {
            // Convergence-based: iterate until L1 < conv_tol
            int gpu_max = cfg.max_iters;
            for (int it = 0; it < gpu_max; ++it) {
                backend.store_tau_to_old();
                backend.run_sweep();
                real_t l1_val, linf_val;
                backend.compute_convergence(l1_val, linf_val);
                gpu_iters_done++;
                gpu_l1_final = l1_val;
                if (l1_val < cfg.conv_tol) break;
            }
        } else {
            // Fixed-iteration mode
            int gpu_max = cfg.iters;
            for (int it = 0; it < gpu_max; ++it) {
                backend.store_tau_to_old();
                backend.run_sweep();
                real_t l1_val, linf_val;
                backend.compute_convergence(l1_val, linf_val);
                gpu_iters_done++;
                gpu_l1_final = l1_val;
            }
        }
        cudaDeviceSynchronize();
        auto t3 = Clock::now();
        double gpu_ms = (gpu_iters_done > 0) ? (ms_t(t3 - t2).count() / gpu_iters_done) : 0;

        // Copy GPU tau back
        backend.copy_tau_to_host(h_tau_gpu.data());
        cudaDeviceSynchronize();

        // Convert tau → T (traveltime):  T = T0v × tau
        std::vector<real_t> h_T_cpu(N), h_T_gpu(N);
        for (idx_t i = 0; i < N; ++i) {
            h_T_cpu[i] = h_T0v[i] * h_tau_cpu[i];
            h_T_gpu[i] = h_T0v[i] * h_tau_gpu[i];
        }

        // Compare CPU and GPU T fields
        double l1_err = 0, linf_err = 0;
        for (idx_t i = 0; i < N; ++i) {
            double diff = std::abs(h_T_cpu[i] - h_T_gpu[i]);
            l1_err += diff;
            if (diff > linf_err) linf_err = diff;
        }
        l1_err /= N;

        double speedup = (gpu_ms > 0) ? (cpu_ms / gpu_ms) : 0;
        double total_mem = cpu_rss + gpu_mem_mb;

        printf("%-6d %-10d | %-10.3f %-10.3f %-8.2f | %-12.6e %-12.6e | %-10.1f %-10.1f %-10.1f\n",
               n, N, cpu_ms, gpu_ms, speedup, l1_err, linf_err, cpu_rss, gpu_mem_mb, total_mem);

        // Save T (traveltime) fields for visualization
        if (cfg.save_fields) {
            char cpu_path[1024], gpu_path[1024];
            snprintf(cpu_path, sizeof(cpu_path), "%s/T_cpu_%d.raw", cfg.output_dir.c_str(), n);
            snprintf(gpu_path, sizeof(gpu_path), "%s/T_gpu_%d.raw", cfg.output_dir.c_str(), n);
            save_field_raw(cpu_path, h_T_cpu.data(), n, n, n);
            save_field_raw(gpu_path, h_T_gpu.data(), n, n, n);
        }

        // Write JSON entry
        if (json_fp) {
            if (s_idx > 0) fprintf(json_fp, ",\n");
            fprintf(json_fp, "    {\n");
            fprintf(json_fp, "      \"grid_size\": %d,\n", n);
            fprintf(json_fp, "      \"total_nodes\": %d,\n", static_cast<int>(N));
            fprintf(json_fp, "      \"cpu_time_ms\": %.6f,\n", cpu_ms);
            fprintf(json_fp, "      \"gpu_time_ms\": %.6f,\n", gpu_ms);
            fprintf(json_fp, "      \"speedup\": %.6f,\n", speedup);
            fprintf(json_fp, "      \"l1_error\": %.6e,\n", l1_err);
            fprintf(json_fp, "      \"linf_error\": %.6e,\n", linf_err);
            fprintf(json_fp, "      \"cpu_rss_mb\": %.6f,\n", cpu_rss);
            fprintf(json_fp, "      \"gpu_mem_mb\": %.6f,\n", gpu_mem_mb);
            fprintf(json_fp, "      \"total_mem_mb\": %.6f,\n", total_mem);
            fprintf(json_fp, "      \"cpu_iters_done\": %d,\n", cpu_iters_done);
            fprintf(json_fp, "      \"gpu_iters_done\": %d,\n", gpu_iters_done);
            fprintf(json_fp, "      \"cpu_l1_final\": %.6e,\n", cpu_l1_final);
            fprintf(json_fp, "      \"gpu_l1_final\": %.6e\n", gpu_l1_final);
            fprintf(json_fp, "    }");
        }

        results.push_back({n, N, cpu_ms, gpu_ms, speedup,
                           l1_err, linf_err, cpu_rss, gpu_mem_mb, total_mem,
                           cpu_iters_done, gpu_iters_done, cpu_l1_final, gpu_l1_final});
    }

    // Close JSON
    if (json_fp) {
        fprintf(json_fp, "\n  ]\n");
        fprintf(json_fp, "}\n");
        fclose(json_fp);
        printf("\n  JSON results saved to: %s\n", json_path.c_str());
    }

    // Summary
    printf("\n============================================================\n");
    printf("  Benchmark Summary\n");
    printf("============================================================\n\n");

    printf("Grid Size | Nodes    | CPU Time  | GPU Time  | Speedup | L1 Error   | Linf Error  | CPU RSS  | GPU Mem  | Total Mem\n");
    printf("----------|----------|-----------|-----------|---------|------------|-------------|----------|----------|----------\n");
    for (const auto& r : results) {
        printf("%-9d | %-8d | %-9.3f | %-9.3f | %-7.2f | %-10.6e | %-11.6e | %-8.1f | %-8.1f | %-8.1f\n",
               r.grid_size, r.total_nodes, r.cpu_ms, r.gpu_ms, r.speedup,
               r.l1_err, r.linf_err, r.cpu_rss_mb, r.gpu_mem_mb, r.total_mem_mb);
    }

    // Scaling analysis
    printf("\n============================================================\n");
    printf("  Scaling Analysis\n");
    printf("============================================================\n\n");
    printf("Grid Size | Nodes    | CPU Time/sweep | GPU Time/sweep | CPU→GPU Speedup\n");
    printf("----------|----------|----------------|----------------|----------------\n");
    for (const auto& r : results) {
        printf("%-9d | %-8d | %-14.3f | %-14.3f | %-14.2f\n",
               r.grid_size, r.total_nodes,
               r.cpu_ms, r.gpu_ms, r.speedup);
    }

    printf("\n============================================================\n");
    printf("  Memory Comparison\n");
    printf("============================================================\n\n");
    printf("Grid Size | Nodes    | CPU RSS (MiB) | GPU Mem (MiB) | Total (MiB)\n");
    printf("----------|----------|----------------|----------------|------------\n");
    for (const auto& r : results) {
        printf("%-9d | %-8d | %-14.1f | %-14.1f | %-10.1f\n",
               r.grid_size, r.total_nodes,
               r.cpu_rss_mb, r.gpu_mem_mb, r.total_mem_mb);
    }

    printf("\n============================================================\n");
    printf("  Traveltime (T) Error Analysis\n");
    printf("  T = T0v × tau  (physical traveltime)\n");
    printf("============================================================\n\n");
    printf("Grid Size | Nodes    | L1 Error    | Linf Error\n");
    printf("----------|----------|--------------|------------\n");
    for (const auto& r : results) {
        printf("%-9d | %-8d | %-12.6e | %-10.6e\n",
               r.grid_size, r.total_nodes, r.l1_err, r.linf_err);
    }

    // Convergence info (only relevant in conv mode)
    if (cfg.mode == "conv") {
        printf("\n============================================================\n");
        printf("  Convergence Analysis (tol=%.2e)\n", cfg.conv_tol);
        printf("============================================================\n\n");
        printf("Grid Size | Nodes    | CPU Iters | GPU Iters | CPU L1_final   | GPU L1_final\n");
        printf("----------|----------|-----------|-----------|----------------|----------------\n");
        for (const auto& r : results) {
            printf("%-9d | %-8d | %-9d | %-9d | %-14.6e | %-14.6e\n",
                   r.grid_size, r.total_nodes,
                   r.cpu_iters_done, r.gpu_iters_done,
                   r.cpu_l1_final, r.gpu_l1_final);
        }
    }

    return 0;
}
