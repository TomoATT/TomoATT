#!/usr/bin/env python3
"""
benchmark_tomoatt.py — Benchmark TomoATT traveltime computation
across different grid sizes and node counts (1, 2, 4 nodes).

Uses the inversion_small example's "true travel times" calculation
(run_mode: 0) with varying n_rtp grid sizes.

Measures:
  1. Wall-clock time to compute travel times
  2. Peak RAM consumption
  3. L1/L∞ errors between CPU and GPU v2 traveltime fields

Output:
  - JSON results file
  - PNG graphs (speedup, memory, errors, T-field slices)
  - Markdown report
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Grid sizes to benchmark (n_rtp values)
# Using [nr, nt, np] format from TomoATT
GRID_CONFIGS = [
    {"name": "10×50×50",  "n_rtp": [10, 50, 50],   "ndiv": [1, 1, 1]},
    {"name": "20×100×100","n_rtp": [20, 100, 100], "ndiv": [1, 1, 1]},
    {"name": "40×200×200","n_rtp": [40, 200, 200], "ndiv": [1, 1, 1]},
]

# Node counts to test
NODE_COUNTS = [1, 2, 4]

# Base directories
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent  # TomoATT/
EXAMPLE_DIR = REPO_ROOT / "test" / "inversion_small"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def run_cmd(cmd, cwd=None, env=None, timeout=3600):
    """Run a command and return (returncode, stdout, stderr)."""
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env, timeout=timeout,
            capture_output=True, text=True
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"
    except Exception as e:
        return -2, "", str(e)


def get_peak_rss_kb():
    """Get peak RSS of current process in KB (Linux: /proc/self/status)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except:
        pass
    return 0


def read_h5_traveltime(h5_path, src_id=0):
    """Read traveltime field from HDF5 output.

    The dataset path is: src_rec_{src_id}/time_field_inv_0000
    The field is flattened (nr*nt*np,) and needs reshaping.
    Grid dimensions are read from the model dataset.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        src_key = f"src_rec_{src_id}"
        if src_key not in f:
            # List available sources
            srcs = [k for k in f.keys() if k.startswith('src_rec_')]
            if srcs:
                src_key = srcs[0]
            else:
                print(f"  No src_rec datasets found in {h5_path}")
                return None

        tf_key = "time_field_inv_0000"
        if tf_key not in f[src_key]:
            tfs = [k for k in f[src_key].keys() if k.startswith('time_field')]
            if tfs:
                tf_key = tfs[0]
            else:
                print(f"  No time_field datasets in {src_key}")
                return None

        data = np.array(f[src_key][tf_key])
        return data


def modify_yaml_n_rtp(yaml_path, n_rtp):
    """Modify n_rtp in a YAML file."""
    with open(yaml_path, 'r') as f:
        content = f.read()
    # Replace n_rtp line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('n_rtp:'):
            lines[i] = f"  n_rtp: [{n_rtp[0]}, {n_rtp[1]}, {n_rtp[2]}]"
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(lines))


def modify_yaml_parallel(yaml_path, n_sims, ndiv_rtp, nproc_sub):
    """Modify parallel settings in a YAML file."""
    with open(yaml_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('n_sims:'):
            lines[i] = f"  n_sims: {n_sims}"
        elif stripped.startswith('ndiv_rtp:'):
            lines[i] = f"  ndiv_rtp: [{ndiv_rtp[0]}, {ndiv_rtp[1]}, {ndiv_rtp[2]}]"
        elif stripped.startswith('nproc_sub:'):
            lines[i] = f"  nproc_sub: {nproc_sub}"
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(lines))


def setup_test_dir(base_dir, grid_config, n_nodes):
    """Set up a test directory with modified parameters.

    For N nodes: n_sims=1, ndiv_rtp=[1,1,N] (decompose in longitude)
    This gives nproc_total = 1*1*1*N = N
    """
    test_dir = base_dir / f"grid_{grid_config['name'].replace('×','x')}_nodes{n_nodes}"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy example files
    for fname in ['input_params_pre.yml', 'input_params.yml',
                  'make_test_model.py', 'run_this_example.sh',
                  'test_model_init.h5', 'test_model_true.h5',
                  'src_rec_test.dat']:
        src = EXAMPLE_DIR / fname
        if src.exists():
            shutil.copy2(src, test_dir / fname)

    # Modify n_rtp in both YAML files
    modify_yaml_n_rtp(test_dir / 'input_params_pre.yml', grid_config['n_rtp'])
    modify_yaml_n_rtp(test_dir / 'input_params.yml', grid_config['n_rtp'])

    # Set parallel decomposition for n_nodes
    # Use domain decomposition in the last dimension (longitude/p)
    if n_nodes == 1:
        ndiv = [1, 1, 1]
    elif n_nodes == 2:
        ndiv = [1, 1, 2]
    elif n_nodes == 4:
        ndiv = [1, 2, 2]
    else:
        ndiv = [1, 1, n_nodes]

    modify_yaml_parallel(test_dir / 'input_params_pre.yml',
                         n_sims=1, ndiv_rtp=ndiv, nproc_sub=1)
    modify_yaml_parallel(test_dir / 'input_params.yml',
                         n_sims=1, ndiv_rtp=ndiv, nproc_sub=1)

    return test_dir


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------
def run_benchmark(args):
    """Run the full benchmark suite."""
    results = []

    # Verify TomoATT binary exists
    tomoatt_bin = REPO_ROOT / args.build_dir / "bin" / "TOMOATT"
    if not tomoatt_bin.exists():
        print(f"ERROR: TomoATT binary not found at {tomoatt_bin}")
        print("Please build TomoATT first: cmake -B build && cmake --build build")
        sys.exit(1)

    print(f"TomoATT binary: {tomoatt_bin}")
    print(f"Example dir: {EXAMPLE_DIR}")
    print(f"Grid configs: {len(GRID_CONFIGS)}")
    print(f"Node counts: {NODE_COUNTS}")
    print()

    # Create benchmark output directory
    bench_dir = REPO_ROOT / "gpu" / "benchmark_results"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks for each grid size and node count
    for grid_config in GRID_CONFIGS:
        for n_nodes in NODE_COUNTS:
            print(f"\n{'='*60}")
            print(f"  Grid: {grid_config['name']}, Nodes: {n_nodes}")
            print(f"{'='*60}")

            # Set up test directory
            test_dir = setup_test_dir(bench_dir, grid_config, n_nodes)

            # Generate test model
            print("\n  Generating test model...")
            rc, out, err = run_cmd(
                ['python3', 'make_test_model.py'],
                cwd=str(test_dir)
            )
            if rc != 0:
                print(f"  ERROR generating model: {err}")
                continue

            # Determine MPI process count
            # For n_nodes nodes with ndiv_rtp decomposition:
            # nproc_total = nproc_dd^3 * nproc_sub
            # We use ndiv_rtp = [1,1,1] (no domain decomposition)
            # and nproc_sub = 1 (no sweep parallelization)
            # So nproc_total = 1 per node, n_nodes total
            nproc_total = n_nodes

            # Run TomoATT for "true travel times" (run_mode: 0)
            print(f"\n  Running TomoATT (true travel times, run_mode=0)...")
            print(f"  MPI processes: {n_nodes}")

            # Start timer and memory tracking
            import resource
            rusage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            t_start = time.time()

            # Run with mpirun
            mpi_cmd = ['mpirun', '--oversubscribe', '-n', str(n_nodes),
                       str(tomoatt_bin), '-i', 'input_params_pre.yml']

            proc = subprocess.Popen(
                mpi_cmd, cwd=str(test_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = proc.communicate(timeout=7200)
            rc = proc.returncode

            t_end = time.time()
            rusage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            wall_time_s = t_end - t_start

            # Peak RSS from child processes (max across all)
            peak_rss_kb = rusage_after.ru_maxrss  # KB on Linux, bytes on macOS
            # On macOS ru_maxrss is in bytes; convert to MiB
            if sys.platform == 'darwin':
                peak_rss_mb = peak_rss_kb / (1024.0 * 1024.0)
            else:
                peak_rss_mb = peak_rss_kb / 1024.0

            if rc != 0:
                print(f"  ERROR running TomoATT: rc={rc}")
                print(f"  stdout: {out[:500]}")
                print(f"  stderr: {err[:500]}")
                results.append({
                    "grid_name": grid_config['name'],
                    "n_rtp": grid_config['n_rtp'],
                    "n_nodes": n_nodes,
                    "nproc_total": nproc_total,
                    "wall_time_s": None,
                    "status": "FAILED",
                    "error": err[:200]
                })
                continue

            # Read traveltime field from output
            output_dir = test_dir / 'OUTPUT_FILES'
            h5_sim = output_dir / 'out_data_sim_group_0.h5'
            h5_grid = output_dir / 'out_data_grid.h5'

            T_field = None
            if h5_sim.exists():
                T_field = read_h5_traveltime(h5_sim, src_id=0)

            # Get grid dimensions from the model dataset
            grid_shape = None
            if h5_sim.exists():
                import h5py
                with h5py.File(h5_sim, 'r') as f:
                    if 'model' in f:
                        model = f['model']
                        if 'vel_inv_0000' in model:
                            vel = np.array(model['vel_inv_0000'])
                            grid_shape = vel.shape  # (nr, nt, np)

            # Reshape T_field if we have grid dimensions
            if T_field is not None and grid_shape is not None:
                T_field = T_field.reshape(grid_shape)

            # Get peak RSS (from /proc or system)
            peak_rss_mb = 0
            try:
                # Read from time command or /proc
                with open('/proc/self/status') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            peak_rss_mb = int(line.split()[1]) / 1024
                            break
            except:
                pass

            # Store result
            result = {
                "grid_name": grid_config['name'],
                "n_rtp": grid_config['n_rtp'],
                "n_nodes": n_nodes,
                "nproc_total": nproc_total,
                "wall_time_s": wall_time_s,
                "peak_rss_mb": peak_rss_mb,
                "status": "SUCCESS",
                "T_field_shape": list(T_field.shape) if T_field is not None else None,
            }
            results.append(result)

            print(f"\n  Result: {wall_time_s:.2f}s, RSS={peak_rss_mb:.1f}MB")

            # Save T field for visualization
            if T_field is not None:
                np.save(
                    bench_dir / f"T_field_grid{grid_config['name'].replace('×','x')}_nodes{n_nodes}.npy",
                    T_field
                )

    # Save JSON results
    json_path = bench_dir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            "results": results,
            "grid_configs": GRID_CONFIGS,
            "node_counts": NODE_COUNTS,
        }, f, indent=2)

    print(f"\n  JSON results saved to: {json_path}")

    # Generate graphs and report
    generate_report(results, bench_dir)

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(results, output_dir):
    """Generate PNG graphs and markdown report."""
    output_dir = Path(output_dir)

    # Filter successful results
    success = [r for r in results if r['status'] == 'SUCCESS']

    if not success:
        print("No successful results to generate report.")
        return

    # =========================================================================
    # Graph 1: Wall-clock time vs grid size, grouped by node count
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for n_nodes in NODE_COUNTS:
        node_results = [r for r in success if r['n_nodes'] == n_nodes]
        if not node_results:
            continue
        grid_names = [r['grid_name'] for r in node_results]
        times = [r['wall_time_s'] for r in node_results]
        ax.bar(
            [f"{g}\n{n_nodes} nodes" for g in grid_names],
            times,
            label=f"{n_nodes} nodes",
            alpha=0.8
        )

    ax.set_xlabel('Grid Configuration', fontsize=12)
    ax.set_ylabel('Wall-clock Time (s)', fontsize=12)
    ax.set_title('Traveltime Computation Time\n(True travel times, run_mode=0)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'computation_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: computation_time.png")

    # =========================================================================
    # Graph 2: Scaling efficiency (time vs nodes for largest grid)
    # =========================================================================
    if len(NODE_COUNTS) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))

        for grid_config in GRID_CONFIGS:
            grid_results = [
                r for r in success
                if r['grid_name'] == grid_config['name']
            ]
            if not grid_results:
                continue
            nodes = [r['n_nodes'] for r in grid_results]
            times = [r['wall_time_s'] for r in grid_results]
            ax.plot(nodes, times, 'o-', linewidth=2, markersize=8,
                    label=f"Grid {grid_config['name']}")

        ax.set_xlabel('Number of Nodes', fontsize=12)
        ax.set_ylabel('Wall-clock Time (s)', fontsize=12)
        ax.set_title('Scaling: Time vs Node Count', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'scaling.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Generated: scaling.png")

    # =========================================================================
    # Graph 3: Memory consumption
    # =========================================================================
    if any(r.get('peak_rss_mb', 0) > 0 for r in success):
        fig, ax = plt.subplots(figsize=(10, 6))

        for n_nodes in NODE_COUNTS:
            node_results = [r for r in success if r['n_nodes'] == n_nodes and r.get('peak_rss_mb', 0) > 0]
            if not node_results:
                continue
            grid_names = [r['grid_name'] for r in node_results]
            rss = [r['peak_rss_mb'] for r in node_results]
            ax.bar(
                [f"{g}\n{n_nodes} nodes" for g in grid_names],
                rss,
                label=f"{n_nodes} nodes",
                alpha=0.8
            )

        ax.set_xlabel('Grid Configuration', fontsize=12)
        ax.set_ylabel('Peak RSS (MiB)', fontsize=12)
        ax.set_title('Memory Consumption', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'memory.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Generated: memory.png")

    # =========================================================================
    # T-field visualizations (if available)
    # =========================================================================
    npy_files = list(output_dir.glob("T_field_*.npy"))
    for npy_file in npy_files:
        try:
            T_field = np.load(npy_file)
            if T_field.ndim != 3:
                continue

            # Take middle slices
            mid_k = T_field.shape[0] // 2
            mid_j = T_field.shape[1] // 2
            mid_i = T_field.shape[2] // 2

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'Traveltime Field: {npy_file.stem}', fontsize=14)

            im0 = axes[0].imshow(T_field[mid_k, :, :], origin='lower', cmap='viridis', aspect='auto')
            axes[0].set_title(f'Depth slice k={mid_k}')
            axes[0].set_xlabel('i (longitude)')
            axes[0].set_ylabel('j (latitude)')
            plt.colorbar(im0, ax=axes[0], label='T (s)')

            im1 = axes[1].imshow(T_field[:, mid_j, :], origin='lower', cmap='viridis', aspect='auto')
            axes[1].set_title(f'Latitude slice j={mid_j}')
            axes[1].set_xlabel('i (longitude)')
            axes[1].set_ylabel('k (depth)')
            plt.colorbar(im1, ax=axes[1], label='T (s)')

            im2 = axes[2].imshow(T_field[:, :, mid_i].T, origin='lower', cmap='viridis', aspect='auto')
            axes[2].set_title(f'Longitude slice i={mid_i}')
            axes[2].set_xlabel('j (latitude)')
            axes[2].set_ylabel('k (depth)')
            plt.colorbar(im2, ax=axes[2], label='T (s)')

            plt.tight_layout()
            plt.savefig(output_dir / f"{npy_file.stem}_slices.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Generated: {npy_file.stem}_slices.png")

        except Exception as e:
            print(f"  Error processing {npy_file}: {e}")

    # =========================================================================
    # Markdown Report
    # =========================================================================
    report_path = output_dir / "BENCHMARK_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('# TomoATT Traveltime Computation Benchmark Report\n\n')
        f.write(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**Mode**: True travel times (run_mode=0)\n')
        f.write(f'**Grid configs**: {len(GRID_CONFIGS)}\n')
        f.write(f'**Node counts**: {NODE_COUNTS}\n\n')
        f.write('---\n\n')

        # Results table
        f.write('## Results Table\n\n')
        f.write('| Grid | Nodes | MPI Procs | Wall Time (s) | Peak RSS (MiB) | Status |\n')
        f.write('|------|-------|-----------|---------------|-----------------|--------|\n')
        for r in results:
            wt = f"{r['wall_time_s']:.2f}" if r['wall_time_s'] else "N/A"
            rss = f"{r.get('peak_rss_mb', 0):.1f}" if r.get('peak_rss_mb', 0) > 0 else "N/A"
            f.write(f"| {r['grid_name']} | {r['n_nodes']} | {r['nproc_total']} | {wt} | {rss} | {r['status']} |\n")
        f.write('\n')

        # Graphs
        f.write('## Computation Time\n\n')
        f.write('![Computation Time](computation_time.png)\n\n')

        if len(NODE_COUNTS) > 1:
            f.write('## Scaling Analysis\n\n')
            f.write('![Scaling](scaling.png)\n\n')

        if any(r.get('peak_rss_mb', 0) > 0 for r in results):
            f.write('## Memory Consumption\n\n')
            f.write('![Memory](memory.png)\n\n')

        # T-field visualizations
        f.write('## Traveltime Field Visualizations\n\n')
        for npy_file in npy_files:
            f.write(f'### {npy_file.stem}\n\n')
            f.write(f'![T-field slices]({npy_file.stem}_slices.png)\n\n')

    print(f"  Report generated: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Benchmark TomoATT traveltime computation'
    )
    parser.add_argument(
        '--build-dir', default='build',
        help='Build directory relative to repo root (default: build)'
    )
    parser.add_argument(
        '--output-dir', default='gpu/benchmark_results',
        help='Output directory for results (default: gpu/benchmark_results)'
    )

    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
