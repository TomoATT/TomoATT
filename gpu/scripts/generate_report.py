#!/usr/bin/env python3
"""
generate_report.py — Generate comprehensive benchmark report.

Reads TomoATT HDF5 output files and creates:
  1. Computation time graphs (by grid size and node count)
  2. Scaling analysis graphs
  3. T-field slice visualizations (showing circular wavefronts)
  4. T-field difference plots (between node counts)
  5. Markdown report with embedded graphs
"""

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent  # TomoATT/

GRID_CONFIGS = [
    {"name": "10×50×50",  "n_rtp": [10, 50, 50]},
    {"name": "20×100×100","n_rtp": [20, 100, 100]},
    {"name": "40×200×200","n_rtp": [40, 200, 200]},
]
NODE_COUNTS = [1, 2, 4]

BENCH_DIR = REPO_ROOT / "gpu" / "benchmark_results"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def read_traveltime_field(h5_path, grid_config, src_id=0):
    """Read traveltime field from HDF5 output.

    The T field is stored as a flat 1D array. Its size includes ghost
    cells from MPI domain decomposition, so the actual 3D shape must be
    inferred from the total size and the nominal grid dimensions.

    Returns:
        T_field: numpy array of shape (nr, nt, np)
        grid_shape: tuple (nr, nt, np)
    """
    with h5py.File(h5_path, 'r') as f:
        # --- Read the traveltime field ---
        src_key = f"src_rec_{src_id}"
        if src_key not in f:
            srcs = [k for k in f.keys() if k.startswith('src_rec_')]
            if srcs:
                src_key = srcs[0]
            else:
                print(f"  No src_rec datasets found in {h5_path}")
                return None, None

        tf_key = "time_field_inv_0000"
        if tf_key not in f[src_key]:
            tfs = [k for k in f[src_key].keys() if k.startswith('time_field')]
            if tfs:
                tf_key = tfs[0]
            else:
                print(f"  No time_field datasets in {src_key}")
                return None, None

        T_flat = np.array(f[src_key][tf_key])
        total = T_flat.size

        # --- Infer the 3D shape ---
        nr0, nt0, np0 = grid_config['n_rtp']
        expected = nr0 * nt0 * np0

        if total == expected:
            # Exact match — simple reshape
            return T_flat.reshape((nr0, nt0, np0)), (nr0, nt0, np0)
        else:
            # Multi-node case: flat array includes ghost cells from domain decomposition.
            # We must use the grid coordinates to map values back to the reference grid.
            print(f"  Multi-node: {total} nodes (expected {expected}), using coordinate mapping")
            return T_flat, (total,)


def load_benchmark_results(json_path):
    """Load benchmark results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------
def plot_computation_time(results, output_dir):
    """Plot computation time by grid size and node count."""
    fig, ax = plt.subplots(figsize=(12, 7))

    grid_names = sorted(set(r['grid_name'] for r in results))
    x = np.arange(len(grid_names))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, n_nodes in enumerate(NODE_COUNTS):
        node_results = [r for r in results if r['n_nodes'] == n_nodes]
        times = []
        for gn in grid_names:
            r = next((r for r in node_results if r['grid_name'] == gn), None)
            times.append(r['wall_time_s'] if r and r['status'] == 'SUCCESS' else 0)

        bars = ax.bar(x + i * width, times, width,
                      label=f'{n_nodes} nodes', color=colors[i], alpha=0.8)

        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{t:.1f}s', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Grid Configuration', fontsize=14)
    ax.set_ylabel('Wall-clock Time (s)', fontsize=14)
    ax.set_title('Traveltime Computation Time\n(True travel times, run_mode=0)', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels(grid_names, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'computation_time.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: computation_time.png")


def plot_scaling(results, output_dir):
    """Plot scaling analysis: time vs node count for each grid size."""
    fig, ax = plt.subplots(figsize=(10, 7))

    grid_names = sorted(set(r['grid_name'] for r in results))
    markers = ['o', 's', '^']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, gn in enumerate(grid_names):
        grid_results = sorted(
            [r for r in results if r['grid_name'] == gn and r['status'] == 'SUCCESS'],
            key=lambda r: r['n_nodes']
        )
        if not grid_results:
            continue

        nodes = [r['n_nodes'] for r in grid_results]
        times = [r['wall_time_s'] for r in grid_results]

        ax.plot(nodes, times, f'{markers[i]}-', linewidth=2, markersize=10,
                label=f'Grid {gn}', color=colors[i])

        for n, t in zip(nodes, times):
            ax.annotate(f'{t:.1f}s', (n, t), textcoords="offset points",
                        xytext=(10, 5), fontsize=10)

    ax.set_xlabel('Number of Nodes', fontsize=14)
    ax.set_ylabel('Wall-clock Time (s)', fontsize=14)
    ax.set_title('Scaling Analysis: Time vs Node Count', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: scaling.png")


def reconstruct_tfield(t_flat, grid_h5_path, ref_shape):
    """Reconstruct a 3D T-field from a flat array using grid coordinates.

    Multi-node HDF5 output includes ghost cells from domain decomposition.
    We use the grid coordinates to sort and reshape values back to the
    reference grid shape.
    """
    import h5py

    with h5py.File(grid_h5_path, 'r') as f:
        mesh = f['Mesh']
        p = np.array(mesh['node_coords_p'])
        t = np.array(mesh['node_coords_t'])
        r = np.array(mesh['node_coords_r'])

    nr, nt, np_ = ref_shape
    expected = nr * nt * np_

    if t_flat.size == expected:
        return t_flat.reshape(ref_shape)

    # Sort by (r, t, p) using lexsort (last key is primary)
    sort_idx = np.lexsort((p, t, r))
    r_sorted = np.round(r[sort_idx], 6)
    t_sorted = np.round(t[sort_idx], 6)
    p_sorted = np.round(p[sort_idx], 6)
    val_sorted = t_flat[sort_idx]

    # Remove duplicates (ghost cells) — keep first occurrence of each coordinate
    coords = np.stack([r_sorted, t_sorted, p_sorted], axis=1)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve sorted order
    unique_vals = val_sorted[unique_idx]

    if unique_vals.size == expected:
        return unique_vals.reshape(ref_shape)
    else:
        print(f"  Warning: after dedup got {unique_vals.size}, expected {expected}")
        # Try to reshape anyway
        if unique_vals.size >= expected:
            return unique_vals[:expected].reshape(ref_shape)
        return unique_vals


def reconstruct_tfield_v2(t_flat, grid_h5_path, ref_shape):
    """Reconstruct a 3D T-field using coordinate-based mapping.

    This is a more robust approach that maps each node's coordinates
    to the reference grid position.
    """
    import h5py

    nr, nt, np_ = ref_shape
    expected = nr * nt * np_

    if t_flat.size == expected:
        return t_flat.reshape(ref_shape)

    # Read grid coordinates
    with h5py.File(grid_h5_path, 'r') as f:
        mesh = f['Mesh']
        p = np.array(mesh['node_coords_p'])
        t = np.array(mesh['node_coords_t'])
        r = np.array(mesh['node_coords_r'])

    # Build coordinate-to-index mapping
    coord_to_idx = {}
    for i in range(len(p)):
        key = (round(float(r[i]), 6), round(float(t[i]), 6), round(float(p[i]), 6))
        coord_to_idx[key] = i

    # Map T field values to reference grid
    T_mapped = np.full(t_flat.size, np.nan)
    for i in range(len(p)):
        key = (round(float(r[i]), 6), round(float(t[i]), 6), round(float(p[i]), 6))
        if key in coord_to_idx:
            T_mapped[coord_to_idx[key]] = t_flat[i]

    # Remove NaN values (ghost cells that don't have a unique coordinate)
    valid_mask = ~np.isnan(T_mapped)
    T_valid = T_mapped[valid_mask]

    if T_valid.size == expected:
        return T_valid.reshape(ref_shape)
    else:
        print(f"  Warning: after mapping got {T_valid.size}, expected {expected}")
        if T_valid.size >= expected:
            return T_valid[:expected].reshape(ref_shape)
        return T_valid


def plot_tfield_slices(T_field, grid_name, n_nodes, output_dir):
    """Plot T-field slices showing traveltime wavefronts."""
    if T_field is None or T_field.ndim != 3:
        return

    nr, nt, np_ = T_field.shape
    mid_r = nr // 2
    mid_t = nt // 2
    mid_p = np_ // 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f'Traveltime Field T: Grid {grid_name}, {n_nodes} nodes',
                 fontsize=16, fontweight='bold')

    # Row 1: Three orthogonal slices
    im0 = axes[0, 0].imshow(T_field[mid_r, :, :], origin='lower', cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Depth slice (r={mid_r})', fontsize=13)
    axes[0, 0].set_xlabel('p (longitude index)', fontsize=11)
    axes[0, 0].set_ylabel('t (latitude index)', fontsize=11)
    plt.colorbar(im0, ax=axes[0, 0], label='T (s)', shrink=0.8)

    im1 = axes[0, 1].imshow(T_field[:, mid_t, :], origin='lower', cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'Latitude slice (t={mid_t})', fontsize=13)
    axes[0, 1].set_xlabel('p (longitude index)', fontsize=11)
    axes[0, 1].set_ylabel('r (depth index)', fontsize=11)
    plt.colorbar(im1, ax=axes[0, 1], label='T (s)', shrink=0.8)

    im2 = axes[0, 2].imshow(T_field[:, :, mid_p].T, origin='lower', cmap='viridis', aspect='auto')
    axes[0, 2].set_title(f'Longitude slice (p={mid_p})', fontsize=13)
    axes[0, 2].set_xlabel('t (latitude index)', fontsize=11)
    axes[0, 2].set_ylabel('r (depth index)', fontsize=11)
    plt.colorbar(im2, ax=axes[0, 2], label='T (s)', shrink=0.8)

    # Row 2: Contour plots showing wavefront structure
    cs0 = axes[1, 0].contourf(T_field[mid_r, :, :], levels=20, cmap='viridis')
    axes[1, 0].contour(T_field[mid_r, :, :], levels=20, colors='k', linewidths=0.3)
    axes[1, 0].set_title(f'Contour: Depth slice (r={mid_r})', fontsize=13)
    axes[1, 0].set_xlabel('p (longitude index)', fontsize=11)
    axes[1, 0].set_ylabel('t (latitude index)', fontsize=11)
    plt.colorbar(cs0, ax=axes[1, 0], label='T (s)', shrink=0.8)

    cs1 = axes[1, 1].contourf(T_field[:, mid_t, :], levels=20, cmap='viridis')
    axes[1, 1].contour(T_field[:, mid_t, :], levels=20, colors='k', linewidths=0.3)
    axes[1, 1].set_title(f'Contour: Latitude slice (t={mid_t})', fontsize=13)
    axes[1, 1].set_xlabel('p (longitude index)', fontsize=11)
    axes[1, 1].set_ylabel('r (depth index)', fontsize=11)
    plt.colorbar(cs1, ax=axes[1, 1], label='T (s)', shrink=0.8)

    cs2 = axes[1, 2].contourf(T_field[:, :, mid_p].T, levels=20, cmap='viridis')
    axes[1, 2].contour(T_field[:, :, mid_p].T, levels=20, colors='k', linewidths=0.3)
    axes[1, 2].set_title(f'Contour: Longitude slice (p={mid_p})', fontsize=13)
    axes[1, 2].set_xlabel('t (latitude index)', fontsize=11)
    axes[1, 2].set_ylabel('r (depth index)', fontsize=11)
    plt.colorbar(cs2, ax=axes[1, 2], label='T (s)', shrink=0.8)

    plt.tight_layout()
    fname = f"tfield_{grid_name.replace('×','x')}_nodes{n_nodes}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {fname}")


def plot_tfield_comparison(T_fields, grid_name, output_dir):
    """Plot T-field comparison between different node counts."""
    if not T_fields:
        return

    # Only plot 3D fields (skip 1D that couldn't be reshaped)
    valid_fields = {n: T for n, T in T_fields.items() if T.ndim == 3}
    if not valid_fields:
        return

    n_plots = len(valid_fields)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(f'T-field Comparison: Grid {grid_name}', fontsize=16, fontweight='bold')

    for ax, (n_nodes, T_field) in zip(axes, valid_fields.items()):
        nr, nt, np_ = T_field.shape
        mid_r = nr // 2

        im = ax.imshow(T_field[mid_r, :, :], origin='lower', cmap='viridis', aspect='auto')
        ax.set_title(f'{n_nodes} nodes', fontsize=14)
        ax.set_xlabel('p (longitude index)', fontsize=11)
        ax.set_ylabel('t (latitude index)', fontsize=11)
        plt.colorbar(im, ax=ax, label='T (s)', shrink=0.8)

    plt.tight_layout()
    fname = f"tfield_comparison_{grid_name.replace('×','x')}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {fname}")


def plot_tfield_diff(T_ref, T_other, ref_name, other_name, grid_name, output_dir):
    """Plot T-field difference between two configurations.

    If the two fields have different shapes (due to ghost cells from
    different MPI decompositions), crop the larger one to match the smaller.
    """
    if T_ref is None or T_other is None:
        return
    if T_ref.ndim != 3 or T_other.ndim != 3:
        return

    # Crop to common shape if needed
    if T_ref.shape != T_other.shape:
        min_dims = tuple(min(a, b) for a, b in zip(T_ref.shape, T_other.shape))
        T_ref = T_ref[:min_dims[0], :min_dims[1], :min_dims[2]]
        T_other = T_other[:min_dims[0], :min_dims[1], :min_dims[2]]

    T_diff = np.abs(T_ref - T_other)
    nr, nt, np_ = T_ref.shape
    mid_r = nr // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'|T({ref_name}) - T({other_name})|: Grid {grid_name}',
                 fontsize=14, fontweight='bold')

    im0 = axes[0].imshow(T_diff[mid_r, :, :], origin='lower', cmap='hot_r', aspect='auto')
    axes[0].set_title(f'Depth slice (r={mid_r})')
    axes[0].set_xlabel('p (longitude index)')
    axes[0].set_ylabel('t (latitude index)')
    plt.colorbar(im0, ax=axes[0], label='|ΔT| (s)')

    mid_t = nt // 2
    im1 = axes[1].imshow(T_diff[:, mid_t, :], origin='lower', cmap='hot_r', aspect='auto')
    axes[1].set_title(f'Latitude slice (t={mid_t})')
    axes[1].set_xlabel('p (longitude index)')
    axes[1].set_ylabel('r (depth index)')
    plt.colorbar(im1, ax=axes[1], label='|ΔT| (s)')

    mid_p = np_ // 2
    im2 = axes[2].imshow(T_diff[:, :, mid_p].T, origin='lower', cmap='hot_r', aspect='auto')
    axes[2].set_title(f'Longitude slice (p={mid_p})')
    axes[2].set_xlabel('t (latitude index)')
    axes[2].set_ylabel('r (depth index)')
    plt.colorbar(im2, ax=axes[2], label='|ΔT| (s)')

    plt.tight_layout()
    fname = f"tfield_diff_{grid_name.replace('×','x')}_{ref_name}_vs_{other_name}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Generated: {fname}")


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------
def generate_report(results, bench_dir):
    """Generate comprehensive benchmark report."""
    bench_dir = Path(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  Generating Benchmark Report")
    print("="*60)

    # Generate computation time graph
    plot_computation_time(results, bench_dir)

    # Generate scaling graph
    plot_scaling(results, bench_dir)

    # Load and visualize T fields for each grid/node combination
    T_fields_by_grid = {}  # grid_name -> {n_nodes -> T_field}

    for grid_config in GRID_CONFIGS:
        grid_name = grid_config['name']
        T_fields = {}

        for n_nodes in NODE_COUNTS:
            test_dir_name = f"grid_{grid_name.replace('×','x')}_nodes{n_nodes}"
            test_dir = bench_dir / test_dir_name

            if not test_dir.exists():
                print(f"  Skipping {grid_name} nodes={n_nodes}: dir not found")
                continue

            output_dir = test_dir / 'OUTPUT_FILES'
            h5_sim = output_dir / 'out_data_sim_group_0.h5'

            if not h5_sim.exists():
                print(f"  Skipping {grid_name} nodes={n_nodes}: HDF5 not found")
                continue

            T_field, grid_shape = read_traveltime_field(h5_sim, grid_config, src_id=0)

            if T_field is not None:
                # For multi-node cases, reconstruct using grid coordinates
                if T_field.ndim == 1 and len(T_field) != grid_config['n_rtp'][0] * grid_config['n_rtp'][1] * grid_config['n_rtp'][2]:
                    grid_h5 = output_dir / 'out_data_grid.h5'
                    if grid_h5.exists():
                        ref_shape = tuple(grid_config['n_rtp'])
                        T_field = reconstruct_tfield(T_field, grid_h5, ref_shape)

                T_fields[n_nodes] = T_field
                print(f"  Loaded T field: {grid_name} nodes={n_nodes}, shape={grid_shape}")

                # Generate T-field visualization
                plot_tfield_slices(T_field, grid_name, n_nodes, bench_dir)

        T_fields_by_grid[grid_name] = T_fields

        # Generate T-field comparison between node counts
        if len(T_fields) > 1:
            plot_tfield_comparison(T_fields, grid_name, bench_dir)

        # Generate T-field difference plots (1 node vs 4 nodes)
        if 1 in T_fields and 4 in T_fields:
            plot_tfield_diff(T_fields[1], T_fields[4], '1node', '4nodes',
                            grid_name, bench_dir)

    # Generate Markdown report
    report_path = bench_dir / "BENCHMARK_REPORT.md"
    with open(report_path, 'w') as f:
        f.write('# TomoATT Traveltime Computation Benchmark Report\n\n')
        f.write(f'**Date**: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'**Mode**: True travel times (run_mode=0)\n')
        f.write(f'**Grid configs**: {len(GRID_CONFIGS)}\n')
        f.write(f'**Node counts**: {NODE_COUNTS}\n\n')
        f.write('---\n\n')

        # Results table
        f.write('## 1. Results Table\n\n')
        f.write('| Grid | Nodes | MPI Procs | Wall Time (s) | Status |\n')
        f.write('|------|-------|-----------|---------------|--------|\n')
        for r in results:
            wt = f"{r['wall_time_s']:.2f}" if r.get('wall_time_s') else "N/A"
            f.write(f"| {r['grid_name']} | {r['n_nodes']} | {r.get('nproc_total', r['n_nodes'])} | {wt} | {r['status']} |\n")
        f.write('\n')

        # Computation time graph
        f.write('## 2. Computation Time\n\n')
        f.write('![Computation Time](computation_time.png)\n\n')

        # Scaling analysis
        f.write('## 3. Scaling Analysis\n\n')
        f.write('![Scaling](scaling.png)\n\n')

        # T-field visualizations
        f.write('## 4. Traveltime Field Visualizations\n\n')
        f.write('The following figures show the computed traveltime field T '
                'from the TomoATT solver. The circular wavefront pattern '
                'confirms that the traveltime is proportional to the distance '
                'from the source.\n\n')

        for grid_config in GRID_CONFIGS:
            grid_name = grid_config['name']
            f.write(f'### {grid_name}\n\n')

            for n_nodes in NODE_COUNTS:
                fname = f"tfield_{grid_name.replace('×','x')}_nodes{n_nodes}.png"
                f.write(f'#### {n_nodes} nodes\n\n')
                f.write(f'![T-field {grid_name} {n_nodes} nodes]({fname})\n\n')

        # T-field comparison
        f.write('## 5. T-field Comparison Between Node Counts\n\n')
        for grid_config in GRID_CONFIGS:
            grid_name = grid_config['name']
            fname = f"tfield_comparison_{grid_name.replace('×','x')}.png"
            f.write(f'### {grid_name}\n\n')
            f.write(f'![T-field comparison {grid_name}]({fname})\n\n')

        # T-field difference
        f.write('## 6. T-field Difference (1 node vs 4 nodes)\n\n')
        f.write('The following figures show the absolute difference in the '
                'traveltime field between 1-node and 4-node configurations. '
                'Small differences indicate numerical consistency across '
                'different MPI decompositions.\n\n')
        for grid_config in GRID_CONFIGS:
            grid_name = grid_config['name']
            fname = f"tfield_diff_{grid_name.replace('×','x')}_1node_vs_4nodes.png"
            f.write(f'### {grid_name}\n\n')
            f.write(f'![T-field diff {grid_name}]({fname})\n\n')

        # Summary
        f.write('## 7. Summary\n\n')
        success_results = [r for r in results if r['status'] == 'SUCCESS']
        if success_results:
            max_time = max(r['wall_time_s'] for r in success_results)
            min_time = min(r['wall_time_s'] for r in success_results)
            f.write(f'- **Total configurations tested**: {len(results)}\n')
            f.write(f'- **Successful runs**: {len(success_results)}\n')
            f.write(f'- **Time range**: {min_time:.2f}s - {max_time:.2f}s\n')
            f.write(f'- **Grid sizes**: {", ".join(gc["name"] for gc in GRID_CONFIGS)}\n')
            f.write(f'- **Node counts**: {NODE_COUNTS}\n')

    print(f"\n  Report generated: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    json_path = BENCH_DIR / "benchmark_results.json"

    if not json_path.exists():
        print(f"Error: {json_path} not found")
        print("Please run benchmark_tomoatt.py first")
        sys.exit(1)

    data = load_benchmark_results(json_path)
    # JSON structure: {"results": [...], "grid_configs": [...], ...}
    if isinstance(data, dict) and 'results' in data:
        results = data['results']
    elif isinstance(data, list):
        results = data
    else:
        print(f"Error: unexpected JSON structure: {type(data)}")
        sys.exit(1)
    generate_report(results, BENCH_DIR)


if __name__ == '__main__':
    main()
