#!/usr/bin/env python3

"""Benchmark scalar and SIMD TomoATT binaries on a direct-solver fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


RUNTIME_PATTERN = re.compile(r"It has run\s+([0-9eE+\-.]+)\s+sec")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SIMD efficiency against a scalar baseline")
    parser.add_argument("--scalar-bin", required=True, help="Path to the non-SIMD TOMOATT binary")
    parser.add_argument("--simd-bin", required=True, help="Path to the SIMD-enabled TOMOATT binary")
    parser.add_argument("--fixture", required=True, help="Benchmark fixture directory")
    parser.add_argument("--generator", default="make_benchmark_model.py", help="Fixture generator script")
    parser.add_argument("--input", default="input_params_benchmark.yaml", help="Benchmark input YAML file")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per binary")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.10,
        help="Maximum allowed SIMD slowdown fraction before failing (default: 0.10 = 10%%)",
    )
    parser.add_argument("--summary-path", help="Optional markdown summary output path")
    return parser.parse_args()


def cleanup_benchmark_outputs(fixture_dir: Path) -> None:
    for relative_path in (
        "OUTPUT_BENCHMARK",
        "src_rec_file_forward.dat",
        "two_layer_model_benchmark.h5",
        "src_rec_benchmark.dat",
        "time.txt",
    ):
        target = fixture_dir / relative_path
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()


def prepare_fixture(fixture_dir: Path, generator: str) -> None:
    cleanup_benchmark_outputs(fixture_dir)
    subprocess.run([sys.executable, generator], cwd=fixture_dir, check=True)


def extract_runtime(output: str, wall_seconds: float) -> float:
    match = RUNTIME_PATTERN.search(output)
    if match:
        return float(match.group(1))
    return wall_seconds


def run_variant(name: str, binary: Path, fixture_dir: Path, input_file: str, runs: int) -> list[float]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")
    env.setdefault("OMPI_MCA_rmaps_base_inherit", "1")
    env.setdefault("TERM", "xterm")

    results: list[float] = []
    for run_index in range(1, runs + 1):
        output_dir = fixture_dir / "OUTPUT_BENCHMARK"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        start = time.perf_counter()
        completed = subprocess.run(
            ["mpirun", "--oversubscribe", "-n", "1", str(binary), "-i", input_file],
            cwd=fixture_dir,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        wall_seconds = time.perf_counter() - start
        runtime = extract_runtime(completed.stdout + completed.stderr, wall_seconds)
        print(f"{name} run {run_index}: {runtime:.6f} s")
        results.append(runtime)

    return results


def build_summary(scalar_runs: list[float], simd_runs: list[float], max_regression: float) -> str:
    scalar_median = statistics.median(scalar_runs)
    simd_median = statistics.median(simd_runs)
    speedup = scalar_median / simd_median
    slowdown_limit = scalar_median * (1.0 + max_regression)
    status = "PASS" if simd_median <= slowdown_limit else "FAIL"

    return "\n".join(
        [
            "## SIMD Efficiency Benchmark",
            "",
            "| Variant | Runs (s) | Median (s) |",
            "| --- | --- | ---: |",
            f"| Scalar | {', '.join(f'{value:.6f}' for value in scalar_runs)} | {scalar_median:.6f} |",
            f"| SIMD | {', '.join(f'{value:.6f}' for value in simd_runs)} | {simd_median:.6f} |",
            "",
            f"- SIMD speedup: {speedup:.3f}x",
            f"- Allowed slowdown threshold: {max_regression * 100:.1f}%",
            f"- Result: {status}",
        ]
    )


def main() -> int:
    args = parse_args()

    fixture_dir = Path(args.fixture).resolve()
    scalar_bin = Path(args.scalar_bin).resolve()
    simd_bin = Path(args.simd_bin).resolve()

    if not fixture_dir.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")
    if not scalar_bin.is_file():
        raise FileNotFoundError(f"Scalar binary not found: {scalar_bin}")
    if not simd_bin.is_file():
        raise FileNotFoundError(f"SIMD binary not found: {simd_bin}")

    prepare_fixture(fixture_dir, args.generator)
    scalar_runs = run_variant("scalar", scalar_bin, fixture_dir, args.input, args.runs)
    simd_runs = run_variant("simd", simd_bin, fixture_dir, args.input, args.runs)

    summary = build_summary(scalar_runs, simd_runs, args.max_regression)
    print(summary)

    if args.summary_path:
        Path(args.summary_path).write_text(summary + "\n", encoding="utf-8")

    github_step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if github_step_summary:
        with open(github_step_summary, "a", encoding="utf-8") as summary_file:
            summary_file.write(summary + "\n")

    scalar_median = statistics.median(scalar_runs)
    simd_median = statistics.median(simd_runs)
    if simd_median > scalar_median * (1.0 + args.max_regression):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())