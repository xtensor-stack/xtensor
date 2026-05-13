#!/usr/bin/env python3

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


NAME_PATTERN = re.compile(r"^(?P<operation>.+)_(?P<implementation>xtensor|numpy)/(?P<size>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run xtensor/NumPy math and reducer benchmarks and generate a Markdown comparison report."
    )
    parser.add_argument(
        "--benchmark-exe",
        type=Path,
        default=Path("build/benchmark/benchmark_xtensor"),
        help="Path to the benchmark executable when running benchmarks.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Existing Google Benchmark JSON output to analyze instead of running the executable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file path to write the Markdown report to.",
    )
    parser.add_argument(
        "--benchmark-filter",
        default=".*_(xtensor|numpy)/.*",
        help="Google Benchmark filter used when running the executable.",
    )
    parser.add_argument(
        "--benchmark-min-time",
        default="0.05s",
        help="Minimum benchmark runtime passed through to Google Benchmark.",
    )
    parser.add_argument(
        "--metric",
        choices=("cpu_time", "real_time"),
        default="cpu_time",
        help="Benchmark metric to compare in the report.",
    )
    return parser.parse_args()


def run_benchmarks(benchmark_exe: Path, benchmark_filter: str, benchmark_min_time: str) -> dict:
    if not benchmark_exe.exists():
        raise FileNotFoundError(f"Benchmark executable not found: {benchmark_exe}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    command = [
        str(benchmark_exe),
        f"--benchmark_filter={benchmark_filter}",
        f"--benchmark_min_time={benchmark_min_time}",
        f"--benchmark_out={output_path}",
        "--benchmark_out_format=json",
    ]

    try:
        subprocess.run(command, check=True)
        return json.loads(output_path.read_text())
    finally:
        output_path.unlink(missing_ok=True)


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def numeric_sort_key(size: str):
    try:
        return (0, int(size))
    except ValueError:
        return (1, size)


def collect_pairs(data: dict, metric: str) -> tuple[dict, list[dict]]:
    paired: dict[tuple[str, str], dict[str, float]] = {}

    for benchmark in data.get("benchmarks", []):
        if benchmark.get("run_type") != "iteration":
            continue
        match = NAME_PATTERN.match(benchmark.get("name", ""))
        if match is None:
            continue
        key = (match.group("operation"), match.group("size"))
        paired.setdefault(key, {})[match.group("implementation")] = benchmark[metric]

    rows = []
    for (operation, size), values in paired.items():
        xtensor_time = values.get("xtensor")
        numpy_time = values.get("numpy")
        if xtensor_time is None or numpy_time is None:
            continue
        ratio = numpy_time / xtensor_time if xtensor_time else math.inf
        rows.append(
            {
                "operation": operation,
                "size": size,
                "xtensor": xtensor_time,
                "numpy": numpy_time,
                "ratio": ratio,
            }
        )

    rows.sort(key=lambda row: (row["operation"], numeric_sort_key(row["size"])))

    summaries: dict[str, list[float]] = {}
    for row in rows:
        summaries.setdefault(row["operation"], []).append(row["ratio"])

    per_operation = {
        operation: {
            "geomean_ratio": statistics.geometric_mean(ratios),
            "best_ratio": min(ratios),
            "worst_ratio": max(ratios),
            "samples": len(ratios),
        }
        for operation, ratios in summaries.items()
    }
    return per_operation, rows


def format_ratio(ratio: float) -> str:
    if not math.isfinite(ratio):
        return "inf"
    return f"{ratio:.2f}x"


def format_time(value: float) -> str:
    return f"{value:.0f} ns"


def classify_ratio(ratio: float) -> str:
    if ratio > 1.05:
        return "xtensor faster"
    if ratio < 0.95:
        return "numpy faster"
    return "similar"


def render_report(data: dict, metric: str) -> str:
    per_operation, rows = collect_pairs(data, metric)
    context = data.get("context", {})

    lines = []
    lines.append("# xtensor vs NumPy Benchmark Report")
    lines.append("")
    lines.append(f"Metric: `{metric}`")
    if context:
        lines.append(f"Executable: `{context.get('executable', 'unknown')}`")
        lines.append(f"Host: `{context.get('host_name', 'unknown')}`")
        lines.append(f"CPU scaling enabled: `{context.get('cpu_scaling_enabled', 'unknown')}`")
        lines.append(f"ASLR enabled: `{context.get('aslr_enabled', 'unknown')}`")
    lines.append("")
    lines.append("## Per-function summary")
    lines.append("")
    lines.append("| Function | Samples | NumPy / xtensor geomean | Best | Worst | Verdict |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for operation in sorted(per_operation):
        summary = per_operation[operation]
        lines.append(
            f"| {operation} | {summary['samples']} | {format_ratio(summary['geomean_ratio'])} | "
            f"{format_ratio(summary['best_ratio'])} | {format_ratio(summary['worst_ratio'])} | {classify_ratio(summary['geomean_ratio'])} |"
        )

    lines.append("")
    lines.append("## Detailed results")
    lines.append("")
    lines.append("| Function | Size | xtensor | NumPy | NumPy / xtensor | Verdict |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            f"| {row['operation']} | {row['size']} | {format_time(row['xtensor'])} | {format_time(row['numpy'])} | "
            f"{format_ratio(row['ratio'])} | {classify_ratio(row['ratio'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.input_json is not None:
        data = load_results(args.input_json)
    else:
        data = run_benchmarks(args.benchmark_exe, args.benchmark_filter, args.benchmark_min_time)

    report = render_report(data, args.metric)
    if args.output is not None:
        args.output.write_text(report)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())