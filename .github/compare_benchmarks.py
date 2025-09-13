#!/usr/bin/env python3
"""
Benchmark comparison script for liquid-cache CI.

Compares current benchmark results with a baseline (typically main branch)
and generates a markdown report highlighting significant performance changes.
"""

import json
import sys
import argparse
from typing import Dict, List, Any


def format_time(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1:
        return f"{ms:.3f}ms"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms / 1000:.2f}s"


def format_memory(bytes_val: int) -> str:
    """Format bytes into a human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f}KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / 1024**2:.1f}MB"
    else:
        return f"{bytes_val / 1024**3:.2f}GB"


def get_cold_metrics(iteration_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Get metrics from the first (cold) iteration."""
    if not iteration_results:
        return {"time_millis": 0, "cache_cpu_time": 0, "liquid_cache_usage": 0}
    
    first_result = iteration_results[0]
    return {
        "time_millis": first_result["time_millis"],
        "cache_cpu_time": first_result.get("cache_cpu_time", 0),
        "liquid_cache_usage": first_result["liquid_cache_usage"],
    }


def get_warm_metrics(iteration_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average metrics from warm iterations (excluding first)."""
    warm_results = iteration_results[1:] if len(iteration_results) > 1 else iteration_results
    if not warm_results:
        return {"time_millis": 0, "cache_cpu_time": 0, "liquid_cache_usage": 0}

    avg_time = sum(r["time_millis"] for r in warm_results) / len(warm_results)
    avg_cpu_time = sum(r.get("cache_cpu_time", 0) for r in warm_results) / len(warm_results)
    avg_memory = sum(r["liquid_cache_usage"] for r in warm_results) / len(warm_results)
    
    return {
        "time_millis": avg_time, 
        "cache_cpu_time": avg_cpu_time,
        "liquid_cache_usage": avg_memory
    }


def calculate_change(old_val: float, new_val: float) -> float:
    """Calculate percentage change between old and new values."""
    if old_val == 0:
        return float("inf") if new_val > 0 else 0
    return ((new_val - old_val) / old_val) * 100


def is_significant_change(change_pct: float, threshold: float = 10.0) -> bool:
    """Determine if a change is significant based on threshold."""
    return abs(change_pct) >= threshold


def format_metric_with_baseline(current: float, baseline: float, formatter_func) -> str:
    """Format metric with baseline in italics: current *(baseline)*."""
    return f"{formatter_func(current)} *({formatter_func(baseline)})*"


def format_change_percentage(current: float, baseline: float) -> str:
    """Format percentage change with bold for significant changes."""
    change_pct = calculate_change(baseline, current)
    
    if abs(change_pct) >= 15.0:  # Significant change threshold
        return f"**{change_pct:+.1f}%**"
    else:
        return f"{change_pct:+.1f}%"


def load_benchmark_data(file_path: str) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in benchmark file {file_path}: {e}")
        sys.exit(1)


def compare_benchmarks(
    current_file: str, baseline_file: str, threshold: float = 10.0
) -> str:
    """
    Compare current benchmark with baseline and generate markdown report.

    Args:
        current_file: Path to current benchmark results
        baseline_file: Path to baseline benchmark results
        threshold: Percentage threshold for significant changes

    Returns:
        Markdown formatted comparison report
    """
    current = load_benchmark_data(current_file)
    baseline = load_benchmark_data(baseline_file)

    comparison = []

    curr_results = current["results"]
    baseline_results = baseline["results"]

    # Compare each query
    for i, (curr_query, baseline_query) in enumerate(
        zip(curr_results, baseline_results)
    ):
        curr_cold = get_cold_metrics(curr_query["iteration_results"])
        baseline_cold = get_cold_metrics(baseline_query["iteration_results"])
        curr_warm = get_warm_metrics(curr_query["iteration_results"])
        baseline_warm = get_warm_metrics(baseline_query["iteration_results"])

        cold_time_change = calculate_change(baseline_cold["time_millis"], curr_cold["time_millis"])
        warm_time_change = calculate_change(baseline_warm["time_millis"], curr_warm["time_millis"])
        cpu_time_change = calculate_change(baseline_warm["cache_cpu_time"], curr_warm["cache_cpu_time"])
        memory_change = calculate_change(baseline_warm["liquid_cache_usage"], curr_warm["liquid_cache_usage"])

        comparison.append(
            {
                "query": i + 1,
                "curr_cold_time": curr_cold["time_millis"],
                "baseline_cold_time": baseline_cold["time_millis"],
                "cold_time_change": cold_time_change,
                "curr_warm_time": curr_warm["time_millis"],
                "baseline_warm_time": baseline_warm["time_millis"],
                "warm_time_change": warm_time_change,
                "curr_cpu_time": curr_warm["cache_cpu_time"],
                "baseline_cpu_time": baseline_warm["cache_cpu_time"],
                "cpu_time_change": cpu_time_change,
                "curr_memory": curr_warm["liquid_cache_usage"],
                "baseline_memory": baseline_warm["liquid_cache_usage"],
                "memory_change": memory_change,
                "cold_time_significant": is_significant_change(cold_time_change, threshold),
                "warm_time_significant": is_significant_change(warm_time_change, threshold),
                "cpu_time_significant": is_significant_change(cpu_time_change, threshold),
                "memory_significant": is_significant_change(memory_change, threshold),
            }
        )

    # Determine modes for clearer labeling
    def extract_mode(d: Dict[str, Any]) -> str:
        try:
            # in_process encodes args.bench_mode as enum variant string
            return d.get("args", {}).get("bench_mode", "unknown")
        except Exception:
            return "unknown"

    current_mode = extract_mode(current)
    baseline_mode = extract_mode(baseline)

    # Generate markdown report
    lines = []
    lines.append("## 📊 Benchmark Comparison")
    lines.append("")

    # Get commit info if available
    current_commit = current.get("commit", "unknown")[:8]
    baseline_commit = baseline.get("commit", "unknown")[:8]
    lines.append(
        f"**Current:** `{current_commit}` ({current_mode}) vs **Baseline:** `{baseline_commit}` ({baseline_mode})"
    )
    lines.append("")

    lines.append(
        "| Query | Cold Time | Δ | Warm Time | Δ | CPU Time | Δ | Memory | Δ |"
    )
    lines.append(
        "|-------|-----------|---|-----------|---|----------|---|--------|---|"
    )

    for comp in comparison:
        cold_time_str = format_metric_with_baseline(
            comp['curr_cold_time'], comp['baseline_cold_time'], format_time
        )
        cold_change_str = format_change_percentage(
            comp['curr_cold_time'], comp['baseline_cold_time']
        )
        
        warm_time_str = format_metric_with_baseline(
            comp['curr_warm_time'], comp['baseline_warm_time'], format_time
        )
        warm_change_str = format_change_percentage(
            comp['curr_warm_time'], comp['baseline_warm_time']
        )
        
        cpu_time_str = format_metric_with_baseline(
            comp['curr_cpu_time'], comp['baseline_cpu_time'], format_time
        )
        cpu_change_str = format_change_percentage(
            comp['curr_cpu_time'], comp['baseline_cpu_time']
        )
        
        memory_str = format_metric_with_baseline(
            comp['curr_memory'], comp['baseline_memory'], format_memory
        )
        memory_change_str = format_change_percentage(
            comp['curr_memory'], comp['baseline_memory']
        )

        lines.append(
            f"| Q{comp['query']} | "
            f"{cold_time_str} | {cold_change_str} | "
            f"{warm_time_str} | {warm_change_str} | "
            f"{cpu_time_str} | {cpu_change_str} | "
            f"{memory_str} | {memory_change_str} |"
        )

    # Summary
    significant_changes = [
        c for c in comparison 
        if c["cold_time_significant"] or c["warm_time_significant"] 
        or c["cpu_time_significant"] or c["memory_significant"]
    ]
    lines.append("")
    if significant_changes:
        lines.append(
            f"**⚠️ {len(significant_changes)} queries have significant performance changes (≥{threshold}%)**"
        )

        # List the most significant changes
        most_significant = sorted(
            significant_changes,
            key=lambda x: max(
                abs(x["cold_time_change"]), 
                abs(x["warm_time_change"]),
                abs(x["cpu_time_change"]), 
                abs(x["memory_change"])
            ),
            reverse=True,
        )[:3]

        if most_significant:
            lines.append("")
            lines.append("**Most significant changes:**")
            for change in most_significant:
                change_desc = []
                if change["cold_time_significant"]:
                    change_desc.append(f"cold time {change['cold_time_change']:+.1f}%")
                if change["warm_time_significant"]:
                    change_desc.append(f"warm time {change['warm_time_change']:+.1f}%")
                if change["cpu_time_significant"]:
                    change_desc.append(f"CPU time {change['cpu_time_change']:+.1f}%")
                if change["memory_significant"]:
                    change_desc.append(f"memory {change['memory_change']:+.1f}%")
                lines.append(f"- Q{change['query']}: {', '.join(change_desc)}")
    else:
        lines.append("✅ No significant performance regressions detected")

    lines.append("")
    lines.append(f"*Compared {current_mode} vs {baseline_mode} on the same runner*")
    lines.append("*Cold Time: first iteration; Warm Time: average of remaining iterations.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results and generate markdown report"
    )
    parser.add_argument("current", help="Path to current benchmark results JSON file")
    parser.add_argument("baseline", help="Path to baseline benchmark results JSON file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="Percentage threshold for significant changes (default: 15.0)",
    )
    parser.add_argument(
        "--output", help="Output file for markdown report (default: stdout)"
    )

    args = parser.parse_args()

    try:
        report = compare_benchmarks(args.current, args.baseline, args.threshold)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Benchmark comparison saved to {args.output}")
        else:
            print(report)

    except Exception as e:
        print(f"Error comparing benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
