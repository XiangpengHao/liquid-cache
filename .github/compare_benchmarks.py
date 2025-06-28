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
        return f"{ms/1000:.2f}s"


def format_memory(bytes_val: int) -> str:
    """Format bytes into a human-readable string."""
    if bytes_val < 1024:
        return f"{bytes_val}B"
    elif bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f}KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val/1024**2:.1f}MB"
    else:
        return f"{bytes_val/1024**3:.2f}GB"


def get_avg_metrics(iteration_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate average metrics from iteration results."""
    if not iteration_results:
        return {"time_millis": 0, "liquid_cache_usage": 0}
    
    avg_time = sum(r["time_millis"] for r in iteration_results) / len(iteration_results)
    avg_memory = sum(r["liquid_cache_usage"] for r in iteration_results) / len(iteration_results)
    return {"time_millis": avg_time, "liquid_cache_usage": avg_memory}


def calculate_change(old_val: float, new_val: float) -> float:
    """Calculate percentage change between old and new values."""
    if old_val == 0:
        return float('inf') if new_val > 0 else 0
    return ((new_val - old_val) / old_val) * 100


def is_significant_change(change_pct: float, threshold: float = 10.0) -> bool:
    """Determine if a change is significant based on threshold."""
    return abs(change_pct) >= threshold


def load_benchmark_data(file_path: str) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Benchmark file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in benchmark file {file_path}: {e}")
        sys.exit(1)


def compare_benchmarks(current_file: str, baseline_file: str, threshold: float = 10.0) -> str:
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
    for i, (curr_query, baseline_query) in enumerate(zip(curr_results, baseline_results)):
        curr_metrics = get_avg_metrics(curr_query["iteration_results"])
        baseline_metrics = get_avg_metrics(baseline_query["iteration_results"])
        
        time_change = calculate_change(baseline_metrics["time_millis"], curr_metrics["time_millis"])
        memory_change = calculate_change(baseline_metrics["liquid_cache_usage"], curr_metrics["liquid_cache_usage"])
        
        comparison.append({
            "query": i + 1,
            "current_time": curr_metrics["time_millis"],
            "baseline_time": baseline_metrics["time_millis"],
            "time_change": time_change,
            "current_memory": curr_metrics["liquid_cache_usage"],
            "baseline_memory": baseline_metrics["liquid_cache_usage"],
            "memory_change": memory_change,
            "time_significant": is_significant_change(time_change, threshold),
            "memory_significant": is_significant_change(memory_change, threshold)
        })
    
    # Generate markdown report
    lines = []
    lines.append("## 📊 Benchmark Comparison")
    lines.append("")
    
    # Get commit info if available
    current_commit = current.get("commit", "unknown")[:8]
    baseline_commit = baseline.get("commit", "unknown")[:8]
    lines.append(f"**Current:** `{current_commit}` vs **Baseline:** `{baseline_commit}`")
    lines.append("")
    
    lines.append("| Query | Time (Current) | Time (Baseline) | Time Change | Memory (Current) | Memory (Baseline) | Memory Change |")
    lines.append("|-------|----------------|-----------------|-------------|------------------|-------------------|---------------|")
    
    for comp in comparison:
        time_change_str = f"{comp['time_change']:+.1f}%"
        memory_change_str = f"{comp['memory_change']:+.1f}%"
        
        # Bold significant changes
        if comp['time_significant']:
            time_change_str = f"**{time_change_str}**"
        if comp['memory_significant']:
            memory_change_str = f"**{memory_change_str}**"
        
        lines.append(f"| Q{comp['query']} | "
                    f"{format_time(comp['current_time'])} | "
                    f"{format_time(comp['baseline_time'])} | "
                    f"{time_change_str} | "
                    f"{format_memory(comp['current_memory'])} | "
                    f"{format_memory(comp['baseline_memory'])} | "
                    f"{memory_change_str} |")
    
    # Summary
    significant_changes = [c for c in comparison if c['time_significant'] or c['memory_significant']]
    lines.append("")
    if significant_changes:
        lines.append(f"**⚠️ {len(significant_changes)} queries have significant performance changes (≥{threshold}%)**")
        
        # List the most significant changes
        most_significant = sorted(significant_changes, 
                                key=lambda x: max(abs(x['time_change']), abs(x['memory_change'])), 
                                reverse=True)[:3]
        
        if most_significant:
            lines.append("")
            lines.append("**Most significant changes:**")
            for change in most_significant:
                change_desc = []
                if change['time_significant']:
                    change_desc.append(f"time {change['time_change']:+.1f}%")
                if change['memory_significant']:
                    change_desc.append(f"memory {change['memory_change']:+.1f}%")
                lines.append(f"- Q{change['query']}: {', '.join(change_desc)}")
    else:
        lines.append("✅ No significant performance regressions detected")
    
    lines.append("")
    lines.append(f"*Benchmark ran {len(curr_results)} queries with liquid-eager-transcode mode*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results and generate markdown report")
    parser.add_argument("current", help="Path to current benchmark results JSON file")
    parser.add_argument("baseline", help="Path to baseline benchmark results JSON file")
    parser.add_argument("--threshold", type=float, default=10.0, 
                       help="Percentage threshold for significant changes (default: 10.0)")
    parser.add_argument("--output", help="Output file for markdown report (default: stdout)")
    
    args = parser.parse_args()
    
    try:
        report = compare_benchmarks(args.current, args.baseline, args.threshold)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Benchmark comparison saved to {args.output}")
        else:
            print(report)
            
    except Exception as e:
        print(f"Error comparing benchmarks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()