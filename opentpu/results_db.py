from __future__ import annotations
import json
import os
import datetime
from typing import Dict, Any

RESULTS_FILE = "results/results.json"
MARKDOWN_FILE = "results/README.md"

def save_benchmark_run(run_data: Dict[str, Any], filepath: str = RESULTS_FILE) -> str:
    """Appends benchmark run results to database and updates markdown table."""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    runs = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                runs = json.load(f)
        except Exception:
            runs = []

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entry = {"timestamp": timestamp, **run_data}
    runs.append(entry)

    with open(filepath, "w") as f:
        json.dump(runs, f, indent=2)

    # Also update markdown summary in the same directory
    md_path = os.path.join(dir_path, "README.md") if dir_path else MARKDOWN_FILE
    update_results_markdown(runs, filepath=md_path)
    return filepath


def update_results_markdown(runs: list[Dict[str, Any]], filepath: str = MARKDOWN_FILE):
    """Generates a summary markdown table across benchmarked hardware."""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    lines = [
        "# OpenTPU-Benchmark Results Database",
        "",
        "Aggregated hardware performance across Google Cloud TPUs and development environments.",
        "",
        "| Date (UTC) | Device | Architecture | FP32 VPU | FP32 GEMM | BF16 MXU | INT8 MXU | HBM Read BW |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for run in runs[-20:]:  # Last 20 runs
        date_str = run.get("timestamp", "")[:10]
        dev_info = run.get("device", {})
        dev_name = dev_info.get("name", "Unknown")
        arch = dev_info.get("architecture", "")
        
        # Extract key metrics
        c_bench = {b["operation"].strip(): b["throughput_tflops_or_tiops"] for b in run.get("compute_benchmarks", [])}
        m_bench = {m["operation"].strip(): m["bandwidth_gb_per_sec"] for m in run.get("memory_benchmarks", [])}

        fp32_vpu = f"{c_bench.get('float , fma', 0.0):.2f} TF"
        fp32_gemm = f"{c_bench.get('float , gemm', 0.0):.1f} TF"
        bf16_mxu = f"{c_bench.get('bfloat, mxu', 0.0):.1f} TF"
        int8_mxu = f"{c_bench.get('int8  , mxu', 0.0):.1f} TOPs"
        hbm_read = f"{m_bench.get('coalesced read', 0.0):.1f} GB/s"

        lines.append(f"| {date_str} | {dev_name} | {arch} | {fp32_vpu} | {fp32_gemm} | {bf16_mxu} | {int8_mxu} | {hbm_read} |")

    lines.append("")
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
