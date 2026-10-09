from __future__ import annotations
import argparse
import json
import sys
from opentpu.device_info import get_platform_info
from opentpu.benchmarks.fp32 import run_fp32_vector_benchmark, run_fp32_gemm_benchmark
from opentpu.reporter import print_banner, print_result_row, print_footer, format_cell

def main():
    parser = argparse.ArgumentParser(description="OpenTPU-Benchmark Suite")
    parser.add_argument("--test", choices=["all", "fp32_vector", "fp32_gemm"], default="all",
                        help="Which FP32 benchmark test to run (default: all)")
    parser.add_argument("--elements", type=int, default=4 * 1024 * 1024,
                        help="Number of float32 elements for Vector ALU benchmark (default: 4194304)")
    parser.add_argument("--matrix-dim", type=int, default=4096,
                        help="Matrix dimension for GEMM benchmark (default: 4096)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of benchmark iterations (default: 20)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations for JIT compile (default: 3)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Device index to run on (default: 0)")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format for automated agent evaluation")
    args = parser.parse_args()

    info = get_platform_info()
    if args.device_id >= len(info.devices):
        print(f"Error: Device ID {args.device_id} invalid. Total available devices: {len(info.devices)}", file=sys.stderr)
        sys.exit(1)

    target_device = info.devices[args.device_id]
    results = []

    if not args.json:
        print_banner(info)
        print(format_cell("JIT compiling and warming up kernels..."))

    # FP32 Vector ALU Benchmark
    if args.test in ["all", "fp32_vector"]:
        if not args.json:
            print(format_cell("Running FP32 Vector ALU FMA benchmark..."))
        vec_res = run_fp32_vector_benchmark(
            n_elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        results.append((vec_res, info.spec.tflops_fp32_vpu))

    # FP32 Matrix GEMM Benchmark
    if args.test in ["all", "fp32_gemm"]:
        if not args.json:
            print(format_cell("Running FP32 Matrix GEMM benchmark..."))
        gemm_res = run_fp32_gemm_benchmark(
            matrix_dim=args.matrix_dim,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        # GEMM runs on MXU / Vector Core
        peak_ref = info.spec.tflops_bf16_mxu if info.is_tpu else info.spec.tflops_fp32_vpu
        results.append((gemm_res, peak_ref))

    if args.json:
        out = {
            "device": {
                "name": info.spec.name,
                "platform": info.platform,
                "architecture": info.spec.architecture,
                "device_count": info.device_count,
                "device_id": args.device_id,
            },
            "benchmarks": [
                {
                    "name": r.name,
                    "precision": r.precision,
                    "operation": r.operation.strip(),
                    "tflops_per_sec": r.tflops_per_sec,
                    "min_time_sec": r.min_time_sec,
                    "avg_time_sec": r.avg_time_sec,
                    "total_flops": r.total_flops,
                    "theoretical_peak_tflops": peak,
                    "efficiency_pct": (r.tflops_per_sec / peak * 100.0) if peak > 0 else 0.0,
                }
                for r, peak in results
            ]
        }
        print(json.dumps(out, indent=2))
    else:
        line = "-" * 77
        print(f"|{line}|")
        for r, peak in results:
            print_result_row(r, peak)
        print_footer()

if __name__ == "__main__":
    main()
