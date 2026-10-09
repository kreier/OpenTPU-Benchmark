from __future__ import annotations
import argparse
import json
import sys
from opentpu.device_info import get_platform_info
from opentpu.benchmarks.fp32 import run_fp32_vector_benchmark, run_fp32_gemm_benchmark
from opentpu.benchmarks.bf16 import run_bf16_gemm_benchmark, run_bf16_vector_benchmark
from opentpu.benchmarks.fp16 import run_fp16_gemm_benchmark, run_fp16_vector_benchmark
from opentpu.benchmarks.integer import run_int8_gemm_benchmark, run_int32_vector_benchmark
from opentpu.benchmarks.memory import run_memory_bandwidth_benchmark
from opentpu.reporter import (
    print_banner,
    print_result_row,
    print_memory_row,
    print_unsupported_row,
    print_footer,
    format_cell,
)

def main():
    parser = argparse.ArgumentParser(description="OpenTPU-Benchmark Suite")
    parser.add_argument("--test", choices=["all", "fp32", "bf16", "fp16", "int", "memory"], default="all",
                        help="Which benchmark suite to run (default: all)")
    parser.add_argument("--elements", type=int, default=4 * 1024 * 1024,
                        help="Number of elements for Vector ALU benchmarks (default: 4194304)")
    parser.add_argument("--matrix-dim", type=int, default=4096,
                        help="Matrix dimension for GEMM benchmarks (default: 4096)")
    parser.add_argument("--buffer-mb", type=int, default=512,
                        help="Buffer size in MB for memory bandwidth benchmarks (default: 512)")
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
    compute_results = []
    memory_results = []

    if not args.json:
        print_banner(info)
        print(format_cell("JIT compiling and warming up kernels..."))

    # FP64 Compute Check (Negligible / Emulated on TPU)
    has_fp64 = not info.is_tpu

    # 1. FP32 Benchmarks
    if args.test in ["all", "fp32"]:
        if not args.json:
            print(format_cell("Running FP32 Vector ALU & Matrix GEMM..."))
        vec_fp32 = run_fp32_vector_benchmark(
            n_elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((vec_fp32, info.spec.tflops_fp32_vpu))

        gemm_fp32 = run_fp32_gemm_benchmark(
            matrix_dim=args.matrix_dim,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        peak_gemm = info.spec.tflops_bf16_mxu if info.is_tpu else info.spec.tflops_fp32_vpu
        compute_results.append((gemm_fp32, peak_gemm))

    # 2. BF16 Benchmarks (Flagship TPU Engine)
    if args.test in ["all", "bf16"]:
        if not args.json:
            print(format_cell("Running BF16 Matrix GEMM & Vector ALU..."))
        gemm_bf16 = run_bf16_gemm_benchmark(
            matrix_dim=args.matrix_dim,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((gemm_bf16, info.spec.tflops_bf16_mxu))

        vec_bf16 = run_bf16_vector_benchmark(
            n_elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((vec_bf16, info.spec.tflops_fp32_vpu * 2.0 if info.is_tpu else info.spec.tflops_fp32_vpu))

    # 3. FP16 Benchmarks
    if args.test in ["all", "fp16"]:
        if not args.json:
            print(format_cell("Running FP16 Matrix GEMM & Vector ALU..."))
        gemm_fp16 = run_fp16_gemm_benchmark(
            matrix_dim=args.matrix_dim,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((gemm_fp16, info.spec.tflops_bf16_mxu))

        vec_fp16 = run_fp16_vector_benchmark(
            n_elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((vec_fp16, info.spec.tflops_fp32_vpu * 2.0 if info.is_tpu else info.spec.tflops_fp32_vpu))

    # 4. Integer Benchmarks (INT32 & INT8)
    if args.test in ["all", "int"]:
        if not args.json:
            print(format_cell("Running INT32 Vector & INT8 Matrix GEMM..."))
        vec_int32 = run_int32_vector_benchmark(
            n_elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((vec_int32, info.spec.tflops_fp32_vpu))

        gemm_int8 = run_int8_gemm_benchmark(
            matrix_dim=args.matrix_dim,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        compute_results.append((gemm_int8, info.spec.tflops_bf16_mxu))

    # 5. Memory Bandwidth Benchmarks
    if args.test in ["all", "memory"]:
        if not args.json:
            print(format_cell("Running HBM Memory Bandwidth tests..."))
        mem_results = run_memory_bandwidth_benchmark(
            buffer_size_mb=args.buffer_mb,
            iterations=args.iterations,
            warmup=args.warmup,
            device=target_device
        )
        memory_results.extend(mem_results)

    # Output formatting
    if args.json:
        out = {
            "device": {
                "name": info.spec.name,
                "platform": info.platform,
                "architecture": info.spec.architecture,
                "device_count": info.device_count,
                "device_id": args.device_id,
            },
            "compute_benchmarks": [
                {
                    "name": r.name,
                    "precision": r.precision.strip(),
                    "operation": r.operation.strip(),
                    "throughput_tflops_or_tiops": r.tflops_per_sec,
                    "min_time_sec": r.min_time_sec,
                    "avg_time_sec": r.avg_time_sec,
                    "total_ops": r.total_flops,
                    "theoretical_peak": peak,
                    "efficiency_pct": (r.tflops_per_sec / peak * 100.0) if peak > 0 else 0.0,
                }
                for r, peak in compute_results
            ],
            "memory_benchmarks": [
                {
                    "name": m.name,
                    "operation": m.operation.strip(),
                    "bandwidth_gb_per_sec": m.bandwidth_gb_per_sec,
                    "min_time_sec": m.min_time_sec,
                    "avg_time_sec": m.avg_time_sec,
                    "buffer_bytes": m.buffer_bytes,
                }
                for m in memory_results
            ]
        }
        print(json.dumps(out, indent=2))
    else:
        line = "-" * 77
        print(f"|{line}|")
        if not has_fp64:
            print_unsupported_row("FP64", "double, fma  ", "not supported")
        for r, peak in compute_results:
            print_result_row(r, peak)
        if memory_results:
            print(f"|{line}|")
            for m in memory_results:
                print_memory_row(m)
        print_footer()

if __name__ == "__main__":
    main()
