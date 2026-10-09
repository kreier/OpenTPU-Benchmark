from __future__ import annotations
import math
from typing import List, Optional
from opentpu.device_info import TPUPlatformInfo
from opentpu.benchmarks.fp32 import BenchmarkResult

FRACTION_VALUES = [
    1.0 / 64.0, 1.0 / 32.0, 1.0 / 24.0, 1.0 / 16.0, 1.0 / 12.0,
    1.0 / 8.0,  1.0 / 4.0,  1.0 / 3.0,  1.0 / 2.0,  2.0 / 3.0,
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0
]
FRACTION_STRINGS = [
    "1/64", "1/32", "1/24", "1/16", "1/12",
    "1/8 ", "1/4 ", "1/3 ", "1/2 ", "2/3 ",
    " 1x ", " 2x ", " 4x ", " 8x ", " 16x", " 32x", " 64x"
]

def format_fraction(percentage_of_peak: float) -> str:
    """Calculates closest ratio string e.g. ( 1x ), (1/2 ), ( 2x )."""
    if percentage_of_peak <= 0.0:
        return "       "
    ratio = percentage_of_peak * 0.01
    best_diff = float("inf")
    best_idx = 10
    for i, val in enumerate(FRACTION_VALUES):
        diff = (ratio - val) ** 2
        if diff <= best_diff:
            best_diff = diff
            best_idx = i
    return f"({FRACTION_STRINGS[best_idx]})"


def format_cell(text: str, width: int = 77) -> str:
    """Pad string to fit inside OpenCL-Benchmark ASCII frame."""
    return f"| {text.ljust(width)} |"


def print_banner(info: TPUPlatformInfo):
    line = "-" * 77
    print(f".{line}.")
    print(format_cell(f"Device: {info.spec.name}"))
    print(format_cell(f"Platform: {info.platform.upper()} | Cores/Chips: {info.device_count} | Architecture: {info.spec.architecture}"))
    print(format_cell(f"HBM Capacity: {info.spec.hbm_capacity_gb:.1f} GB | HBM Peak Bandwidth: {info.spec.hbm_bandwidth_gbs:.0f} GB/s"))
    print(format_cell(f"Peak Spec: FP32 VPU: {info.spec.tflops_fp32_vpu:.1f} TFLOPS | BF16 MXU: {info.spec.tflops_bf16_mxu:.1f} TFLOPS"))
    print(f"|{line}|")


def print_result_row(result: BenchmarkResult, theoretical_peak_tflops: float):
    percentage = (result.tflops_per_sec / theoretical_peak_tflops * 100.0) if theoretical_peak_tflops > 0 else 0.0
    frac_str = format_fraction(percentage)
    
    # Format matching: | FP32   Compute   (float , fma  )                      12.300 TFLOPs/s ( 1x ) |
    label = f"{result.precision.ljust(7)} Compute   ({result.operation})"
    tflops_str = f"{result.tflops_per_sec:10.3f} TFLOPs/s"
    
    # Inner width = 77 chars: label (32) + space + tflops_str (20) + space + frac (7)
    content = f"{label}   {tflops_str:>26} {frac_str}"
    print(format_cell(content))


def print_footer():
    line = "-" * 77
    print(f"'{line}'")
