from __future__ import annotations
import math
from typing import List, Optional
from opentpu.device_info import TPUPlatformInfo
from opentpu.benchmarks.fp32 import BenchmarkResult
from opentpu.benchmarks.memory import MemoryBenchmarkResult
from opentpu.benchmarks.interconnect import InterconnectBenchmarkResult

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


def format_cell(text: str, width: int = 75) -> str:
    """Pad string to fit inside OpenCL-Benchmark ASCII frame (total 79 chars)."""
    return f"| {text.ljust(width)} |"


def print_banner(info: TPUPlatformInfo):
    from opentpu import __version__
    title = f" OpenTPU Benchmark v{__version__} "
    total_width = 79
    hyphens_total = total_width - 2 - len(title)
    left_h = hyphens_total // 2
    right_h = hyphens_total - left_h
    top_line = f".{'-' * left_h}{title}{'-' * right_h}."
    line = "-" * 77
    print(top_line)
    print(format_cell(f"Device: {info.spec.name}"))
    print(format_cell(f"Platform: {info.platform.upper()} | Cores/Chips: {info.device_count} | Architecture: {info.spec.architecture}"))
    print(format_cell(f"HBM Capacity: {info.spec.hbm_capacity_gb:.1f} GB | HBM Peak Bandwidth: {info.spec.hbm_bandwidth_gbs:.0f} GB/s"))
    print(format_cell(f"Peak Spec: FP32 VPU: {info.spec.tflops_fp32_vpu:.1f} TFLOPS | BF16 MXU: {info.spec.tflops_bf16_mxu:.1f} TFLOPS"))
    print(f"|{line}|")


def print_result_row(result: BenchmarkResult, theoretical_peak_tflops: float):
    percentage = (result.tflops_per_sec / theoretical_peak_tflops * 100.0) if theoretical_peak_tflops > 0 else 0.0
    frac_str = format_fraction(percentage)
    unit = " TIOPs/s" if "INT" in result.precision else "TFLOPs/s"
    
    label = f"{result.precision.strip().ljust(6)} Compute   ({result.operation.strip().ljust(13)})"
    flops_str = f"{result.tflops_per_sec:10.3f}".rjust(26)
    row = f"| {label} {flops_str} {unit} {frac_str} |"
    print(row)


def print_unsupported_row(precision: str, operation: str, reason: str = "not supported"):
    label = f"{precision.strip().ljust(6)} Compute   ({operation.strip().ljust(13)})"
    row = f"| {label}                       {reason.rjust(13)}        |"
    print(row)


def print_memory_row(result: MemoryBenchmarkResult):
    op = result.operation.strip()
    if op == "coalesced read":
        label = "Memory Bandwidth ( coalesced read      )"
    elif op == "coalesced write":
        label = "Memory Bandwidth ( coalesced      write)"
    else:
        label = f"Memory Bandwidth ({result.operation.ljust(21)})"
    bw_str = f"{result.bandwidth_gb_per_sec:10.2f}".rjust(29)
    print(f"| {label} {bw_str} GB/s |")


def print_interconnect_row(result: InterconnectBenchmarkResult):
    prefix = "ICI  " if "ICI" in result.name else "PCIe "
    op = result.operation.strip()
    bw_str = f"{result.bandwidth_gb_per_sec:10.2f}".rjust(29)
    if result.pcie_gen:
        label = f"{prefix}  Bandwidth (        bidirectional)"
        gen_str = f"{result.pcie_gen}"
        num_str = f"{result.bandwidth_gb_per_sec:.2f}".rjust(8)
        print(f"| {label}            {gen_str}{num_str} GB/s |")
    else:
        if op == "send":
            label = f"{prefix}  Bandwidth (send                 )"
        elif op == "receive":
            label = f"{prefix}  Bandwidth (   receive           )"
        else:
            label = f"{prefix}  Bandwidth ({op.ljust(21)})"
        print(f"| {label} {bw_str} GB/s |")


def print_footer():
    line = "-" * 77
    print(f"'{line}'")
