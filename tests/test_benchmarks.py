import pytest
import jax
from opentpu.device_info import get_platform_info, KNOWN_SPECS
from opentpu.benchmarks.fp32 import run_fp32_vector_benchmark, run_fp32_gemm_benchmark
from opentpu.reporter import format_fraction

def test_device_info_discovery():
    info = get_platform_info()
    assert info.device_count >= 1
    assert info.platform in ["cpu", "tpu", "cuda", "rocm"]
    assert info.spec is not None
    assert "tpu v5e" in KNOWN_SPECS
    assert KNOWN_SPECS["tpu v5e"].tflops_bf16_mxu == 197.0
    assert KNOWN_SPECS["tpu v5e"].tflops_fp32_vpu == 12.3

def test_fp32_vector_benchmark_correctness():
    # Small test for fast execution
    res = run_fp32_vector_benchmark(
        n_elements=16384,
        iterations=2,
        warmup=1,
        unroll_steps=16,
    )
    assert res.precision == "FP32"
    assert res.flops_per_element == 16 * 4
    assert res.total_flops == 16384 * 64
    assert res.min_time_sec > 0
    assert res.tflops_per_sec > 0

def test_fp32_gemm_benchmark_correctness():
    # Small matrix GEMM
    dim = 256
    res = run_fp32_gemm_benchmark(
        matrix_dim=dim,
        iterations=2,
        warmup=1,
    )
    expected_flops = 2.0 * (dim ** 3)
    assert res.total_flops == expected_flops
    assert res.min_time_sec > 0
    assert res.tflops_per_sec > 0

def test_format_fraction():
    assert format_fraction(100.0) == "( 1x )"
    assert format_fraction(50.0) == "(1/2 )"
    assert format_fraction(200.0) == "( 2x )"
    assert format_fraction(0.0) == "       "
