import pytest
import jax
from opentpu.device_info import get_platform_info, KNOWN_SPECS
from opentpu.benchmarks.fp32 import run_fp32_vector_benchmark, run_fp32_gemm_benchmark
from opentpu.benchmarks.bf16 import run_bf16_vector_benchmark, run_bf16_gemm_benchmark
from opentpu.benchmarks.fp16 import run_fp16_vector_benchmark, run_fp16_gemm_benchmark
from opentpu.benchmarks.integer import run_int8_gemm_benchmark, run_int32_vector_benchmark
from opentpu.benchmarks.memory import run_memory_bandwidth_benchmark
from opentpu.reporter import format_fraction

def test_device_info_discovery():
    info = get_platform_info()
    assert info.device_count >= 1
    assert info.platform in ["cpu", "tpu", "cuda", "rocm"]
    assert info.spec is not None
    assert "tpu v5 lite" in KNOWN_SPECS
    assert "tpu v5e" in KNOWN_SPECS
    assert KNOWN_SPECS["tpu v5 lite"].tflops_bf16_mxu == 197.0
    assert KNOWN_SPECS["tpu v5 lite"].tflops_fp32_vpu == 12.3

def test_fp32_benchmarks():
    res_vec = run_fp32_vector_benchmark(n_elements=8192, iterations=2, warmup=1, unroll_steps=16)
    assert res_vec.precision == "FP32"
    assert res_vec.tflops_per_sec > 0

    res_gemm = run_fp32_gemm_benchmark(matrix_dim=128, iterations=2, warmup=1)
    assert res_gemm.precision == "FP32"
    assert res_gemm.tflops_per_sec > 0

def test_bf16_benchmarks():
    res_gemm = run_bf16_gemm_benchmark(matrix_dim=128, iterations=2, warmup=1)
    assert res_gemm.precision == "BF16"
    assert res_gemm.tflops_per_sec > 0

    res_vec = run_bf16_vector_benchmark(n_elements=8192, iterations=2, warmup=1, unroll_steps=16)
    assert res_vec.precision == "BF16"
    assert res_vec.tflops_per_sec > 0

def test_fp16_benchmarks():
    res_gemm = run_fp16_gemm_benchmark(matrix_dim=128, iterations=2, warmup=1)
    assert res_gemm.precision == "FP16"
    assert res_gemm.tflops_per_sec > 0

    res_vec = run_fp16_vector_benchmark(n_elements=8192, iterations=2, warmup=1, unroll_steps=16)
    assert res_vec.precision == "FP16"
    assert res_vec.tflops_per_sec > 0

def test_integer_benchmarks():
    res_vec = run_int32_vector_benchmark(n_elements=8192, iterations=2, warmup=1, unroll_steps=16)
    assert res_vec.precision == "INT32"
    assert res_vec.tflops_per_sec > 0

    res_gemm = run_int8_gemm_benchmark(matrix_dim=128, iterations=2, warmup=1)
    assert res_gemm.precision == "INT8 "
    assert res_gemm.tflops_per_sec > 0

def test_memory_bandwidth_benchmark():
    mem_res = run_memory_bandwidth_benchmark(buffer_size_mb=16, iterations=2, warmup=1)
    assert len(mem_res) == 2
    assert mem_res[0].bandwidth_gb_per_sec > 0
    assert mem_res[1].bandwidth_gb_per_sec > 0

def test_format_fraction():
    assert format_fraction(100.0) == "( 1x )"
    assert format_fraction(50.0) == "(1/2 )"
    assert format_fraction(200.0) == "( 2x )"
    assert format_fraction(0.0) == "       "
