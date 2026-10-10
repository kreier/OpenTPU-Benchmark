import os
import pytest
import jax
from opentpu.device_info import get_platform_info, KNOWN_SPECS
from opentpu.benchmarks.fp32 import run_fp32_vector_benchmark, run_fp32_gemm_benchmark
from opentpu.benchmarks.bf16 import run_bf16_vector_benchmark, run_bf16_gemm_benchmark
from opentpu.benchmarks.fp16 import run_fp16_vector_benchmark, run_fp16_gemm_benchmark
from opentpu.benchmarks.integer import run_int8_gemm_benchmark, run_int32_vector_benchmark
from opentpu.benchmarks.memory import run_memory_bandwidth_benchmark
from opentpu.benchmarks.interconnect import run_host_bandwidth_benchmark, run_ici_bandwidth_benchmark, estimate_pcie_gen
from opentpu.results_db import save_benchmark_run
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

def test_host_interconnect_benchmark():
    host_res = run_host_bandwidth_benchmark(buffer_size_mb=16, iterations=2, warmup=1)
    assert len(host_res) == 3
    for r in host_res:
        assert r.bandwidth_gb_per_sec > 0
    assert estimate_pcie_gen(30.0) == "Gen4"
    assert estimate_pcie_gen(60.0) == "Gen5"

    ici_res = run_ici_bandwidth_benchmark(buffer_size_mb=4, iterations=2, warmup=1)
    if len(jax.devices()) <= 1 or jax.devices()[0].platform.lower() != "tpu":
        assert ici_res is None
    else:
        assert ici_res is not None
        assert ici_res.bandwidth_gb_per_sec > 0

def test_ici_multi_device_execution():
    import subprocess
    import sys
    code = """
import jax
from opentpu.benchmarks.interconnect import run_ici_bandwidth_benchmark
devices = jax.devices()
res = run_ici_bandwidth_benchmark(buffer_size_mb=4, iterations=2, warmup=1, devices=devices)
assert res is not None
assert res.bandwidth_gb_per_sec > 0
"""
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"Error running multi-device ICI test: {result.stderr}"

def test_results_db(tmp_path):
    test_file = str(tmp_path / "test_results.json")
    dummy_data = {
        "device": {"name": "Test Device", "architecture": "Test Arch"},
        "compute_benchmarks": [{"operation": "float , fma", "throughput_tflops_or_tiops": 1.0}],
        "memory_benchmarks": [{"operation": "coalesced read", "bandwidth_gb_per_sec": 500.0}],
    }
    saved_path = save_benchmark_run(dummy_data, filepath=test_file)
    assert os.path.exists(saved_path)

def test_pallas_benchmark():
    from opentpu.benchmarks.pallas_kernel import run_pallas_vector_benchmark
    res = run_pallas_vector_benchmark(n_elements=8192, iterations=2, warmup=1, unroll_steps=16)
    if res is not None:
        assert res.tflops_per_sec > 0
        assert "pallas" in res.operation

def test_format_fraction():
    assert format_fraction(100.0) == "( 1x )"
    assert format_fraction(50.0) == "(1/2 )"
    assert format_fraction(200.0) == "( 2x )"
    assert format_fraction(0.0) == "       "
