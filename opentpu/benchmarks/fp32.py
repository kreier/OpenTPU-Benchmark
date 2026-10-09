from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax import lax

@dataclass
class BenchmarkResult:
    name: str
    precision: str
    operation: str
    num_elements: int
    flops_per_element: int
    total_flops: float
    min_time_sec: float
    avg_time_sec: float
    tflops_per_sec: float
    device_name: str
    iterations: int


def create_vector_fma_kernel(unroll_steps: int = 512):
    """
    Creates a JIT-compiled kernel matching OpenCL's kernel_float:
    For 512 iterations:
        x = fma(y, x, y)  # 2 ops
        y = fma(x, y, x)  # 2 ops
    Total = 512 * 4 = 2048 FLOPs per element.
    """
    @jax.jit
    def vector_fma_kernel(x_init: jnp.ndarray, y_init: jnp.ndarray) -> jnp.ndarray:
        def body_fn(i, val):
            x, y = val
            x = y * x + y
            y = x * y + x
            return (x, y)

        x_final, y_final = lax.fori_loop(0, unroll_steps, body_fn, (x_init, y_init))
        return y_final

    return vector_fma_kernel


def run_fp32_vector_benchmark(
    n_elements: int = 4 * 1024 * 1024,
    iterations: int = 50,
    warmup: int = 5,
    unroll_steps: int = 512,
    device: Optional[any] = None,
) -> BenchmarkResult:
    """
    Runs the FP32 Vector ALU FMA microbenchmark (direct port of OpenCL kernel_float).
    Measures TPU Vector Processing Unit (VPU) throughput.
    """
    if device is None:
        device = jax.devices()[0]

    # Initialize data on device
    key = jax.random.PRNGKey(0)
    x = jax.device_put(jax.random.uniform(key, shape=(n_elements,), dtype=jnp.float32), device)
    y = jax.device_put(jax.random.uniform(key, shape=(n_elements,), dtype=jnp.float32) * 0.01 + 1.0, device)

    kernel = create_vector_fma_kernel(unroll_steps=unroll_steps)

    # Warmup and trigger JIT compilation
    for _ in range(warmup):
        res = kernel(x, y)
        res.block_until_ready()

    # Benchmark loop
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = kernel(x, y)
        res.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)

    flops_per_element = unroll_steps * 4  # 4 FLOPs per loop iteration (2 FMAs)
    total_flops = float(n_elements) * float(flops_per_element)
    tflops_per_sec = (total_flops / min_time) * 1e-12

    device_name = getattr(device, "device_kind", device.platform)

    return BenchmarkResult(
        name="FP32 Compute (Vector ALU)",
        precision="FP32",
        operation="float , fma  ",
        num_elements=n_elements,
        flops_per_element=flops_per_element,
        total_flops=total_flops,
        min_time_sec=min_time,
        avg_time_sec=avg_time,
        tflops_per_sec=tflops_per_sec,
        device_name=device_name,
        iterations=iterations,
    )


def run_fp32_gemm_benchmark(
    matrix_dim: int = 4096,
    iterations: int = 30,
    warmup: int = 5,
    device: Optional[any] = None,
) -> BenchmarkResult:
    """
    Runs the FP32 Matrix Multiplication (GEMM) benchmark: C = A @ B.
    Measures TPU Matrix Multiply Unit (MXU) FP32 throughput.
    Total FLOPs = 2 * (matrix_dim ** 3).
    """
    if device is None:
        device = jax.devices()[0]

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.device_put(jax.random.normal(k1, shape=(matrix_dim, matrix_dim), dtype=jnp.float32), device)
    b = jax.device_put(jax.random.normal(k2, shape=(matrix_dim, matrix_dim), dtype=jnp.float32), device)

    @jax.jit
    def gemm_kernel(mat_a, mat_b):
        return jnp.matmul(mat_a, mat_b)

    # Warmup and JIT compile
    for _ in range(warmup):
        res = gemm_kernel(a, b)
        res.block_until_ready()

    # Benchmark loop
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = gemm_kernel(a, b)
        res.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)

    total_flops = 2.0 * (float(matrix_dim) ** 3)
    tflops_per_sec = (total_flops / min_time) * 1e-12
    device_name = getattr(device, "device_kind", device.platform)

    return BenchmarkResult(
        name="FP32 Compute (Matrix GEMM)",
        precision="FP32",
        operation="float , gemm ",
        num_elements=matrix_dim * matrix_dim,
        flops_per_element=2 * matrix_dim,
        total_flops=total_flops,
        min_time_sec=min_time,
        avg_time_sec=avg_time,
        tflops_per_sec=tflops_per_sec,
        device_name=device_name,
        iterations=iterations,
    )
