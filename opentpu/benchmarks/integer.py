from __future__ import annotations
import time
from typing import Optional
import jax
import jax.numpy as jnp
from jax import lax
from opentpu.benchmarks.fp32 import BenchmarkResult


def run_int8_gemm_benchmark(
    matrix_dim: int = 4096,
    iterations: int = 30,
    warmup: int = 5,
    device: Optional[any] = None,
) -> BenchmarkResult:
    """
    Runs the INT8 Quantized Matrix Multiplication benchmark (equivalent to OpenCL dp4a / MXU int8).
    Accumulates into int32. Measures TPU INT8 Systolic Array throughput (up to 197 TOPs on TPU v5e).
    Total IOPs = 2 * (matrix_dim ** 3).
    """
    if device is None:
        device = jax.devices()[0]

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.device_put(jax.random.randint(k1, shape=(matrix_dim, matrix_dim), minval=-128, maxval=127, dtype=jnp.int8), device)
    b = jax.device_put(jax.random.randint(k2, shape=(matrix_dim, matrix_dim), minval=-128, maxval=127, dtype=jnp.int8), device)

    @jax.jit
    def int8_gemm_kernel(mat_a, mat_b):
        return jnp.matmul(mat_a, mat_b, preferred_element_type=jnp.int32)

    for _ in range(warmup):
        res = int8_gemm_kernel(a, b)
        res.block_until_ready()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = int8_gemm_kernel(a, b)
        res.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)

    total_ops = 2.0 * (float(matrix_dim) ** 3)
    tiops_per_sec = (total_ops / min_time) * 1e-12
    device_name = getattr(device, "device_kind", device.platform)

    return BenchmarkResult(
        name="INT8 Compute (Matrix GEMM / dp4a)",
        precision="INT8 ",
        operation="int8  , mxu  ",
        num_elements=matrix_dim * matrix_dim,
        flops_per_element=2 * matrix_dim,
        total_flops=total_ops,
        min_time_sec=min_time,
        avg_time_sec=avg_time,
        tflops_per_sec=tiops_per_sec,
        device_name=device_name,
        iterations=iterations,
    )


def run_int32_vector_benchmark(
    n_elements: int = 4 * 1024 * 1024,
    iterations: int = 50,
    warmup: int = 5,
    unroll_steps: int = 512,
    device: Optional[any] = None,
) -> BenchmarkResult:
    """
    Runs the INT32 Vector ALU benchmark (direct port of OpenCL kernel_int).
    x = y * x + y; y = x * y + x in int32.
    """
    if device is None:
        device = jax.devices()[0]

    key = jax.random.PRNGKey(0)
    x = jax.device_put(jax.random.randint(key, shape=(n_elements,), minval=1, maxval=100, dtype=jnp.int32), device)
    y = jax.device_put(jax.random.randint(key, shape=(n_elements,), minval=1, maxval=10, dtype=jnp.int32), device)

    @jax.jit
    def int32_kernel(x_init: jnp.ndarray, y_init: jnp.ndarray) -> jnp.ndarray:
        def body_fn(i, val):
            x, y = val
            x = y * x + y
            y = x * y + x
            return (x, y)

        x_final, y_final = lax.fori_loop(0, unroll_steps, body_fn, (x_init, y_init))
        return y_final

    for _ in range(warmup):
        res = int32_kernel(x, y)
        res.block_until_ready()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = int32_kernel(x, y)
        res.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    min_time = min(times)
    avg_time = sum(times) / len(times)

    ops_per_element = unroll_steps * 4
    total_ops = float(n_elements) * float(ops_per_element)
    tiops_per_sec = (total_ops / min_time) * 1e-12
    device_name = getattr(device, "device_kind", device.platform)

    return BenchmarkResult(
        name="INT32 Compute (Vector ALU)",
        precision="INT32",
        operation="int   , a*b+c",
        num_elements=n_elements,
        flops_per_element=ops_per_element,
        total_flops=total_ops,
        min_time_sec=min_time,
        avg_time_sec=avg_time,
        tflops_per_sec=tiops_per_sec,
        device_name=device_name,
        iterations=iterations,
    )
