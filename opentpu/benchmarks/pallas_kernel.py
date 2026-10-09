from __future__ import annotations
import time
from typing import Optional
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from opentpu.benchmarks.fp32 import BenchmarkResult

def patch_libtpu_version_check():
    """
    Patches JAX's internal libtpu semver check on Cloud TPU / Google Colab environments
    where Google uses a custom internal TFRT build string without semver numbers.
    """
    try:
        from jax._src import cloud_tpu_init
        cloud_tpu_init.is_libtpu_at_least = lambda v: True
    except Exception:
        pass

    try:
        from jax._src.pallas.mosaic import lowering
        lowering.is_libtpu_at_least = lambda v: True
    except Exception:
        pass


def run_pallas_vector_benchmark(
    n_elements: int = 1048576,
    iterations: int = 20,
    warmup: int = 3,
    unroll_steps: int = 64,
    device: Optional[any] = None,
) -> Optional[BenchmarkResult]:
    """
    Executes a custom low-level TPU kernel using Google Pallas / Mosaic MLIR.
    Tiles data directly in TPU Vector Memory (VMEM) and executes unrolled FMA loops.
    Returns None if Pallas TPU backend is not supported by the environment.
    """
    patch_libtpu_version_check()

    if device is None:
        device = jax.devices()[0]

    is_tpu = device.platform.lower() == "tpu"
    interpret = not is_tpu

    block_size = 1024
    num_blocks = n_elements // block_size

    def pallas_fma_kernel(x_ref, y_ref, out_ref):
        x = x_ref[...]
        y = y_ref[...]
        for _ in range(unroll_steps):
            x = y * x + y
            y = x * y + x
        out_ref[...] = y

    try:
        kernel_fn = pl.pallas_call(
            pallas_fma_kernel,
            out_shape=jax.ShapeDtypeStruct((n_elements,), jnp.float32),
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i * block_size,)),
                pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i * block_size,)),
            ],
            out_specs=pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i * block_size,)),
            interpret=interpret,
        )

        key = jax.random.PRNGKey(0)
        x = jax.device_put(jax.random.uniform(key, shape=(n_elements,), dtype=jnp.float32), device)
        y = jax.device_put(jax.random.uniform(key, shape=(n_elements,), dtype=jnp.float32) * 0.01 + 1.0, device)

        # Warmup and compile
        for _ in range(warmup):
            res = kernel_fn(x, y)
            res.block_until_ready()

        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            res = kernel_fn(x, y)
            res.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        min_time = min(times)
        avg_time = sum(times) / len(times)
        flops_per_element = unroll_steps * 4
        total_flops = float(n_elements) * float(flops_per_element)
        tflops_per_sec = (total_flops / min_time) * 1e-12
        device_name = getattr(device, "device_kind", device.platform)

        return BenchmarkResult(
            name="Pallas/Mosaic Compute (Vector ALU)",
            precision="FP32 ",
            operation="pallas, fma  ",
            num_elements=n_elements,
            flops_per_element=flops_per_element,
            total_flops=total_flops,
            min_time_sec=min_time,
            avg_time_sec=avg_time,
            tflops_per_sec=tflops_per_sec,
            device_name=device_name,
            iterations=iterations,
        )
    except Exception:
        # Pallas lowering is not supported or failed on this platform/libtpu
        return None
