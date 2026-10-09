from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List
import jax
import jax.numpy as jnp

@dataclass
class MemoryBenchmarkResult:
    name: str
    operation: str
    buffer_bytes: int
    bandwidth_gb_per_sec: float
    min_time_sec: float
    avg_time_sec: float
    device_name: str
    iterations: int


def run_memory_bandwidth_benchmark(
    buffer_size_mb: int = 512,
    iterations: int = 20,
    warmup: int = 5,
    device: Optional[any] = None,
) -> List[MemoryBenchmarkResult]:
    """
    Measures TPU HBM memory bandwidth for streaming read and streaming copy/write.
    Matches OpenCL-Benchmark's Coalesced Read and Coalesced Write metrics.
    """
    if device is None:
        device = jax.devices()[0]

    n_floats = (buffer_size_mb * 1024 * 1024) // 4
    key = jax.random.PRNGKey(123)
    data = jax.device_put(jax.random.normal(key, shape=(n_floats,), dtype=jnp.float32), device)
    total_bytes = data.nbytes

    # 1. Coalesced Streaming Read: Read entire HBM buffer into ALU reduction
    @jax.jit
    def stream_read(buf):
        return jnp.sum(buf)

    for _ in range(warmup):
        res = stream_read(data)
        res.block_until_ready()

    read_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = stream_read(data)
        res.block_until_ready()
        t1 = time.perf_counter()
        read_times.append(t1 - t0)

    min_read_time = min(read_times)
    avg_read_time = sum(read_times) / len(read_times)
    # Bandwidth = Bytes read / time
    bw_read_gb_s = (float(total_bytes) / min_read_time) * 1e-9

    # 2. Coalesced Streaming Copy / Write (1 read + 1 write = 2 * total_bytes)
    @jax.jit
    def stream_copy(buf):
        return buf + 1.0

    for _ in range(warmup):
        res = stream_copy(data)
        res.block_until_ready()

    write_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = stream_copy(data)
        res.block_until_ready()
        t1 = time.perf_counter()
        write_times.append(t1 - t0)

    min_write_time = min(write_times)
    avg_write_time = sum(write_times) / len(write_times)
    # Effective streaming transfer bandwidth = 2 * total_bytes / time
    bw_write_gb_s = (float(total_bytes * 2) / min_write_time) * 1e-9

    device_name = getattr(device, "device_kind", device.platform)

    return [
        MemoryBenchmarkResult(
            name="Memory Bandwidth (coalesced read)",
            operation="coalesced read      ",
            buffer_bytes=total_bytes,
            bandwidth_gb_per_sec=bw_read_gb_s,
            min_time_sec=min_read_time,
            avg_time_sec=avg_read_time,
            device_name=device_name,
            iterations=iterations,
        ),
        MemoryBenchmarkResult(
            name="Memory Bandwidth (coalesced write)",
            operation="coalesced      write",
            buffer_bytes=total_bytes * 2,
            bandwidth_gb_per_sec=bw_write_gb_s,
            min_time_sec=min_write_time,
            avg_time_sec=avg_write_time,
            device_name=device_name,
            iterations=iterations,
        ),
    ]
