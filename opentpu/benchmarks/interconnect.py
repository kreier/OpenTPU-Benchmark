from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

@dataclass
class InterconnectBenchmarkResult:
    name: str
    operation: str
    bandwidth_gb_per_sec: float
    min_time_sec: float
    avg_time_sec: float
    buffer_bytes: int
    pcie_gen: Optional[str]
    device_name: str
    iterations: int


def estimate_pcie_gen(bandwidth_gb_s: float) -> str:
    """Estimates PCIe generation for an x16 link based on measured bandwidth."""
    if bandwidth_gb_s > 100.0:
        return "Gen6"
    elif bandwidth_gb_s > 45.0:
        return "Gen5"
    elif bandwidth_gb_s > 22.0:
        return "Gen4"
    elif bandwidth_gb_s > 11.0:
        return "Gen3"
    elif bandwidth_gb_s > 5.5:
        return "Gen2"
    elif bandwidth_gb_s > 2.5:
        return "Gen1"
    else:
        return "Link"


def run_host_bandwidth_benchmark(
    buffer_size_mb: int = 256,
    iterations: int = 15,
    warmup: int = 3,
    device: Optional[any] = None,
) -> List[InterconnectBenchmarkResult]:
    """
    Measures Host RAM <-> TPU HBM transfer speeds over PCIe.
    Matches OpenCL-Benchmark's PCIe Bandwidth (send, receive, bidirectional).
    """
    if device is None:
        device = jax.devices()[0]

    n_elements = (buffer_size_mb * 1024 * 1024) // 4
    host_data = np.random.uniform(0.0, 1.0, size=(n_elements,)).astype(np.float32)
    total_bytes = host_data.nbytes
    device_name = getattr(device, "device_kind", device.platform)

    # 1. Host to Device (send)
    for _ in range(warmup):
        dev_buf = jax.device_put(host_data, device)
        dev_buf.block_until_ready()

    send_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        dev_buf = jax.device_put(host_data, device)
        dev_buf.block_until_ready()
        t1 = time.perf_counter()
        send_times.append(t1 - t0)

    min_send_time = min(send_times)
    avg_send_time = sum(send_times) / len(send_times)
    bw_send = (float(total_bytes) / min_send_time) * 1e-9

    # 2. Device to Host (receive)
    dev_buf = jax.device_put(host_data, device)
    dev_buf.block_until_ready()
    host_dest = np.empty_like(host_data)

    for _ in range(warmup):
        np.copyto(host_dest, dev_buf)

    receive_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        np.copyto(host_dest, dev_buf)
        t1 = time.perf_counter()
        receive_times.append(t1 - t0)

    min_receive_time = min(receive_times)
    avg_receive_time = sum(receive_times) / len(receive_times)
    bw_receive = (float(total_bytes) / min_receive_time) * 1e-9

    # 3. Bidirectional (Send + Receive simultaneously)
    half_bytes = total_bytes // 2
    half_host = host_data[: len(host_data) // 2]
    half_dest = np.empty_like(half_host)
    half_dev = jax.device_put(half_host, device)
    half_dev.block_until_ready()

    for _ in range(warmup):
        d_out = jax.device_put(half_host, device)
        np.copyto(half_dest, half_dev)
        d_out.block_until_ready()

    bidi_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        # Transfer host -> device
        d_out = jax.device_put(half_host, device)
        # Transfer device -> host into distinct host memory
        np.copyto(half_dest, half_dev)
        d_out.block_until_ready()
        t1 = time.perf_counter()
        bidi_times.append(t1 - t0)

    min_bidi_time = min(bidi_times)
    avg_bidi_time = sum(bidi_times) / len(bidi_times)
    bw_bidi = (float(total_bytes) / min_bidi_time) * 1e-9
    pcie_gen = estimate_pcie_gen(max(bw_send, bw_receive, bw_bidi))

    return [
        InterconnectBenchmarkResult(
            name="PCIe Bandwidth (send)",
            operation="send                 ",
            bandwidth_gb_per_sec=bw_send,
            min_time_sec=min_send_time,
            avg_time_sec=avg_send_time,
            buffer_bytes=total_bytes,
            pcie_gen=None,
            device_name=device_name,
            iterations=iterations,
        ),
        InterconnectBenchmarkResult(
            name="PCIe Bandwidth (receive)",
            operation="   receive           ",
            bandwidth_gb_per_sec=bw_receive,
            min_time_sec=min_receive_time,
            avg_time_sec=avg_receive_time,
            buffer_bytes=total_bytes,
            pcie_gen=None,
            device_name=device_name,
            iterations=iterations,
        ),
        InterconnectBenchmarkResult(
            name="PCIe Bandwidth (bidirectional)",
            operation="        bidirectional",
            bandwidth_gb_per_sec=bw_bidi,
            min_time_sec=min_bidi_time,
            avg_time_sec=avg_bidi_time,
            buffer_bytes=total_bytes,
            pcie_gen=f"({pcie_gen} x16)",
            device_name=device_name,
            iterations=iterations,
        ),
    ]


def run_ici_bandwidth_benchmark(
    buffer_size_mb: int = 128,
    iterations: int = 15,
    warmup: int = 3,
) -> Optional[InterconnectBenchmarkResult]:
    """
    Measures TPU-to-TPU Inter-Chip Interconnect (ICI) direct ring/torus bandwidth.
    Only executed when multiple TPU chips/cores are detected (e.g. v5e-4, v5e-8).
    """
    devices = jax.devices()
    num_devices = len(devices)
    if num_devices <= 1 or devices[0].platform.lower() != "tpu":
        return None

    n_elements_per_dev = (buffer_size_mb * 1024 * 1024) // (4 * num_devices)
    key = jax.random.PRNGKey(42)
    # Replicate/split across devices
    arr = jax.device_put_replicated(
        jax.random.normal(key, shape=(n_elements_per_dev,), dtype=jnp.float32),
        devices,
    )

    # Collective all-reduce across ICI ring
    @jax.pmap
    def ring_all_reduce(x):
        return lax.psum(x, axis_name="i")

    for _ in range(warmup):
        res = ring_all_reduce(arr)
        res.block_until_ready()

    ici_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = ring_all_reduce(arr)
        res.block_until_ready()
        t1 = time.perf_counter()
        ici_times.append(t1 - t0)

    min_time = min(ici_times)
    avg_time = sum(ici_times) / len(ici_times)
    total_bytes = n_elements_per_dev * 4 * num_devices
    # Ring all-reduce transfers 2 * (N - 1) / N * size
    ring_factor = 2.0 * (num_devices - 1) / num_devices
    bw_gb_s = (float(total_bytes) * ring_factor / min_time) * 1e-9

    return InterconnectBenchmarkResult(
        name=f"ICI Bandwidth (All-Reduce Ring, {num_devices} chips)",
        operation=f"ICI Ring ({num_devices} chips) ",
        bandwidth_gb_per_sec=bw_gb_s,
        min_time_sec=min_time,
        avg_time_sec=avg_time,
        buffer_bytes=total_bytes,
        pcie_gen=None,
        device_name="TPU ICI Subsystem",
        iterations=iterations,
    )
