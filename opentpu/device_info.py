from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List
import jax

@dataclass
class DeviceSpec:
    name: str
    architecture: str
    tflops_fp32_vpu: float  # Theoretical Peak FP32 Vector ALU TFLOPS
    tflops_bf16_mxu: float  # Theoretical Peak BF16/INT8 MXU TFLOPS
    tflops_fp64: float      # Theoretical Peak FP64 TFLOPS (usually negligible on TPU)
    hbm_capacity_gb: float  # HBM capacity in GB
    hbm_bandwidth_gbs: float # HBM peak bandwidth in GB/s
    interconnect_gbps: float # ICI bandwidth per chip in Gbps

# Known Google TPU specs (per single chip / core)
KNOWN_SPECS: Dict[str, DeviceSpec] = {
    "tpu v5 lite": DeviceSpec(
        name="Google TPU v5e (v5 lite)",
        architecture="Viperfish (v5e / v5 lite)",
        tflops_fp32_vpu=12.3,      # Vector Processing Unit FP32 peak (~12.3 TFLOPS)
        tflops_bf16_mxu=197.0,     # Matrix Multiply Unit BF16 peak (197 TFLOPS)
        tflops_fp64=0.0,           # Hardware FP64 not supported in MXU
        hbm_capacity_gb=16.0,      # 16 GB HBM2e
        hbm_bandwidth_gbs=816.0,   # 816 GB/s
        interconnect_gbps=400.0,   # 400 Gbps ICI
    ),
    "tpu v5e": DeviceSpec(
        name="Google TPU v5e",
        architecture="Viperfish (v5e)",
        tflops_fp32_vpu=12.3,      # Vector Processing Unit FP32 peak (~12.3 TFLOPS)
        tflops_bf16_mxu=197.0,     # Matrix Multiply Unit BF16 peak (197 TFLOPS)
        tflops_fp64=0.0,           # Hardware FP64 not supported in MXU
        hbm_capacity_gb=16.0,      # 16 GB HBM2e
        hbm_bandwidth_gbs=816.0,   # 816 GB/s
        interconnect_gbps=400.0,   # 400 Gbps ICI
    ),
    "tpu v5p": DeviceSpec(
        name="Google TPU v5p",
        architecture="Viperfish-High (v5p)",
        tflops_fp32_vpu=28.7,
        tflops_bf16_mxu=459.0,
        tflops_fp64=0.0,
        hbm_capacity_gb=95.0,
        hbm_bandwidth_gbs=2765.0,
        interconnect_gbps=4800.0,
    ),
    "tpu v4": DeviceSpec(
        name="Google TPU v4",
        architecture="Pufferfish (v4)",
        tflops_fp32_vpu=17.2,
        tflops_bf16_mxu=275.0,
        tflops_fp64=0.0,
        hbm_capacity_gb=32.0,
        hbm_bandwidth_gbs=1200.0,
        interconnect_gbps=4800.0,
    ),
    "tpu v3": DeviceSpec(
        name="Google TPU v3",
        architecture="TPU v3",
        tflops_fp32_vpu=8.0,
        tflops_bf16_mxu=123.0,
        tflops_fp64=0.0,
        hbm_capacity_gb=16.0,
        hbm_bandwidth_gbs=900.0,
        interconnect_gbps=650.0,
    ),
    "tpu v2": DeviceSpec(
        name="Google TPU v2",
        architecture="TPU v2",
        tflops_fp32_vpu=4.0,
        tflops_bf16_mxu=45.0,
        tflops_fp64=0.0,
        hbm_capacity_gb=8.0,
        hbm_bandwidth_gbs=600.0,
        interconnect_gbps=0.0,
    ),
    "cpu": DeviceSpec(
        name="Host CPU (Emulation / Fallback)",
        architecture="x86_64/ARM",
        tflops_fp32_vpu=1.0,
        tflops_bf16_mxu=1.0,
        tflops_fp64=0.5,
        hbm_capacity_gb=32.0,
        hbm_bandwidth_gbs=50.0,
        interconnect_gbps=0.0,
    ),
}

@dataclass
class TPUPlatformInfo:
    platform: str
    device_kind: str
    device_count: int
    devices: List[any]
    spec: DeviceSpec

    @property
    def is_tpu(self) -> bool:
        return self.platform.lower() == "tpu"


def get_platform_info() -> TPUPlatformInfo:
    """Discover current hardware platform and match known specs."""
    devices = jax.devices()
    first_device = devices[0]
    platform = first_device.platform
    device_kind = getattr(first_device, "device_kind", platform).lower()

    matched_spec: Optional[DeviceSpec] = None
    for key, spec in KNOWN_SPECS.items():
        if key in device_kind:
            matched_spec = spec
            break

    if not matched_spec:
        if platform == "tpu":
            matched_spec = DeviceSpec(
                name=f"Generic Google TPU ({device_kind})",
                architecture=device_kind,
                tflops_fp32_vpu=12.0,
                tflops_bf16_mxu=197.0,
                tflops_fp64=0.0,
                hbm_capacity_gb=16.0,
                hbm_bandwidth_gbs=816.0,
                interconnect_gbps=400.0,
            )
        else:
            matched_spec = DeviceSpec(
                name=f"{platform.upper()} Device ({device_kind})",
                architecture=platform,
                tflops_fp32_vpu=1.0,
                tflops_bf16_mxu=1.0,
                tflops_fp64=0.5,
                hbm_capacity_gb=16.0,
                hbm_bandwidth_gbs=50.0,
                interconnect_gbps=0.0,
            )

    return TPUPlatformInfo(
        platform=platform,
        device_kind=device_kind,
        device_count=len(devices),
        devices=devices,
        spec=matched_spec,
    )
