"""OpenTPU-Benchmark: High-performance microbenchmark suite for Google TPUs.

Ported from OpenCL-Benchmark, targeting Google Cloud TPUs (TPU v5e, v4, v5p, v6e)
and providing compatibility with CPU/GPU backends via JAX/XLA.
"""
import os
import warnings

# Suppress XLA C++ verbose/debug warnings (e.g. Invalid stack_frame_id)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# Suppress cloud TPU transparent hugepages warning in Colab
warnings.filterwarnings("ignore", message=".*Transparent hugepages.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*cloud_tpu_init.*")

def patch_libtpu_version_check():
    """
    Patches JAX's internal libtpu semver check on Cloud TPU / Google Colab environments
    where Google uses a custom internal TFRT build string without semver numbers.
    Must be called before Pallas / Mosaic lowering is initialized.
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

patch_libtpu_version_check()

__version__ = "0.3.2"
