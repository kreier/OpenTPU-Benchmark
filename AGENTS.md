# AGENTS.md — Agent Developer Guide for OpenTPU-Benchmark

This repository is designed to be agent-friendly. Whether you are an autonomous coding agent, a pair-programming assistant, or a developer running automated workflows, this document contains all the context, architecture patterns, and conventions necessary to contribute effectively.

---

## 1. Project Mission & Overview

**OpenTPU-Benchmark** is the Google TPU port of the acclaimed [OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark).

While GPUs execute arbitrary C/C++ OpenCL/CUDA kernels via SIMT (Single Instruction Multiple Threads), **Google TPUs** (Tensor Processing Units) operate under a distinct architectural paradigm:
* **VLIW + Systolic Array Architecture**:
  * **VPU (Vector Processing Unit)**: Handles elementwise floating-point and integer vector math (such as FMA loops, activations, transcendental functions).
  * **MXU (Matrix Multiply Unit)**: Specialized 2D systolic arrays optimized for dense matrix multiplication (GEMM) in BF16, INT8, and FP32.
  * **HBM Subsystem**: High Bandwidth Memory (e.g., HBM2e on TPU v5e providing 816 GB/s).
  * **Inter-Chip Interconnect (ICI)**: High-speed ring/torus network interconnecting TPU chips directly without host CPU routing.
* **Software Stack**: TPUs run via **XLA (Accelerated Linear Algebra)** and **PJRT** (Platform-independent JIT Runtime). High-performance TPU programming is done using **JAX** (`@jax.jit`, `jax.lax`) and **Pallas** (`jax.experimental.pallas`, compiling down to Google Mosaic MLIR dialect).

---

## 2. Target TPU Hardware: Google Cloud TPU v5e (`v5e-1`)

The baseline target device is the **Google Cloud TPU v5e-1** (Viperfish architecture):
* **Chip Count / TensorCores**: 1 Chip / 1 TensorCore per `v5e-1`.
* **Peak Compute**:
  * **BF16 / INT8 (MXU)**: **197.0 TFLOPs/s** (Systolic matrix engine).
  * **FP32 (Vector ALU / VPU)**: **~12.3 TFLOPs/s** (Vector elementwise ALU).
  * **FP64**: Not hardware accelerated (emulated / negligible).
* **Memory Subsystem**:
  * **HBM Capacity**: 16 GB HBM2e.
  * **Peak HBM Bandwidth**: 816 GB/s.
* **Interconnect**: 400 Gbps ICI per chip.

---

## 3. Directory Structure

```
OpenTPU-Benchmark/
├── AGENTS.md                 # This file: Guide and conventions for AI agents
├── ROADMAP.md                # Development roadmap across precisions and memory tests
├── README.md                 # User-facing documentation
├── pyproject.toml            # Python packaging configuration
├── requirements.txt          # CPU development & testing dependencies
├── requirements-tpu.txt      # Google Cloud TPU VM dependencies (jax[tpu])
├── setup_tpu.sh              # One-step bootstrap script for Cloud TPU VMs
├── make_tpu.sh               # Benchmark execution script (parity with make.sh)
├── run_tpu.py                # Python root entrypoint
├── pytest.ini                # Pytest configuration
├── opentpu/                  # Core Python package
│   ├── __init__.py
│   ├── device_info.py        # Platform detection & TPU hardware specs database
│   ├── reporter.py           # Authentic OpenCL-Benchmark ASCII table formatter
│   ├── main.py               # CLI entrypoint supporting human ASCII & JSON output
│   └── benchmarks/
│       ├── __init__.py
│       └── fp32.py           # Ported FP32 Vector ALU (FMA) & Matrix GEMM tests
├── tests/                    # Pytest test suite
│   └── test_benchmarks.py
└── src/                      # Original C++ OpenCL benchmark (preserved for reference)
```

---

## 4. How Agents Should Run & Verify Code

### 4.1 Running Tests
Always run tests before concluding any changes:
```bash
# Using local environment
.venv/bin/pytest -v
# Or
pytest -v
```

### 4.2 Running Benchmarks Locally (CPU Emulation)
Even without a physical TPU attached, the entire codebase runs transparently in CPU fallback mode via JAX:
```bash
./make_tpu.sh --iterations 5 --warmup 2 --elements 1048576 --matrix-dim 1024
```

### 4.3 Automated JSON Output for Agent Verification
Agents can pass `--json` to inspect structured telemetry without regex-parsing ASCII tables:
```bash
./make_tpu.sh --json
```

---

## 5. Porting & Benchmarking Methodology

When adding new precisions (BF16, FP16, INT8, INT32) or memory bandwidth tests:

1. **Vector ALU vs Matrix Systolic Array**:
   * OpenCL microbenchmarks measure per-thread ALU loops (e.g. 512 iterations of double FMA = 2048 FLOPs/element). On TPUs, this maps to the **Vector Processing Unit (VPU)**.
   * TPUs achieve their peak advertised TFLOPS (e.g., 197 TFLOPS on v5e) in the **Matrix Multiply Unit (MXU)**.
   * Provide both benchmarks:
     - `Vector ALU`: tests elementwise compute density on the VPU.
     - `Matrix GEMM`: tests systolic array throughput on the MXU.

2. **Precise Timing & Asynchronous Execution**:
   * JAX / XLA dispatches asynchronously. Always call `.block_until_ready()` on the output buffer before capturing `time.perf_counter()`.
   * Always run warmup passes to exclude XLA JIT compilation time from benchmark measurements.

3. **Compiler Elision Prevention**:
   * XLA aggressively optimizes away unused buffers or constant expressions. Ensure output buffers are used or returned and inputs are dynamic/device-placed.

4. **Formatting Parity**:
   * Output rows must match OpenCL-Benchmark's ASCII frame:
     ```
     | FP32   Compute   (float , fma  )                      12.300 TFLOPs/s ( 1x ) |
     ```
   * Maintain the `fraction(...)` ratio display comparing measured throughput to theoretical hardware peak.
