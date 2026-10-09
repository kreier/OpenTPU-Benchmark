# OpenTPU-Benchmark Roadmap

This roadmap outlines the milestones for porting the OpenCL-Benchmark suite to Google Cloud TPUs (with primary focus on `TPU v5e-1` and scaling to `v4`, `v5p`, and `v6e`).

---

## 🎯 High-Level Status Overview

| Phase | Description | Status | Target Hardware |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Agent Infrastructure & Development Harness | ✅ Completed | CPU / Local Dev & Cloud TPU |
| **Phase 2** | FP32 Compute Benchmark (Vector ALU & Matrix GEMM) | ✅ Completed | TPU v5e-1 (VPU + MXU) |
| **Phase 3** | BF16 & FP16 Precision Benchmarks | ✅ Completed | TPU v5e-1 (197 TFLOPS MXU Peak) |
| **Phase 4** | Integer Compute Benchmarks (INT32 & INT8 dp4a) | ✅ Completed | TPU v5e-1 (INT8 MXU Systolic) |
| **Phase 5** | HBM Memory Bandwidth (Coalesced Read & Write) | ✅ Completed | TPU v5e-1 (816 GB/s HBM2e) |
| **Phase 6** | Host PCIe & Inter-Chip Interconnect (ICI) | ⏳ Up Next | TPU v5e-1 to v5e-4/8 Pods |
| **Phase 7** | Low-Level Custom Kernels with Pallas / Mosaic | 📅 Scheduled | TPU VectorCore / VMEM |
| **Phase 8** | Unified Comparison Database & CI Dashboard | 📅 Scheduled | All TPU Generations |

---

## 📋 Detailed Phase Breakdown

### Phase 1: Agent Infrastructure & Python/JAX Harness (Completed ✅)
- [x] Standardize Python virtual environment & packaging (`pyproject.toml`, `requirements.txt`, `requirements-tpu.txt`).
- [x] Create `AGENTS.md` and agent workflows.
- [x] Implement TPU device info discovery module (`opentpu/device_info.py`) with theoretical peak specs for TPU v5e, v4, v5p, v6e, and CPU fallback.
- [x] Implement OpenCL-Benchmark ASCII frame reporter with `fraction(...)` peak ratio calculations (`opentpu/reporter.py`).
- [x] Add `--json` flag for machine-readable output for automated agent evaluation.
- [x] Add automated unit test suite with `pytest`.
- [x] Create deployment scripts (`setup_tpu.sh`, `make_tpu.sh`, `run_tpu.py`).

---

### Phase 2: FP32 Compute Benchmark Port (Completed ✅)
- [x] **FP32 Vector ALU FMA Microbenchmark**:
  - Direct 1:1 port of OpenCL `kernel_float`: 512 unrolled loop iterations with 2 FMAs per iteration (2048 FLOPs per element).
  - JIT-compiled with `lax.fori_loop` targeting TPU Vector Processing Unit (VPU).
  - Accurate barrier timing using `res.block_until_ready()`.
- [x] **FP32 Matrix Multiplication (GEMM)**:
  - Benchmark FP32 dense GEMM ($2 \cdot N^3$ FLOPs) measuring matrix throughput.
- [x] Verification on CPU emulation and ready for execution on `v5e-1`.

---

### Phase 3: BF16 & FP16 Precision Benchmarks (Completed ✅)
*Goal: Measure the flagship compute capability of TPU v5e-1 (197.0 TFLOPs/s).*
- [x] **BF16 Matrix GEMM (MXU Peak)**:
  - Benchmark $N \times N$ matrix multiplications in `bfloat16`.
  - Validate against the 197 TFLOPs/s theoretical maximum of TPU v5e's 4 MXUs.
- [x] **BF16 Vector ALU**:
  - Elementwise FMA arithmetic loop in `bfloat16` on VPU.
- [x] **FP16 Matrix & Vector**:
  - Compatibility testing with IEEE `float16` on TPU v5e.

---

### Phase 4: Integer Compute Benchmarks (Completed ✅)
*Goal: Benchmark TPU integer ALU and quantized inference performance.*
- [x] **INT32 ALU Compute**:
  - Elementwise integer arithmetic loop ($a \cdot b + c$) on VPU.
- [x] **INT8 Matrix / Dot Product (dp4a equivalent)**:
  - Port OpenCL `kernel_char` (which uses `dp4a`) to TPU INT8 quantized matrix multiplication and vector dot product.
  - TPU v5e MXU supports INT8 at up to 197 TOPs/s.

---

### Phase 5: HBM Memory Bandwidth Benchmarks (Completed ✅)
*Goal: Saturate and measure the 816 GB/s HBM2e memory bus on TPU v5e.*
- [x] **Sequential / Coalesced Read & Write**:
  - Pure memory streaming kernels measuring read, write, and copy bandwidth in GB/s.


---

### Phase 6: Host PCIe & Inter-Chip Interconnect (ICI)
*Goal: Measure communication bottlenecks across host and multi-chip pods.*
- [ ] **Host-to-Device (PCIe) Bandwidth**:
  - Host RAM to TPU HBM (`jax.device_put`) and TPU HBM to Host (`jax.device_get`) transfer speeds.
- [ ] **Inter-Chip Interconnect (ICI)**:
  - Measure TPU-to-TPU direct peer bandwidth on multi-chip slices (e.g. `v5e-4`, `v5e-8`) using collective communication primitives (`lax.all_gather`, `lax.psum`).

---

### Phase 7: Low-Level Custom Kernels with Pallas / Mosaic
*Goal: Hardware-level control bypassing high-level XLA heuristics.*
- [ ] Implement custom TPU kernels using **Pallas** (`jax.experimental.pallas`).
- [ ] Direct management of VMEM (Vector Memory), SMEM (Scalar Memory), and asynchronous DMA transfers.
- [ ] Direct invocation of Mosaic MLIR pipeline for exact instruction scheduling.

---

### Phase 8: Unified Comparison Database & CI
- [ ] Store hardware results in a benchmark database (`results/results.json`).
- [ ] Generate comparative markdown reports against Nvidia GPUs and Apple Silicon from the original OpenCL benchmark.
