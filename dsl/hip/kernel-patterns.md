---
name: kernel-patterns
description: HIP C++ idioms for CDNA kernels — wave64 masks and reductions (DPP vs ds_bpermute), uniform values, register-pinned MFMA loops, phase-templated software pipelines, persistent and split-K grids, PyTorch-extension and ISA-check traps.
source: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html ; https://llvm.org/docs/AMDGPUUsage.html ; https://github.com/HazyResearch/HipKittens ; https://github.com/ROCm/composable_kernel ; https://arxiv.org/abs/2301.03598
---

# HIP kernel patterns (CDNA)

Only the HIP-specific *how*. The general *why* lives in [kernel-optimization](../../kernel-optimization/README.md).
Builtin details are in [intrinsics-map](intrinsics-map.md).

## 1. Wave64 surface

| CUDA habit | HIP on CDNA |
|---|---|
| 32-bit lane mask | `unsigned long long`. `*_sync` functions `static_assert` that the mask is 8 bytes. |
| `__popc(__ballot(p))` | `__popcll(__ballot(p))` |
| lane = `tid % 32` | `tid % warpSize` (64) |
| block of 32 | multiple of 64 (a 32-thread block is half a wave) |
| `__shfl` on `half` | shuffle as 32-bit and repack |

Contiguous prefix masks (`0xFF…`) are faster than masks with holes. Cross-lane builtins carry no memory fence.

## 2. Wave reductions: let the compiler pick DPP

```
 __shfl_xor loop (off=32..1)  ──►  6 × ds_bpermute_b32   (LDS crossbar, lgkmcnt waits)
 __reduce_add_sync(mask, v)   ──►  DPP row_shl/row_mirror/wave_shl + 2 × v_readlane   (no LDS)
```

Measured on gfx942 with a ROCm 7.x clang. A hand-written `__shfl_xor` butterfly goes through the LDS crossbar. HIP's `__reduce_{add,min,max}_sync` (int, float, and double overloads in recent headers) lowers to DPP.
Prefer the `__reduce_*_sync` family, or hand DPP via `__builtin_amdgcn_update_dpp`, inside hot loops that also use LDS. Check that your HIP header has the overload you need.

Block reduce = wave reduce → one LDS slot per wave → `__syncthreads()` → wave 0 reduces the slots.

## 3. Make uniform values scalar

```cpp
int row0 = __builtin_amdgcn_readfirstlane(offsets[blockIdx.x]);   // → SGPR
```

Use it for buffer descriptors, group or expert offsets, and loop bounds read from memory. If the compiler cannot prove a value is uniform, a descriptor built from it becomes a waterfall loop (`v_readfirstlane` + exec-mask loop per use).

## 4. MFMA loop with a stable accumulator

```cpp
f32x4 acc[TM][TN] = {};                                   // stable storage across the K loop
for (int k = 0; k < K; k += KT) {
  #pragma unroll
  for (...) acc[i][j] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a[i], b[j], acc[i][j], 0, 0, 0);
}
```

| Check in ISA | Good | Bad |
|---|---|---|
| inner loop | dense `v_mfma_*` | `v_accvgpr_read/write` between MFMAs |
| scratch | `.private_segment_fixed_size: 0` | nonzero = spill to HBM |
| loads | `buffer_load_dwordx4`, `ds_read_b128` | `_dword`, `ds_read_b32` |

The AGPR-copy trap with large tiles and software pipelining is tracked in LLVM issue #131954.
Fixes, cheapest first: keep the accumulator in one vector variable → `"+a"` asm constraint → explicit register pinning (as HipKittens does; see [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md)).

### Register pinning layout (full manual control)

With pinning, the code names fixed register ranges. The compiler only gets what is left over:

```
v[0 : P)        compiler-managed: addresses, loop counters
v[P : P+n)      A/B fragments, buffer 0 | buffer 1   (ping-pong in registers)
v[...]          MX scales, buffer 0 | buffer 1
a[0 : 256)      C accumulators
```

The pinned operand loads are inline-asm `ds_read_b128 v[N:N+3], addr offset:IMM` with constant N. The two register buffers swap by template parameter, not at runtime.
Cost: 256 VGPR + 256 AGPR per wave → **1 wave per SIMD**. All latency hiding must come from the instruction schedule.

## 5. Phase-templated software pipeline

Write the K-loop as a few templated phase functions. Then the prologue, steady state, and drain reuse one body each:

```
            ┌─ steady state, per K-iter (4 sub-tile phases) ───────────────────────┐
 prologue → │ MFMA(A0·B0) + ds_read B1 + gload(k+2, B)                             │
            │ MFMA(A0·B1) + ds_read A1 + gload(k+2, A)                             │
            │ MFMA(A1·B0) + ds_read A0' (other LDS buf) + gload(k+3, A)            │
            │ MFMA(A1·B1) + ds_read B0' (other LDS buf) + gload(k+3, B)            │
            └──────────────────────────────────────────────────────────────────────┘
 drain:  phase_mfma_lds_gload  →  phase_mfma_lds  →  phase_mfma_only
```

- Each phase issues one sub-tile of MFMAs and the LDS reads for the *next* phase. Global prefetch goes 2 iterations ahead.
- Wait with `vmcnt(N)`, where N = global loads still allowed in flight. Do not use `vmcnt(0)`. Use `lgkmcnt(0)` only right before the barrier that flips LDS buffers.
- Pass register indices and accumulator offsets as template parameters, so each phase compiles to straight-line code.
- Count the result: a good loop has far fewer `s_waitcnt` than MFMAs (well under 0.1 waits per MFMA).

The general pipeline rules are in [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md).

## 6. Persistent grid with an atomic tile counter

```cpp
__global__ void k(int* tile_ctr, ...) {           // grid = #CU (x blocks/CU)
  setup_once();                                   // per-WG prologue, amortized
  for (;;) {
    int t = 0;
    if (threadIdx.x == 0) t = atomicAdd(tile_ctr, 1);
    t = __shfl(t, 0);                             // or via LDS for multi-wave WGs
    if (t >= total_tiles) return;
    process(decode(t));                           // order the decode for L2 reuse
  }
}
```

- Reset the counter **before each launch** (a memset on the stream). A stale counter makes the kernel exit at once with no error.
- This pays only when per-tile setup is large or tiles are uneven. With tens of tiles it is noise (see the pipelining lessons).

## 7. Split-K plus a separate reduce kernel

- Partials in **fp32**, even if the output is bf16. For attention, store the LSE per split and merge with online-softmax rescaling.
- Round the split count to a value the reduce kernel supports. A mismatch gives silent garbage.
- Order the two kernels with a stream dependency. A spin-wait on a "done" counter deadlocks if the consumer starts first and holds the CUs.
- Reduce kernel: `b128` loads, wide workgroups (memory-bound), branch-free masked accumulate.

## 8. PyTorch extension traps

| Symptom | Cause | Fix |
|---|---|---|
| fp16/bf16/fp8 operator or conversion errors in device headers | `cpp_extension` defines `__HIP_NO_HALF_OPERATORS__` and `__HIP_NO_HALF_CONVERSIONS__` | add `-U__HIP_NO_HALF_OPERATORS__ -U__HIP_NO_HALF_CONVERSIONS__` |
| `TORCH_CHECK` dtype fails on scales | same bytes, different tag (`uint8` vs `float8_e8m0fnu`) | `.view(torch.float8_e8m0fnu)`, free |
| step slower than kernel sum | host sync in the wrapper (`.item()`, `torch.equal` on device tensors) | keep shape checks on host metadata |
| dynamic LDS > default fails to launch | missing attribute | `hipFuncSetAttribute(k, hipFuncAttributeMaxDynamicSharedMemorySize, bytes)` |

## 9. ISA check commands

```bash
hipcc --offload-arch=gfx950 -O3 --save-temps ...                      # keep .s
hipcc ... -Rpass-analysis=kernel-resource-usage                         # VGPR/AGPR/SGPR/LDS/scratch per kernel
hipcc ... --cuda-device-only -S -o k.s                                  # device asm only
```

Sources: [HIP kernel language](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html) · [HIP warp-sync header (ROCm/clr)](https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h) · [LLVM AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html) · [LLVM #131954 AGPR copies in pipelined MFMA loops](https://github.com/llvm/llvm-project/issues/131954) · [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens) · [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) · [Stream-K (arXiv 2301.03598)](https://arxiv.org/abs/2301.03598) · [PyTorch cpp_extension](https://pytorch.org/docs/stable/cpp_extension.html)
Paths drift; verify before use.

---
Related: [intrinsics-map](intrinsics-map.md) · [signal-recipes](signal-recipes.md) · [mxfp8-grouped-gemm](mxfp8-grouped-gemm.md) · [gfx942-registers-occupancy](../../hardware/amd-cdna/gfx942-registers-occupancy.md)
