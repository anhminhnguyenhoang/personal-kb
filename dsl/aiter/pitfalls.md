---
name: pitfalls
description: "aiter traps that give a silent no-op or silent wrong result: dispatch misses, stale JIT, hidden build errors, preshuffle and scale-layout mismatches, fp8 dialects, FP4 on CDNA3, split-K workspace under graph capture, and in-place fused ops."
source: https://github.com/ROCm/aiter ; https://github.com/ROCm/aiter/blob/main/aiter/tuned_gemm.py ; https://github.com/ROCm/aiter/blob/main/aiter/ops/shuffle.py
---

# aiter pitfalls

aiter is a library plus a dispatcher. Most "kernel bugs" are build, dispatch, or
layout-contract problems. They rarely crash.

## Symptom → cause

| Symptom | Likely cause | Check / fix |
|---|---|---|
| Tuned CSV changes nothing | Key miss (`bias`, `gfx`, `cu_num`, quant field), wrong env var for the family, framework gate off | `AITER_LOG_TUNED_CONFIG=1`; capture live shapes; see [tuned-config-dispatch](tuned-config-dispatch.md) |
| `TypeError: 'NoneType' object is not callable` | `@compile_ops` swallowed a build error | `AITER_LOG_LEVEL=DEBUG AITER_LOG_MORE=1`; read the module's build log in the JIT dir |
| Source edit has no effect | Cached `.so` in the JIT dir | `AITER_REBUILD=1` (or `2`), or clear the JIT dir |
| CK op missing at call time | Built with `ENABLE_CK=0` | Rebuild with CK on |
| Missing symbols or odd results after an image change | Installed aiter does not match the ROCm in the image | Use the image's aiter or rebuild from source |
| First call hangs for many seconds | JIT build on first use | Warm up before timing; prebuild or keep a persistent JIT dir |
| Hot shape runs a Triton kernel | No native path or tuned row for this arch | Check trace; newer kernels often land on the newest arch first |
| Memory fault with very many heads | 32-bit stride overflow | `AITER_INT64_STRIDES=1` |

## Layout and numerics contracts (no error, wrong answer)

- **Preshuffle constraints.** Preshuffled B needs N and K divisible by the shuffle
  tile (asserted in `shuffle_weight`). The layout tile must match the consuming
  kernel; a wrong layout gives garbage, not an error.
- **The `is_shuffled` tag is a Python attribute.** Ops that build a new tensor drop it,
  and the dispatcher then picks a non-preshuffle kernel for shuffled data.
- **Scales follow weights.** For MXFP4 / e8m0, shuffle the block scales with the matching
  scale shuffle. `tl.dot_scaled` wants unshuffled scales; the raw scaled-MFMA kernels want
  shuffled ones.
- **fp8 dialect.** gfx942 MFMA uses FNUZ fp8 (`float8_e4m3fnuz`); gfx950 uses OCP
  (`float8_e4m3fn`). A dialect mismatch between producer and consumer (quant vs GEMM,
  KV store vs attention read, checkpoint vs kernel) gives about 2x error, silently.
- **FP4 needs CDNA4.** gfx942 has no FP4 matrix hardware. A4W4 paths raise or fall back,
  and vLLM's FP4 BMM flag must be off there.
- **KV-cache contract.** Store and read must agree on shuffle layout, fp8 dialect, and
  scale granularity (per-tensor vs per-block vs per-token). A padded slot is marked
  `-1` in `slot_mapping` and must be skipped.
- **Do not replicate KV for GQA.** The attention kernels broadcast KV heads in registers.
  `repeat_kv` first moves `num_q_heads/num_kv_heads` times the bytes.
- **In-place fused ops.** `fused_add_rms_norm_cu` overwrites both `input` and
  `residual_in`. The residual output must stay in the model's residual dtype, or error
  compounds across layers.
- **fp8 test tolerance is loose.** A test that passes at fp8 `atol` is not parity with
  bf16. Add a cosine or task-accuracy gate.

## Split-K workspace and graph capture (opus)

The opus split-K GEMM writes fp32 partials to a per-stream workspace and grows it with
`hipMalloc`. Allocation is illegal during stream capture. If the workspace was not sized
before capture, the graph is broken and replay writes zeros. Initialize the workspace on
**every** capture stream and run the largest shape before capture. The reduce kernel
already adds bias: adding bias again outside gives `A·Bᵀ + 2·bias`.

## Comparing variants

The same op often exists as CK, asm, HIP, and Triton (`gemm_a8w8_ck` vs `gemm_a8w8_asm`,
`pa_fwd_asm` vs `paged_attention_v1`). They differ in shape envelope and sometimes KV
layout. Compare the same variant across runs, and confirm which one ran.

## Monkey-patching

`from aiter.ops.x import f` binds `f` at import. Patching `aiter.ops.x.f` later does not
change call sites that already hold the old reference.

Sources: [ROCm/aiter](https://github.com/ROCm/aiter) (`aiter/jit/core.py`, `aiter/tuned_gemm.py`, `aiter/ops/shuffle.py`, `aiter/ops/opus/`, `aiter/ops/cache.py`) ·
[CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) (FNUZ fp8) ·
[CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) (OCP fp8, FP4)
Paths drift; verify before use.

---
Related: [architecture](architecture.md) · [tuned-config-dispatch](tuned-config-dispatch.md) · [failure-modes](../../kernel-optimization/failure-modes/failure-modes.md)
