---
name: aiter-asm-kernel-patterns
description: Reusable patterns visible when you disassemble the public ROCm/aiter hand-written asm kernels (paged attention, fused MoE, flatmm GEMM, top-k softmax, allreduce+RMSNorm) and how aiter ships, selects, and launches them.
source: https://github.com/ROCm/aiter (hsa/gfx942, hsa/gfx950 code objects; hsa/codegen.py; csrc/py_itfs_cu/asm_*.cu; csrc/include/aiter_hip_common.h) ; AMD CDNA3 and CDNA4 ISA reference guides
---

# Patterns in public aiter asm kernels

aiter ships hand-tuned kernels as prebuilt code objects (`hsa/<arch>/**/*.co`), not as
`.s` source. `llvm-objdump -d --mcpu=gfx942` on them is the best public corpus of
production CDNA asm. These are the patterns worth copying.

## 1. How aiter ships and picks them

```
 hsa/<arch>/<family>/*.csv ──hsa/codegen.py──► header: map["<arch><kernel_name>"] → {co file, tile, flags}
                                                     │
 host op (csrc/py_itfs_cu/asm_<family>.cu) ──pick──► load .co (file dir or embedded) ──► hipModuleLaunchKernel
```

- One `.co` per specialization (dtype pair, GQA ratio, tile, mask, persistent or not).
  Specializing beats one generic kernel because register maps, LDS layout and barrier
  protocol all change with the variant.
- Selection rule (MoE, GEMM): among valid tiles, pick the fewest **CU rounds**
  (`ceil(tiles / num_CUs)`), then the fewest idle CUs in the last round, then prefer the
  persistent variant.
- Kernargs go through `HIP_LAUNCH_PARAM_BUFFER_POINTER` as a packed struct. Each field sits
  in a 16-byte slot (explicit padding members), matching the `.args` offsets in the `.co`.
- Name tokens carry the config. PA: `a16w8` (16-bit Q, 8-bit KV), `1tg`/`2tg`
  (thread groups), `g8` (GQA), `blk256`, `msk1`, `_ps` (persistent). MoE: `g1u1`
  (fused gate+up), `subGU_256` (N tile of the gate/up GEMM), `blockscale` / `pertoken`.

## 2. Shared skeleton

- 256-thread workgroups (4 waves, one per SIMD), `reqd_workgroup_size` set in metadata.
- Prologue: `v_lshrrev_b32 v, 6, tid` → wave id, `v_readfirstlane_b32` → SGPR, then
  scalar branches per wave role.
- Wave-uniform data in SGPRs: buffer descriptors, block tables, per-block scales.
- Tails and paged gaps handled by buffer OOB-returns-zero, not EXEC masking.

## 3. Paged-attention decode

| Step | Instruction pattern |
|---|---|
| QK | `v_mfma_f32_16x16x32_fp8_fp8` for fp8 KV; bf16 variants use 16x16x16 bf16 |
| Per-block KV scale | one `s_load_dword` per KV block; `s_cmp` + `s_cselect` zeroes the offset past the end |
| Causal mask | after the MFMA: `v_cmp` then `v_cndmask_b32` to a `-inf` constant (`0xff800000`) held in a VGPR. No branch in the loop |
| Row max | `v_max3_f32` chain (3 inputs per op), then LDS + barrier across waves |
| exp | `v_fma_f32` (scale·S − max) then `v_exp_f32` (2ˣ; log2e folded into the scale) |
| Prefetch | deep `s_waitcnt vmcnt(8..12)` keeps many KV loads in flight |

## 4. flatmm GEMM (small M, large N)

```asm
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[0:1], a[64:65], 0        ; C=0 literal starts the tile
buffer_load_dwordx4         a[16:19], v4, s[20:23], 0 offen      ; next K stripe straight into AGPRs
v_mfma_f32_16x16x32_fp8_fp8 v[56:59], a[2:3], a[66:67], v[56:59]
...                                                              ; one load per ~4 MFMAs
```

- Operands live in **AGPRs** (loaded directly by `buffer_load`), accumulators in **VGPRs**.
  The reverse of the textbook split; the epilogue reads results with no `v_accvgpr_read`.
- Loads interleaved one per few MFMAs, `vmcnt(16)` at loop top: software pipelining with
  no LDS for the streamed operand.
- Scales applied with `v_mul_f32` / `v_fmac_f32` slotted between MFMAs, which fill
  otherwise idle VALU issue cycles.
- Epilogue: `v_cvt_f16_f32` + `v_pack_b32_f16`, `global_store_dwordx2`.

## 5. Fused MoE (gate+up → act → down in one dispatch)

- Input rows come through sorted token ids, so each tile owns one expert's tokens.
- Gate and up accumulators sit in AGPRs (100+ AGPRs in these kernels).
- Per-block scales go HBM → LDS with `buffer_load_dword ... lds`, advancing `m0` between loads.
- SiLU in 5 VALU + 1 trans op:
  `v_mul_f32 t, -g, log2e` → `v_exp_f32` → `v_add_f32 t, t, 1.0` → `v_rcp_f32` → `v_mul_f32` (× g) → `v_mul_f32` (× up).
- Output scatter: `global_atomic_pk_add_bf16` adds two bf16 per lane into the token's row,
  already weighted by the routing weight. No separate reduction kernel.
- Variants trade LDS for barriers: sharing one LDS buffer between two experts cuts LDS by
  about 3× but adds barriers and MFMAs per tile.

## 6. Kernels with no MFMA

- **Top-k softmax:** zero LDS, ~24 VGPRs. Lane-local exp + sum, then DPP butterflies
  (`quad_perm:[1,0,3,2]`, `quad_perm:[2,3,0,1]`, `row_shr:4`, `row_shr:8`,
  `row_bcast:15`, `row_bcast:31`). Arg-max: `v_readlane_b32` the max, `v_cmp_eq_f32` per
  slot, `s_ff1_i32_b64` for the winning lane; repeat k times.
- **Allreduce + RMSNorm:** `v_pk_fma_f32 v[38:39], v[4:5], v[4:5], v[38:39]` squares two
  values per issue; DPP `row_ror` inside the wave, LDS + barrier across 16 waves;
  `v_rsq_f32` once; system-scope atomics for the peer handshake.

## 7. What gfx950 adds that older kernels do not use yet

`v_mfma_scale_*_f8f6f4` (folds block-scale multiplies into the MFMA), `ds_read_b64_tr_*`
(transpose loads instead of perm shuffles), `v_mfma_i32_16x16x64_i8` (doubled-K int8),
`v_bitop3_b32` (three-input bitwise in one op). Check for them when porting a gfx942 kernel.

Sources: [ROCm/aiter — `hsa/`, `hsa/codegen.py`, `csrc/py_itfs_cu/asm_fmoe.cu`, `csrc/include/aiter_hip_common.h`](https://github.com/ROCm/aiter) ·
[AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [kernel-abi-and-prologue](kernel-abi-and-prologue.md) · [mfma-in-asm](mfma-in-asm.md) · [memory-instruction-choice](memory-instruction-choice.md) · [lds-patterns-asm](lds-patterns-asm.md)
