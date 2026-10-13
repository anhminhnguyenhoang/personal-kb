---
name: operator-recipes
description: Reusable FlyDSL recipes for three operator shapes — preshuffle GEMM (wave mapping, tile algebra, data paths, MFMA and epilogue choice, ISA sanity count), two-stage fused MoE (sorted-M grouping, gate modes, split-K, atomic vs reduce combine), and the wave64 block reduction.
source: https://github.com/ROCm/FlyDSL (kernels/gemm/preshuffle_gemm.py, kernels/common/mma/mfma_preshuffle_pipeline.py, kernels/moe/, kernels/norm/, docs/kernel_tuning_guide.md, .claude/skills/gemm-optimization/SKILL.md); https://github.com/ROCm/aiter (aiter/ops/flydsl/)
---

# FlyDSL operator recipes

## A. Preshuffle GEMM  `C[M,N] = A[M,K] · B[N,K]ᵀ`

### Wave and lane mapping (256 threads, 16×16 MFMA)

```
 block tile (tile_m × tile_n), K in steps of tile_k
 ┌──────── tile_n ────────┐
 │ wave0 │ wave1 │ wave2 │ wave3 │   wave_id = tid // 64  → N quarter (n_per_wave = tile_n/4)
 └───────────────────────┘
 inside a wave: lane % 16 → N column of the MFMA ; lane // 16 → 4-row group in M

 m_repeat  = tile_m / 16           MFMA rows per wave
 num_acc_n = n_per_wave / 16       MFMA columns per wave
 k_steps   = tile_k / K_mfma       MFMAs along K per tile
 MFMA/tile/wave = k_steps · m_repeat · num_acc_n
 acc VGPRs      = m_repeat · num_acc_n · 4          (16×16 f32 acc = 4 regs/lane)
```

Example: 64×256×128 fp8, `K_mfma = 32` → 4 · 4 · 4 = 64 MFMAs per tile, 64 acc registers.

### Constraints

- `tile_m % 16 == 0`, `tile_n % 64 == 0`, `tile_k · elem_bytes % 64 == 0`. For a preshuffled B,
  `tile_k` must divide K.
- LDS ≈ `stages · tile_m · tile_k · elem_bytes` (+ `tile_m · tile_n · 2` if CShuffle).
  Limits: 64 KB on gfx942, 160 KB on gfx950.

### Data paths

```
 A : global ─buffer_load_dwordx4→ VGPR ─ds_write→ LDS (XOR-16B swizzle, 2 stages) ─ds_read→ MFMA
       └─ or DMA global→LDS (async_copy; frees ~4 VGPR per 16-B load)
 B : host-preshuffled (N/16, K/kblk, 4, 16, kpack) ─buffer_load_dwordx4→ VGPR → MFMA  (no LDS)
       "16" = 16 lanes of one MFMA column ; "4" = 4 dwords per lane ; one load = one operand pack
```

- **A0 prefetch:** right after the iteration barrier, read the first A pack from the fresh LDS
  buffer. Its `ds_read` latency then overlaps the next VMEM issue.
- The schedule is a sequence of `sched_vmem(1)` · `sched_mfma(g)` · `sched_dsrd(1)` · `sched_mfma(g)`
  groups, with `sched_dswr` at the tail
  ([hot-loop-scheduling](hot-loop-scheduling.md)). The per-tile "preload" counts
  (`dsrd`, `vmem`) come from a tuned table keyed by `(tile_m, tile_n, tile_k)`. A new
  tile shape needs its own tuning; the default is `(0, 0)`.
- In the current library kernel, `use_async_copy` is gfx950-only and `lds_stage ∈ {1, 2}`.

### MFMA choice

| dtype | gfx942 | gfx950 |
|---|---|---|
| fp8 / i8 | `16x16x32` | `16x16x32`; also the `f8f6f4` family (`16x16x128`, `32x32x64`) |
| f16 / bf16 | `16x16x16`, `32x32x8` | **`16x16x32`, `32x32x16`** — double K, half the MFMA count |
| MXFP4 / FP6 | — | `mfma_scale_f32_16x16x128_f8f6f4` + e8m0 block scales |

### Epilogue

- **Direct:** each lane stores its accumulator rows (`row = m0 + mi·16 + (lane//16)·4 + ii`,
  `col = n0 + wave·n_per_wave + ni·16 + lane%16`). It needs no LDS. Stores can be narrow.
- **CShuffle:** write the accumulators to LDS row-major → barrier → re-read as an (8 × 32)-lane
  grid → `buffer_store_dwordx2`, with 4-element vectors when `tile_n % 128 == 0`, else 2. Use it when
  `tile_n ≥ 128` and direct stores show low bandwidth.

### ISA sanity count

In the dumped `final_isa.s`, count the hot loop between two `s_barrier`s:
`v_mfma`, `buffer_load`, `ds_read`, `ds_write`, `s_barrier`. The MFMA count must equal the
formula above. Barrier and memory-op counts must not be higher than in the reference kernel (aiter / CK).
If the MFMA share is below about 30 % of the loop, there is too much overhead. Check the schedule and
address math.

### Library knobs worth knowing (aiter `flydsl_hgemm`)

`split_k` (deep K, small M·N), `stages`, `b_to_lds` (stage B through LDS instead of direct),
`c_to_lds` (CShuffle), `async_copy`, `persistent_n_tiles`, `waves_per_eu`,
`kernel_family = SMALL_M` (decode). The scaled / A4W4 paths go through the MoE/quant
kernels, not hgemm.

## B. Two-stage fused MoE

```
 tokens ─ top-k routing ─► moe_sorting: rows grouped by expert, each run padded to tile_m
                               │ sorted_token_ids, sorted_expert_ids, num_valid_ids
                               ▼
 stage 1  X_e · W1_eᵀ   W1 = [E, 2·inter, H]  (gate ‖ up)
          A = act(G) ⊙ U   [× routed weight if doweight_stage1]
          → bf16, or fused requant → fp8/fp4 + block scales in sorted tile order
                               ▼
 stage 2  A_e · W2_eᵀ   W2 = [E, H, inter]   [× routed weight]
          combine over top-k:  atomic  (into a zeroed out)   |   reduce (per-slot partials → sum)
```

- Only **M** is grouped. N and K are fixed per stage. One kernel covers all experts through the
  expert id of each sorted M-block. The moe_sorting block size must equal the stage `tile_m`.
- **Gate modes (stage 1):** `separated` = two B-tile streams (gate and up). `interleave` = weight
  rows alternate gate/up, so one stream gives both halves to neighbouring N sub-tiles.
  `mock_gate_only` = one stream over `[0, 2·inter)`; it needs split-K.
- **Split-K (`k_batch > 1`):** stage 1 writes fp32 partials. A separate fused
  `silu_and_mul + quant` kernel sums them, applies the activation, and quantizes.
- **Combine (stage 2):**

| mode / out dtype | speed | risk |
|---|---|---|
| atomic, `f16` (half2 atomics) | fastest | **overflows to ±inf** on bf16-range activations |
| atomic, `bf16` | fast | `global_atomic_pk_add_bf16` on gfx942, buffer atomics on gfx950 |
| atomic, `f32` | slower | no overflow |
| reduce (no atomics) | extra pass | deterministic; f16/bf16 only |

- Weights (and fp4 scales) must use the **exact** preshuffle the kernel decodes. A wrong layout
  gives silent garbage, not an error. MXFP4 uses 1×32 e8m0 scales, so `inter_dim % 32 == 0`.
  fp4 MFMA needs gfx950.
- For a lossy fp4 path, validate with a fraction-close metric against a dequantized reference,
  then with task accuracy. Do not expect bit parity.
- Other knobs: a persistent round-robin grid when there are many M-blocks, `xcd_swizzle`
  ([chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md)), and a
  non-temporal B-load hint.

## C. Block reduction (norm / softmax building block)

```
 per lane : reduce the vector in registers  (range_constexpr over VEC)
 per wave : xor-shuffle 32,16,8,4,2,1        gpu.ShuffleOp(v, sh, 64, mode="xor")
 per block: lane0 of each wave → LDS[wave] ; barrier
            wave0 reads NUM_WAVES partials (others pad with identity) → shuffle-reduce
            lane0 → LDS[0] ; barrier ; everyone reads LDS[0]
```

- Two barriers per reduction. Fuse max and sum into one pass (online softmax) to halve them.
- Padding lanes must hold the **identity** (`-inf` for max, `0` for sum).
- Library versions live with the norm kernels (`kernels/norm/*`, `kernels/common/kernels_common.py`).

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `kernels/gemm/preshuffle_gemm.py` (`compile_preshuffle_gemm`, `_TILE_PRELOAD_TABLE`), `kernels/common/mma/mfma_preshuffle_pipeline.py` (`make_preshuffle_b_layout`, `xcd_remap_bx_by`), `kernels/moe/moe_common.py` (`GateMode`), `kernels/moe/moe_gemm_2stage/gemm1.py`, `gemm2.py` (`compile_moe_gemm2` docstring), `kernels/moe/mxfp_moe/`, `kernels/norm/`, `docs/kernel_tuning_guide.md`, `docs/prebuilt_kernels_guide.md`, `.claude/skills/gemm-optimization/SKILL.md`; [ROCm/aiter](https://github.com/ROCm/aiter) — `aiter/ops/flydsl/moe_kernels.py`, `aiter/ops/flydsl/gemm_kernels.py`; [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) (MFMA shapes and lane maps); [AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf).
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [lds-and-prefetch-idioms](lds-and-prefetch-idioms.md) · [hot-loop-scheduling](hot-loop-scheduling.md) · [epilogue](../../kernel-optimization/levers/epilogue.md) · [matrix-instructions](../../kernel-optimization/levers/matrix-instructions.md)
