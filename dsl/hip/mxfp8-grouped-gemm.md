---
name: mxfp8-grouped-gemm
description: "Design lessons for an MXFP8 grouped (MoE) GEMM in HIP on gfx950: reuse a tuned single-GEMM tile via a block→expert table, E8M0 scale preshuffle and K-tail padding, dgrad by operand-role swap, why wgrad is a different kernel, and when HIP beats Triton."
source: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ; CDNA4 ISA PDF ; https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/ ; https://github.com/ROCm/aiter ; https://github.com/ROCm/hipBLASLt
---

# MXFP8 grouped GEMM in HIP: design lessons

Problem: `G` experts, each `C_g[M_g, N] = A_g[M_g, K] · B_g[N, K]ᵀ`, with fp8 data and one E8M0 scale per 32 K-elements (OCP MX). Training needs fwd, dgrad, and wgrad.

```
             fwd                      dgrad                       wgrad
   A[M_tot,K] · B[G,N,K]ᵀ     dC[M_tot,N] · B[G,K,N]ᵀ      dCᵀ_g[N,M_g] · A_g[M_g,K]
   reduce over K (fixed)       reduce over N (fixed)        reduce over M_g (VARIABLE per expert)
   └──────── same kernel, roles swapped ───────┘            └── different kernel shape ──┘
```

## 1. Reuse the tuned single-GEMM tile; add only dispatch

The tile body (MFMA schedule, register pinning, direct-to-LDS loads, double-buffered LDS) carries the speed. Do not rewrite it per expert. Add a flat grid with a host-built lookup table:

```cpp
int g        = block_to_expert[blockIdx.x];        // precomputed, one entry per tile
int local    = blockIdx.x - tile_offset[g];
int pid_m    = local / tiles_n,  pid_n = local % tiles_n;
int m_g      = group_offs[g+1] - group_offs[g];     // rows of this expert
// A base = A + (group_offs[g] + pid_m*BM) * K ;  B base = B + g*N*K + pid_n*BN*K
// ... unchanged single-GEMM tile body, with the M bound = m_g (buffer OOB handles the tail)
```

One launch covers all experts. The M tail of each expert is free, because buffer-descriptor OOB loads return 0.

## 2. Scales: preshuffle and pad the K tail

```
 E8M0 scales [rows, K/32] uint8  ──preshuffle kernel──►  [rows/16, ceil4(K/32)/4, 16, 4] uint32
                                                          (zero-extended, MFMA-consumable 16×4 blocks)
```

- The scaled MFMA reads one scale byte per lane per 32-K block. A 16×4 block layout lets one `ds_read_b32` per fragment feed it, with four scales packed per register and chosen by `opsel`. HipKittens' public MXFP8 GEMM uses the same packing ([hipkittens-idioms](hipkittens-idioms.md)).
- **When K/32 is not a multiple of 4, pad the scale columns and zero-fill them.** E8M0 `0x00` = 2^-127, so the padded tail contributes about zero. The data tail is already zero via OOB loads. Without the pad, the last K-iteration reads uninitialised scales and gives garbage. Pass the padded stride to the GEMM.
- Scales from another producer may carry a different dtype tag (`uint8` vs `float8_e8m0fnu`). Reinterpret with a view. Do not convert.

## 3. dgrad = fwd kernel with swapped roles

Pre-transpose `B` to `[G, K, N]` once. The weight quantizer can emit this layout directly. Then call the fwd kernel with `A ← dC`, `B ← Bᵀ`, reduction length `N`.
No new kernel. dgrad often gains *more* than fwd over a generic kernel, because a longer reduction amortizes per-iteration overhead.

## 4. wgrad is a different problem

The reduction runs over `M_g`, which varies per expert, so the fwd tile does not fit:

| Approach | Outcome pattern |
|---|---|
| Per-expert loop of fwd launches | Launch overhead × G dominates. Far slower. |
| Balanced MoE only: stack experts into one fwd launch (`[G·N, M_g]` × `[G, K, M_g]`) | One launch, but the fwd-tuned tile runs at a fraction of its efficiency (short K, tile-unaligned N) |
| Per-expert padded fallback | Wins only on degenerate routing (few non-empty experts) |
| Dedicated variable-K kernel (jagged scales, LDS transpose) | Expected parity, not a win: the generic wgrad is usually HBM-bound |

The usual result is a hybrid: hand-written HIP for fwd and dgrad, a generic (e.g. Triton) kernel for wgrad.

## 5. Where the time actually went

- **Profile the whole training step first.** A framework `.permute().contiguous()` on fp8 can run far below HBM peak. A tiled-LDS transpose kernel fixed most of one "slow kernel" gap with no GEMM change.
- **Remove host syncs from the wrapper.** A device-side equality check on group offsets forced a CPU-GPU sync every call.
- **Hoist weight quantization** to once per optimizer step, not once per forward. Weights do not change between micro-batches.
- **Fusing a permute into a quant kernel can lose.** If the compiler's store layout fights the permuted store pattern, the fused kernel is slower than the two separate ones.

## 6. Why HIP beat the generic kernel, and its ceiling

| Signal | Hand HIP tile | Generic compiled kernel |
|---|---|---|
| `s_waitcnt` per MFMA | ≪ 0.1 | close to 1 |
| LDS-wait / LDS-instr ratio | low | very high (small scale reads wait on LDS) |
| `ds_write` in loop | 0 (direct-to-LDS) | present |
| VGPR / occupancy | 256 + AGPR, 1 wave/SIMD | lower VGPR, more waves |

The ceiling: about 144 KB of LDS per WG on a 160 KB CU forces occupancy 1, so MFMA utilization stalls near half. Ways forward: a smaller LDS footprint, a 32×32×64 atom (half the MFMA issues), or a persistent kernel across experts.
Decision rule: port to HIP only when the generic kernel's stall is inside the CU (LDS or waitcnt). See [signal-recipes](signal-recipes.md).

Verify specifics against the primary source.

Sources: [OCP Microscaling Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) · [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) · [AMD lab notes: Matrix Cores (GPUOpen)](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/) · [CDNA4 whitepaper (160 KB LDS)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf) · [HipKittens MXFP8 GEMM kernels](https://github.com/HazyResearch/HipKittens/tree/main/kernels/cdna4/gemm/mxfp8) · [ROCm/aiter](https://github.com/ROCm/aiter) · [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt) · [PyTorch float8_e8m0fnu dtype](https://pytorch.org/docs/stable/tensor_attributes.html)

---
Related: [kernel-patterns](kernel-patterns.md) · [intrinsics-map](intrinsics-map.md) · [varlen-idioms](../flydsl/varlen-idioms.md) · [algorithmic-mapping](../../kernel-optimization/levers/algorithmic-mapping.md)
