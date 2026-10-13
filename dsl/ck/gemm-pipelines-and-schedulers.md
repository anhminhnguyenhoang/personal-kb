---
name: gemm-pipelines-and-schedulers
description: "CK GEMM parameter stack and its constraints, the pipeline families (classic v1–v5, CK-Tile mem/compv3–v6/comp_async/eight_waves), Intrawave vs Interwave scheduling, compile-time sched_group_barrier recipes, prefetch-stage LDS budgets, grid mapping, split-K and Stream-K."
source: https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html ; https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/gemm/pipeline ; https://github.com/ROCm/composable_kernel/issues/1727
---

# CK GEMM: pipelines, schedulers, knobs

## 1. Parameter stack

```
Block tile   MPerBlock × NPerBlock × KPerBlock        (e.g. 256×256×64, BlockSize 256 = 4 waves)
  Wave map   MWaves × NWaves                          MWaves·NWaves·64 = BlockSize
    Repeats  MRepeat × NRepeat  (= MXdlPerWave × NXdlPerWave)  MFMA tiles per wave
      MFMA   MPerXDL × NPerXDL × KPack                (16×16 or 32×32)
        Load AK1 / BK1                                vector width along K per lane
```

Constraints:
- `MPerBlock = MPerXDL · MRepeat · MWaves` (same for N).
- `KPack` divisible by the MFMA K; `KPerBlock` a multiple of `AK1 ×` MFMA K.
- `AK1/BK1` × element size ≥ 128 bit (bf16 → 8, fp8 → 16) so each lane emits
  `buffer_load_dwordx4`. Pointers must be aligned to it or the instance is rejected.
- Grid = `ceil(M/MPerBlock)·ceil(N/NPerBlock)`; aim for a multiple of the CU count.

Ranked knobs (tune top-down): block tile → `KPerBlock` → pipeline version + scheduler →
MFMA shape → wave map → load width → padding spec → `KBatch` (split-K) → C-store width.

## 2. Pipeline names (CK-Tile)

`gemm_pipeline_ag_bg_cr_*` = **A** from **g**lobal, **B** from **g**lobal, **C** in
**r**egisters.

| Pipeline | Prefetch | LDS buffers | Notes |
|---|---|---|---|
| `agmem_bgmem_creg_v1` | none | 1 | teaching / lowest VGPR |
| `mem` | deep global prefetch | 1 | memory-bound shapes |
| `comp_v3` | 2 | 1 + register prefetch | compute-bound workhorse |
| `comp_v4` | 2 | 2 (ping-pong) | adds DS-write interleave |
| `comp_v5` | 1 | 1 | always-hot loop, no tail branch |
| `comp_v6` | 3 | 1, 2 global buffers | hot loop unrolled ×2 |
| `comp_async` | 2 | 2 | `buffer_load … lds` direct to LDS, no VGPR staging |
| `comp_async_eight_waves` | 2 | 2 | 8 waves per WG, preshuffled B |

Classic CK names the same ideas `BlockGemmPipelineVersion::v1…v5`: v1 single buffer,
v2 explicit overlap with two LDS arrays, **v3** compiler-overlapped 2-stage prefetch
(gfx94x default), v4 deeper ping-pong, v5 persistent / async input.

**Tail handling**: the K-loop count mod prefetch depth selects a `TailNumber`
(Odd/Even, One/Two/Three, Empty/Full) at launch. Each tail is a separate compiled path,
so there is no runtime branch in the hot loop.

## 3. Scheduler: Intrawave vs Interwave

| | Intrawave | Interwave |
|---|---|---|
| Latency hidden by | interleaving loads and MFMAs inside one wave | switching between waves |
| Mechanism | `sched_group_barrier` sequence per iteration + `s_setprio` | default compiler schedule, more waves |
| Best for | compute-bound, large tiles (prefill) | memory-bound, skinny M, low occupancy, or when Intrawave spills |

**Recipe pattern** (comp_async, per buffer-load issue):
`MFMA(1) → DS_READ(1) → MFMA(1) → VMEM_READ(1) → MFMA(n − 2)`.
Masks: `0x008` MFMA, `0x020` VMEM read, `0x100` DS read, `0x200` DS write.

Rules:
- Every count is a `constexpr` function of tile geometry (MFMA count per stage ÷ load
  count). Do not hand-tune a literal; change the tile and the recipe follows.
- The count is **how many instructions of that class go in the group**, not "wait for N".
- DS-read rate scales with MFMA length: shorter MFMAs (fp8) get fewer reads per MFMA slot.
- `sched_barrier(0)` is a hard fence. Use it to stop a load (e.g. bias) from drifting into
  the GEMM, not as a tuning knob.
- Use the sync that matches the hazard: after async global→LDS wait on `vmcnt`
  (`block_sync_lds_direct_load`); after DS writes wait on `lgkmcnt` (`block_sync_lds`).

## 4. LDS budget per prefetch stage (derived)

```
LDS per stage = (MPerBlock + NPerBlock) × KPerBlock × bytes
256×256×64 bf16 → (256+256)·64·2 = 64 KB
```
- gfx942 has 64 KB LDS per CU: this tile fills it with one stage. That is why the 256²
  workhorse prefetches the next K-slice into **registers**, not a second LDS buffer.
- gfx950 has 160 KB: two LDS stages of the same tile fit. Deeper LDS pipelines become
  possible on CDNA4 only.
- More stages or `GlobalBufferNum = 2` also cost VGPRs. Check spills before keeping them.

## 5. Grid mapping, split-K, Stream-K

- **Block-to-tile map** (`BlockToCTileMap_M00_N0_M01`): groups `M01` consecutive block ids
  down M before stepping N, so neighbors share B in L2. `M01 ≈ 8` is typical; it is
  divide-free adaptor math. It is not XCD-aware; see
  [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md).
- **Split-K** (`KBatch ≥ 2`): K is split across blocks, partial C merged with atomic add
  (epilogue memory op `update` instead of `set`). Wins only when it fills idle CUs —
  skinny-M decode. It adds HBM traffic and is not bitwise deterministic.
- **Stream-K**: first CTAs own whole tiles (data-parallel); the remainder split K-iterations
  evenly by cumulative iteration count and reduce through a workspace (atomic or per-CTA
  slots + flags). Same warp-GEMM core; only the partitioner differs. Workspace accesses use
  system-scope non-temporal coherence on gfx94x/gfx950.
- **Decode shape** (M ≪ N, K): small M tile (16/32), split-K, Interwave, 16×16 MFMA; or a
  flatmm-style kernel ([fused-moe-and-flatmm](fused-moe-and-flatmm.md)).

## 6. Reference point

bf16 4096³ RCR on MI300X, classic universal v3: tile 256×256×64, wave tile 32×32, wave map
4×4, load vector 8×8, Intrawave, prefetch 2 → 0.223 ms, 615 TFLOP/s (Issue #1727). Use as
a sanity anchor, not a target; pinned instances drift across CK releases.

Sources: [A Block GEMM on MI300 (CK docs)](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html) ·
[ck_tile gemm pipelines](https://github.com/ROCm/composable_kernel/tree/develop/include/ck_tile/ops/gemm/pipeline) ·
[Issue #1727](https://github.com/ROCm/composable_kernel/issues/1727) ·
[MI300X workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) ·
[Stream-K paper (arXiv 2301.03598)](https://arxiv.org/abs/2301.03598) ·
[LLVM AMDGPU usage (sched_group_barrier)](https://llvm.org/docs/AMDGPUUsage.html)
Paths drift; verify before use. Verify specifics against the primary source.

---
Related: [tile-distribution](tile-distribution.md) · [epilogues-and-quant-gemm](epilogues-and-quant-gemm.md) · [pipelining-scheduling](../../kernel-optimization/levers/pipelining-scheduling.md)
