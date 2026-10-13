---
name: signal-recipes
description: Profiler signal → HIP code change table for CDNA kernels (LDS conflicts, LDS waits, waitcnt-bound loops, MFMA issue limits, fp8 upcasts, spills, occupancy-1, small grids, host syncs), with gotchas and an apply order.
source: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/ ; https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/ ; https://llvm.org/docs/AMDGPUUsage.html ; https://arxiv.org/abs/2511.08083
---

# Signal → HIP recipe

Classify the kernel first ([bottleneck-classification](../../kernel-optimization/method/bottleneck-classification.md)). Then pick the row whose signal is largest.
Thresholds are starting heuristics, not laws. Re-profile after each change. If the target signal did not move, revert: the diagnosis was wrong.

```
        rocprofv3 PMC / PC sampling / ISA
                     │
     ┌───────────────┼──────────────────┬───────────────────┐
   LDS-side       issue-side        memory-side        grid/host-side
  (rows 1-2)     (rows 3-6)         (rows 7-8)          (rows 9-12)
```

| # | Signal | HIP change | Gotcha |
|---|---|---|---|
| 1 | `SQ_LDS_BANK_CONFLICT` / LDS active cycles above a few % | XOR-swizzle the LDS layout on **both** write and read sides. On gfx950 read the transposed operand with `ds_read_tr16_b64` / `ds_read_tr8_b64`. With direct-to-LDS, put the swizzle into the global address. | A read-only swizzle just moves the conflict. Each `ds_read` width has its own bank and phase pattern. |
| 2 | `SQ_WAIT_INST_LDS` / `SQ_INSTS_LDS` high, many `ds_read_b32` for small operands (scales, Q, softmax state) | Keep small per-tile operands in registers across the K loop. Widen the rest to `ds_read_b128`. Replace VGPR staging with `buffer_load … lds`. | More register state can drop occupancy. Check `-Rpass-analysis=kernel-resource-usage`. |
| 3 | PC samples on `s_waitcnt` > ~25 %, MFMA busy low, HBM not saturated | Hand pipeline: `vmcnt(N)` instead of `vmcnt(0)`, `sched_group_barrier` interleave (e.g. 1 VMEM : 4 MFMA : 1 DS-write : 1 DS-read), one drain before the dependent phase. | The barriers are hints that leave no trace in ISA. Verify the instruction *order* and that waitcnt share fell. |
| 4 | MFMA pipe issue-limited (many small MFMAs, low work per issue) | Larger-K atom (`16x16x32` bf16 on gfx950). Scaled `16x16x128` / `32x32x64` for fp8 or MX. A `32x32` atom halves the instruction count per FLOP versus `16x16`. | Change `BLOCK_K`, LDS strides, and lane maps together. K=32 strides with a K=128 atom is a classic bug. |
| 5 | `.private_segment_fixed_size > 0` or `v_accvgpr_*` inside the K loop | Stable vector accumulator → `"+a"` constraint → smaller tile → looser `__launch_bounds__` | Spills to HBM cost several × run time. Always check after raising unroll or tile size. |
| 6 | 1 WG/CU because LDS ≥ half the CU, MFMA util stuck near 50 % | Smaller LDS footprint, a larger MFMA atom (fewer issue gaps), or a persistent kernel. Otherwise accept the ceiling. | At occupancy 1 nothing hides latency across waves. Only the in-wave schedule can. |
| 7 | VMEM traffic > algorithmic bytes, kernel upcasts fp8 → bf16 on load | Native fp8 MFMA with the caller passing pre-quantized inputs. Buffer descriptors with b128 loads. Preshuffle weights on the host. | fp8 encoding differs: FNUZ on gfx942, OCP on gfx950. Validate against an fp32 reference. |
| 8 | `FETCH_SIZE` well above the minimum with high L2 hit | Buffer loads with `num_records` bounds instead of predicated tail loads. XCD-aware tile order. | The descriptor flags word differs by arch. Reuse your library's constant. |
| 9 | Grid < number of CUs (low-batch decode, long reduction) | Split-K / split-KV + a dedicated reduce kernel, fp32 partials + LSE | Cap the split count to what the reduce kernel supports. |
| 10 | PC samples heavy in per-tile prologue | Persistent grid (CU-sized) + atomic tile counter, prologue hoisted | Reset the counter each launch. No gain with only tens of tiles. |
| 11 | Reduce kernel > ~20 % of the chain | b128 loads, 256-thread WGs, branch-free accumulate | Do not fuse it into the producer without checking for spills. |
| 12 | Step time ≫ sum of kernel times | Remove host syncs (`.item()`, device `torch.equal`). Replace slow framework permutes or copies with a tiled transpose kernel. | Profile the whole step timeline before you rewrite a kernel. |

## Apply order (when several rows fire)

```
 1 LDS conflicts → 7 datapath bytes → 4 MFMA atom → 8 descriptors → 9 split
   → 2 register-resident state → 3 hand schedule → 10 persistent → 11 reduce (any time)
```

The hand schedule (row 3) comes late. It only pays once data reaches the MFMA pipe without conflicts, and the remaining stall is the wait counter.

## Decision rule before porting to raw HIP

Port a kernel to hand-written HIP only if the existing kernel is limited **inside the CU**: high LDS-wait ratio, waitcnt-bound loop, MFMA under-fed. If it is HBM-bound, fewer waits do not move the bandwidth ceiling. Expect parity, not a win.

Sources: [rocprofv3 / rocprofiler-sdk](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/) · [rocprofiler-compute metrics](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) · [LLVM AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html) · [HipKittens (arXiv 2511.08083)](https://arxiv.org/abs/2511.08083) · [AMD lab notes: Matrix Cores (GPUOpen)](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/) · [Stream-K (arXiv 2301.03598)](https://arxiv.org/abs/2301.03598)
Verify specifics against the primary source.

---
Related: [intrinsics-map](intrinsics-map.md) · [kernel-patterns](kernel-patterns.md) · [diagnostic-decision-tree](../../kernel-optimization/method/diagnostic-decision-tree.md) · [memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md)
