---
name: kernel-fusion
description: When fusing kernels pays (cut HBM round-trips, hide work inside a donor's pipeline, overlap communication), the usual donors and fusion patterns, and when fusion loses (occupancy, reuse, mismatched geometry, numerics).
source: Williams et al., "Roofline", CACM 2009 ; FlashAttention arXiv 2205.14135 ; https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html ; https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
---

# Kernel fusion strategy

Fusion is a bytes lever first and a launch lever second. Decide with the
roofline, not with the kernel count.

```text
staged:   A ──HBM──► B ──HBM──► C        bytes = in + 2·mid + out
fused:    A ─regs/LDS─► B ─regs/LDS─► C  bytes = in + out
```

## Why fusion helps

1. **It removes an HBM round-trip.** Two bandwidth-bound ops chained
   through HBM read and write the intermediate twice. Fused, the
   intermediate stays in registers or LDS. On the bandwidth roof, time
   scales with bytes, so this is the main win.
2. **A donor hides the small op.** A compute-bound kernel (GEMM,
   attention) has spare VALU and memory issue slots while the matrix pipe
   runs. Bias, activation, scale, or quantize in its epilogue, or
   dequantize or normalize in its prologue, costs little extra time.
3. **It overlaps communication with compute.** A collective fused with
   the op that consumes it (e.g. reduce + norm) lets the vector work run
   while data moves.
4. **It cuts launches.** This matters only when the timeline shows the
   GPU idle between short kernels. On kernels of tens of µs or more,
   graph replay or fewer launches buys only a few µs.

## Donors and patterns

| Pattern | Donor | Payoff |
|---|---|---|
| epilogue → GEMM (bias, activation, scale, quantize) | GEMM | no output round-trip; epilogue in the shadow of the main loop |
| prologue → GEMM (dequantize, normalize A) | GEMM | skip a full pre-pass over activations |
| residual add + norm | norm pass | one read of each input, one write |
| norm + quantize (+ scale out) | norm pass | one pass writes quantized data and scales |
| position encoding + cache write | copy pass | apply and store in one pass |
| QK → softmax → PV | attention | the score matrix never reaches HBM (FlashAttention) |
| collective + norm | collective | hide collective latency behind vector work |

## When not to fuse

- **A faster standalone kernel exists.** A hand-fused GEMM + epilogue
  that loses to a tuned library GEMM plus a cheap separate epilogue is a
  regression. Compare against the best staged pair.
- **The register or LDS budget breaks.** Extra fused state can drop
  occupancy below the latency-hiding level or cause spills. Check the
  resource report (see [isa-resource-inspection](isa-resource-inspection.md)).
- **It destroys reuse.** Fusing a high-reuse op into a streaming one can
  force recompute or re-fetch. Recompute AI after fusion.
- **The ops want different launch geometry.** One grid and tile shape
  forced on both ops penalizes both.
- **It crosses a numeric boundary.** Fusing across an fp32 accumulate or
  rescale step can lose accuracy (see [numerical-stability](numerical-stability.md)).
- **It scalarizes the store.** A fused epilogue that turns vector stores
  into per-element stores loses more than it saves (see [epilogue](epilogue.md)).

## Verify

- HBM bytes (`FetchSize` + `WriteSize`) should drop by about the
  intermediate size, and the roofline point should move.
- Fusion changes AI. Two bandwidth-bound ops fused may become balanced or
  compute-bound. Re-classify before choosing the next lever.
- A/B the fused kernel against the staged pair in the same process, then
  check numerics against an unfused fp32 reference.
- Over-fusion signature: a megakernel that spills and runs slower than
  the staged version.

Sources: Williams et al., "Roofline: an insightful visual performance model", CACM 52(4), 2009 · [FlashAttention, arXiv 2205.14135](https://arxiv.org/abs/2205.14135) · [MI300X workload tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html) · [HIP performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)

---
Related: [algorithmic-mapping](algorithmic-mapping.md) · [epilogue](epilogue.md) · [op-specific-checklists](../autotuning/op-specific-checklists.md) · [profiling-tiers](../method/profiling-tiers.md)
