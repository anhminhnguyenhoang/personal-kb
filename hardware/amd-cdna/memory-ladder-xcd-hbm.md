---
name: memory-ladder-xcd-hbm
description: CDNA3/CDNA4 memory ladder (registers → LDS → L1 → per-XCD L2 → Infinity Cache → HBM), roofline ridge per GPU and dtype, XCD locality rules, the 512 B-stride Tagram cliff, and Infinity Fabric numbers.
source: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; AMD CDNA3 white paper ; arXiv 2511.08083 ; arXiv 2510.27583
---

# Memory ladder, roofline ridge, and XCD locality (CDNA3 / CDNA4)

## 1. The ladder

```
 per SIMD   VGPR+AGPR   512 regs × 64 lanes × 4 B = 128 KB / SIMD → 512 KB / CU
 per CU     LDS         64 KB (CDNA3)  | 160 KB (CDNA4)   software-managed
 per CU     L1 vector   32 KB, 128 B line                 hardware cache
 per XCD    L2          4 MB  (8 per package, NOT shared)  hardware cache
 device     Infinity Cache (MALL)  256 MB on the IODs      first device-wide cache
 device     HBM         5.3 / 6.0 / 8.0 TB/s
            ▲
            └── a line missed in XCD 3's L2 is not found in XCD 5's L2.
                Cross-XCD reuse is served by Infinity Cache or HBM.
```

What each tier means for the category (scratchpad vs cache): see
[memory-hierarchy](../../kernel-optimization/levers/memory-hierarchy.md).

## 2. HBM by generation

| GPU | Arch | HBM | Capacity | Peak BW |
|---|---|---|---|---|
| MI100 | CDNA1 | HBM2 | 32 GB | 1.2 TB/s |
| MI250X | CDNA2 | HBM2e | 128 GB (2 dies) | 3.2 TB/s (two separate devices) |
| MI300X | CDNA3 | HBM3 | 192 GB | 5.3 TB/s |
| MI325X | CDNA3 | HBM3E | 256 GB | 6.0 TB/s |
| MI350X / MI355X | CDNA4 | HBM3E | 288 GB | 8.0 TB/s |

Peak is not sustained. Measure your own ceiling with a streaming copy before you call a kernel
"at roofline".

## 3. Roofline ridge (FLOP per byte where compute = memory)

```
ridge = peak_FLOPs / peak_BW
```

| GPU | BF16/FP16 | FP8 | FP4 (MX) |
|---|---|---|---|
| MI300X (1.3 PF, 5.3 TB/s) | ~247 | ~493 | — |
| MI325X (1.3 PF, 6.0 TB/s) | ~218 | ~436 | — |
| MI350X (2.3 PF, 8.0 TB/s) | ~288 | ~575 | ~1150 |
| MI355X (2.5 PF, 8.0 TB/s) | ~313 | ~625 | ~1250 |

Rule for weight-streaming GEMM (`M` tokens, weights `N×K`, `M ≪ N,K`):

```
intensity ≈ 2·M·N·K / (N·K·bytes_per_weight) = 2·M / bytes_per_weight
   BF16 weights → ≈ M      FLOP/B
   FP8  weights → ≈ 2·M    FLOP/B
   FP4  weights → ≈ 4·M    FLOP/B
```

So decode with tens of tokens sits far left of every ridge: count bytes, not FLOPs.
Peak FLOPs overstate the ridge. With sustained FLOPs (about half of peak on MI300X) the ridge
moves left by about 2×.

## 4. XCD locality

```
 dispatch:  workgroup k ──► XCD (k mod 8)        (round-robin)

  row-major grid, no remap         grid with chunked remap
  ┌──┬──┬──┬──┬──┬──┬──┬──┐       ┌──┬──┬──┬──┬──┬──┬──┬──┐
  │0 │1 │2 │3 │4 │5 │6 │7 │       │0 │0 │1 │1 │2 │2 │3 │3 │
  │0 │1 │2 │3 │4 │5 │6 │7 │       │0 │0 │1 │1 │2 │2 │3 │3 │
  └──┴──┴──┴──┴──┴──┴──┴──┘       └──┴──┴──┴──┴──┴──┴──┴──┘
  each A row is read into 8 L2s    each XCD owns a 2-D block
```

- Use grid and tile counts that are multiples of 8 (both generations have 8 XCDs).
- Put workgroups that share an operand on the same XCD. This improves hit rate and run-to-run
  stability. Remap algorithm: [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md).
- Fill the device: launch many more workgroups than CUs (304 or 256) so the tail is short.

## 5. The 512 B-stride Tagram cliff (MI300)

```
 stride = 512 B × n  ──► many rows map to the same L2 tag channel ──► VMEM latency ↑, perf drop
 stride = 512 B × n + pad ──► rows spread over channels
```

- Worst for TN GEMM. Symptom: one shape is much slower than its neighbours with similar size.
- Fix: pad the leading dimension so the byte stride is not a multiple of 512 B. AMD's example: TN
  FP16 with `K % 256 == 0` → use `lda = ldb = K + 128` elements.
- Same idea as LDS padding, one level up.

## 6. Infinity Fabric (GPU to GPU)

| Item | MI300X / MI325X | MI350X / MI355X |
|---|---|---|
| Link speed | 32 Gbps | 38.4 Gbps |
| P2P ring aggregate | 896 GB/s | 1075.2 GB/s |
| Total peak aggregate I/O | 1024 GB/s | 1203.2 GB/s |
| Topology | 8 GPUs, fully connected | same |

Fully connected means each GPU pair has its own link. A collective over 2 or 4 GPUs uses only
some links, so it gets a fraction of node bandwidth. A small all-reduce is latency-bound,
not bandwidth-bound.

Verify specifics against the primary source.

Sources:
- [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) (cache sizes, Tagram stride, multiples of 8, fabric table)
- [AMD CDNA 3 white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA 4 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [HipKittens, arXiv 2511.08083](https://arxiv.org/abs/2511.08083) (round-robin XCD dispatch, remap)
- [arXiv 2510.27583](https://arxiv.org/abs/2510.27583) (HBM table across generations, sustained-of-peak)

---
Related: [cdna3-cdna4-deltas](cdna3-cdna4-deltas.md) · [lds-model-and-banks](lds-model-and-banks.md) · [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md) · [chiplet-scheduling](../../papers/hipkittens/01-paper/chiplet-scheduling.md)
