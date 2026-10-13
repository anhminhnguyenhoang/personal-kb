---
name: hipkittens-idioms
description: "HipKittens library usage idioms checked against the public repo: arch macro, gl/st/rt/art tile types, shared_allocator, prefilled swizzled offsets, chiplet remap call, scaled MMA with packed E8M0, and per-kernel sched helpers."
source: https://github.com/HazyResearch/HipKittens (commit be1c918) ; https://arxiv.org/abs/2511.08083
---

# HipKittens: library usage idioms

The *why* (schedules, AGPR pinning, chiplet algorithm) is in [papers/hipkittens](../../papers/hipkittens/README.md). This note covers only *how the code is written*, checked against the repo at commit `be1c918`.
HK is a research artifact. Pin a commit; the API changes.

```
gl<T, B, D, R, C>  ──G::load──►  st<T, R, C, st_shape>  ──load──►  rt<T, R, C, layout, rt_shape>  ──mma_*──►  rt<float,…>
 (HBM view)        buffer_load…lds  (LDS, swizzled)        ds_read(_tr)   (registers)                 v_mfma
                                                                          art<…, ranges>  = pinned registers
```

| Idiom | Form | Note |
|---|---|---|
| Select arch | `-DKITTENS_CDNA3` / `-DKITTENS_CDNA4` / `-DKITTENS_CDNA5` | One tree; `kittens.cuh` picks `cdnaN/includes.cuh` |
| Global view | `gl<fp8e4m3, 1, 1, M, K>`, `-1` = runtime dim | Compile-time dims fold address math |
| LDS allocation | `extern __shared__ int __shm[]; shared_allocator al((int*)&__shm[0]);` then `al.allocate<ST, 2>()` | Create the allocator before any other LDS use |
| Block-wide group | `using G = kittens::group<NUM_WARPS>;` → `G::load(st, gl, {b, d, r, c})` | Group ops = all waves cooperate on one tile |
| Hoist swizzle math | `G::prefill_swizzled_offsets(st, gl, offs);` once, then `G::load(st, gl, coord, offs)` | Removes per-iteration VALU address math from the hot loop |
| MMA | `mma_AB`, `mma_ABt`, `mma_AtB`, `mma_AtBt` | Suffix = which operand is transposed |
| Scaled MMA (MX) | `mma_ABt_scaled(c, a, b, c, &sa, &sb)` with `fp8e8m0_4` packs | Four E8M0 scales per 32-bit register (one `opsel` byte each) |
| Scale tiles | `st<fp8e8m0, 16, 64, st_16x64_s>`, double-buffered like data | Scales prefetch on the same ping-pong as A/B |
| Pinned registers | `art<T, R, C, layout, shape, ranges>` with `range<lo, hi>` | Index ≥ 256 means AGPR (`range<368,383>` = `a[112:127]`) |
| Chiplet remap | `chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, chunk)` | `NUM_XCDS = 8` on MI350-class parts |
| Occupancy | `__launch_bounds__(512, 2)` for 8-wave ping-pong, `(256, 1)` for 4-wave interleave | Matches the two schedules |
| Sched helpers | `sched_barrier_pairs<Pairs, VALU_CNT, Group>()` = recursive `sched_group_barrier(0x008,1,G)` + `(0x002,N,G)` | Defined **in kernel files**, not the library; copy and adapt |

## Lessons

- **The MX GEMM structure matches a hand-written HIP one.** A 2×2 sub-tile pattern (A0·B0, A0·B1, A1·B0, A1·B1) per K-step, scales packed four per register, and data plus scale double buffers. See [mxfp8-grouped-gemm](mxfp8-grouped-gemm.md) and [kernel-patterns](kernel-patterns.md).
- **Grab the attention-backward register map before you write your own pinning.** The kernel spells out every `range<>` with comments. It is the best public example of splitting 512 registers across Q, K, dO, and accumulators.

Sources: [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens) (`include/kittens.cuh`, `include/cdna4/`, `kernels/cdna4/gemm/{fp8fp32,mxfp8}/`, `kernels/cdna4/attn/gqa_backwards/`) · [HipKittens paper (arXiv 2511.08083)](https://arxiv.org/abs/2511.08083)
Paths drift; verify before use.

---
Related: [hipcc-agpr-pinning](../../papers/hipkittens/01-paper/hipcc-agpr-pinning.md) · [schedules](../../papers/hipkittens/01-paper/schedules.md) · [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md) · [intrinsics-map](intrinsics-map.md)
