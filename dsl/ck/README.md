# Composable Kernel C++ (classic CK + CK-Tile)

CK is AMD's C++ template library for GPU kernels: `include/ck` (classic device ops) and
`include/ck_tile` (tile programming). Development lives in
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries/tree/develop/projects/composablekernel);
[ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) is a read-only mirror.
The Python DSL in the same repo has its own folder: [CK DSL](../ck-dsl/README.md).
General method lives in [`../../kernel-optimization/`](../../kernel-optimization/README.md).

```
 architecture ─► tile-distribution ─► gemm-pipelines ─► epilogues + quant
                                            │
                     fmha-forward ─► fmha-backward     fused-moe + flatmm     row-ops
                                            │
                                  build, codegen, debug
```

## Notes (reading order)

- [architecture-classic-vs-tile](architecture-classic-vs-tile.md) — Classic device-op stack vs CK-Tile Partitioner/Pipeline/Epilogue, the IsSupportedArgument gate, and which front-end to pick.
- [tile-distribution](tile-distribution.md) — Coordinate-transform chains and the R/H/P/Y distribution encoding behind tile windows and register tiles.
- [gemm-pipelines-and-schedulers](gemm-pipelines-and-schedulers.md) — GEMM parameter stack, pipeline families, Intrawave vs Interwave, sched-group recipes, LDS stage budget, split-K, Stream-K.
- [epilogues-and-quant-gemm](epilogues-and-quant-gemm.md) — CShuffle and quant epilogues, group-scale folding, MX/E8M0 scaled MFMA, MXFP4 scale math, fp8 dialects.
- [fmha-forward](fmha-forward.md) — FA-2 on CK-Tile: pipeline family, online-softmax edge cases, mask elision, split-KV combine, paged KV, fp8.
- [fmha-backward](fmha-backward.md) — D pre-kernel, five GEMMs with recomputed S, K/Kᵀ from one LDS copy, deterministic dQ.
- [fused-moe-and-flatmm](fused-moe-and-flatmm.md) — Expert sorting, fused gate-up/down with atomic epilogue, routed-weight placement, topk-softmax, flatmm for decode.
- [row-ops-reduce-norm](row-ops-reduce-norm.md) — One reduction primitive for softmax, norms, fused add-norm-quant and topk; pass-count and trait idioms.
- [build-codegen-and-debug](build-codegen-and-debug.md) — Scoped builds, codegen trait pruning, profiler sweeps, pin drift, and a symptom-to-cause table.
