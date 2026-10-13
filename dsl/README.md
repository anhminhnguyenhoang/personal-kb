# Kernel DSLs and libraries

How one DSL, language, or library spells a concept. The general method lives in [`../kernel-optimization/`](../kernel-optimization/README.md).

## Kernel languages and DSLs

- [FlyDSL](flydsl/README.md) — Layout algebra, authoring primitives, compilation pipeline, pitfalls, LDS/prefetch idioms, hot-loop scheduling, varlen idioms, operator recipes.
- [Triton on AMD](triton/README.md) — AMD-only knobs, codegen stages, reading AMDGCN, fp8/MXFP4 types, operator recipes, and the Triton ceiling.
- [HIP](hip/README.md) — Builtin intrinsics map, CDNA kernel patterns, profiler-signal-to-recipe table, MXFP8 grouped GEMM, HipKittens idioms.
- [Composable Kernel C++](ck/README.md) — Classic CK vs CK-Tile, tile distribution, GEMM pipelines, epilogues and quant GEMM, FMHA, fused MoE, row ops, build and debug.
- [CK DSL (ck_dsl)](ck-dsl/README.md) — The CK Python DSL: primitive map, commands, probe workflow, and knob catalog.
- [AMDGCN assembly](amdgcn-asm/README.md) — Hand-written gfx942/gfx950 kernels: ABI and prologue, registers, waitcnt and hazards, memory instructions, coherence bits, MFMA, debugging.

## Libraries and serving engines

- [aiter](aiter/README.md) — Architecture, tuned-config dispatch and engagement proof, operator catalog, tuning workflow, pitfalls.
- [MoRI](mori/README.md) — Expert-parallel dispatch/combine data flow and launch tuning.
- [hipBLASLt](hipblaslt/README.md) — API concepts, TensileLite kernel naming and selection, benchmarking and tuning, pitfalls.
- [sglang](sglang/README.md) — ROCm dispatch gates, MoE dispatch, profiling endpoints.
- [vLLM](vllm/README.md) — ROCm code map, attention backend and MoE/quant selection, V1 engine graphs, the model forward-map method.
