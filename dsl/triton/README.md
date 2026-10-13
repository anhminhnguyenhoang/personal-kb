# Triton on AMD (ROCm backend)

Triton-specific notes for AMD CDNA3/CDNA4 (gfx942, gfx950).
The general method lives in [`../../kernel-optimization/`](../../kernel-optimization/README.md).
Upstream: [triton-lang/triton `third_party/amd`](https://github.com/triton-lang/triton/tree/main/third_party/amd). Option names drift; verify on your build.

## Notes (reading order)

- [scope-and-ceiling](scope-and-ceiling.md) — When to use Triton on AMD, NVIDIA habits to drop, and signs a kernel hit the compiler-scheduling ceiling.
- [amd-knobs](amd-knobs.md) — AMD-only options (nonkdim, waves_per_eu, kpack, num_stages, knobs.amd): effect, start values, where to set.
- [codegen-pipeline](codegen-pipeline.md) — TTIR to AMDGCN stages, layouts and convert_layout LDS cost, tl.dot to MFMA, the stream pipeliner.
- [reading-amdgcn](reading-amdgcn.md) — Dump the ISA, read resource metadata, good-vs-bad inner-loop table, occupancy check for waves_per_eu.
- [low-precision-dtypes](low-precision-dtypes.md) — FNUZ vs OCP fp8 per arch, scale placement, tl.dot_scaled MXFP4 and scale preshuffling on gfx950.
- [operator-recipes](operator-recipes.md) — General recipes for prefill attention, fused MoE, fp8/MXFP4 quant, and RMSNorm in Triton.
