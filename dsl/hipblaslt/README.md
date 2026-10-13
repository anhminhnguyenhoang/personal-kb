# hipBLASLt

hipBLASLt is AMD's GEMM library, with a cuBLASLt-style API and TensileLite kernels underneath. It is filed under `dsl/` by KB convention.
Upstream: [ROCm/rocm-libraries projects/hipblaslt](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt). Docs: [rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/).

## Notes

- [api-concepts](api-concepts.md) — Descriptor lifecycle, heuristic results and workspace, epilogue bit flags, FP8/MX scale modes, and the ext API with split-K and WGM tuning.
- [tensilelite-kernels-and-selection](tensilelite-kernels-and-selection.md) — Cijk naming, Equality, GridBased, and Range logic tables, what each solution parameter controls, and starting points by shape.
- [benchmarking-and-tuning](benchmarking-and-tuning.md) — Capture with LOG_MASK=32, tune, replay through the override file or cache, bench flags, and Stream-K env vars.
- [pitfalls](pitfalls.md) — Causes of zero algorithms, silent corruption from layouts and alpha, epilogue pointers, grouped-GEMM quirks, and testing patterns.
