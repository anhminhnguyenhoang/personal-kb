# vLLM

vLLM is an LLM serving engine. It is filed under `dsl/` by KB convention. These notes cover how the V1 engine picks kernels on ROCm and how to profile it.
Upstream: [vllm-project/vllm](https://github.com/vllm-project/vllm). Paths drift; verify before use.

## Notes

- [rocm-code-map](rocm-code-map.md) — Platform class, arch flags, device-name map, the aiter custom-op shim with fake impls, and the csrc/rocm kernels.
- [attention-backend-selection-rocm](attention-backend-selection-rocm.md) — Enum registry plus platform priority list, the per-layer pick, the aiter master-AND-sub env rule, and log lines that confirm it.
- [moe-and-quant-selection](moe-and-quant-selection.md) — Triton versus aiter versus ASM MoE, device-named tuning JSONs, the FP8 linear path, and the silent scale-layout trap.
- [v1-engine-graphs-and-profiling](v1-engine-graphs-and-profiling.md) — The V1 engine's per-rank process topology, CUDAGraphMode dispatch, the eager-trace trap, and the profiler chain.
- [model-forward-map-method](model-forward-map-method.md) — An engine-neutral method: walk the forward pass, resolve each gate, tabulate the kernels, confirm with a warm trace.
