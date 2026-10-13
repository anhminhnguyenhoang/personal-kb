# sglang

sglang is an LLM serving engine. It is filed under `dsl/` by KB convention. These notes cover how it picks kernels on ROCm and how to profile it.
Upstream: [sgl-project/sglang](https://github.com/sgl-project/sglang). Paths drift; verify before use.

## Notes

- [rocm-dispatch-and-gates](rocm-dispatch-and-gates.md) — The is_hip, SGLANG_USE_AITER, and gfx95 gates, the attention backend registry and per-model defaults, and where AMD code lives.
- [moe-dispatch-on-rocm](moe-dispatch-on-rocm.md) — Router, top-k, and expert call chain, the TP versus EP choice, and adapter traps: sentinel ids, folded scale, activation mapping.
- [profiling-endpoints](profiling-endpoints.md) — /start_profile request fields, RPD on ROCm, per-rank trace naming, and a minimal capture recipe.
- [model-forward-map-method](../vllm/model-forward-map-method.md) — How to trace which kernels a model forward calls, for any engine (stored under vllm/).
