# aiter (AI Tensor Engine for ROCm)

aiter is AMD's LLM-inference operator library and per-shape dispatcher. It is not a
kernel language: its kernels are written in CK, HIP, Triton, FlyDSL, opus, or raw asm,
and those are covered in their own `dsl/` folders. This folder covers the **library
control plane**: which op to call, how a call picks a kernel, how to tune that choice,
and how to prove it took effect.

Upstream: [ROCm/aiter](https://github.com/ROCm/aiter). Paths and symbols drift; verify before use.

## Notes

- [architecture](architecture.md) — Repo layout, the libtype router, the JIT cache, torch.compile opacity, and the SGLang/vLLM gates that turn aiter on.
- [tuned-config-dispatch](tuned-config-dispatch.md) — The per-shape lookup key, M buckets, fallback order, config env overrides and merging, and how to prove engagement.
- [operator-catalog](operator-catalog.md) — Ops by family: GEMM, the MoE pipeline stage by stage, attention, and the fused norm/quant/activation entries.
- [tuning-workflow](tuning-workflow.md) — Profile, capture live shapes, race backends with an error gate, deploy by env, prove, A/B; when to author instead.
- [pitfalls](pitfalls.md) — Silent no-ops and silent wrong results: dispatch misses, stale JIT, layout and scale contracts, fp8 dialects, graph-capture workspace.

Op-test conventions (reference, model-faithful inputs, TFLOP/s and TB/s columns, arch gating) live in the [aiter-op-test](https://github.com/ROCm/aiter/tree/main/.claude/skills/aiter-op-test) skill. FlyDSL legacy→current API migration: [flydsl-kernel-code-cleanup](https://github.com/ROCm/aiter/tree/main/.claude/skills/flydsl-kernel-code-cleanup).

Related: expert-parallel dispatch/combine lives in [MoRI](../mori/README.md).
