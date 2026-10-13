---
name: model-forward-map-method
description: An engine-neutral method to map which kernel each op in a model forward actually runs on a given platform, by static gate-walking first and trace confirmation second.
source: https://github.com/vllm-project/vllm, https://github.com/sgl-project/sglang, https://pytorch.org/docs/stable/profiler.html
---

# Model forward map: which kernels does a forward pass call?

The goal is a table of the form "op in the model → kernel symbol on this platform → source branch that chose it". Use the table to rank optimization targets and to check that a trace is what you expect.

## Method

```
1. resolve config ─► 2. walk the call tree ─► 3. resolve each gate ─► 4. tabulate ─► 5. confirm with a trace
```

1. **Resolve the effective config first.** Engines rewrite defaults by model architecture: attention backend, page size, KV-cache dtype, MoE backend. Read the resolved args and the startup log. Do not use the CLI you typed.
2. **Walk the call tree from the top.** Start at `ForCausalLM.forward`, then `Model.forward`, then one `DecoderLayer.forward`. Go into each submodule (norm, attention, MLP or MoE) until you reach a custom op, a library call, or a kernel launch.
3. **Resolve every gate on the path.** Common gate kinds:
   - platform: `is_hip()`, `is_rocm()`, arch flags (`gfx942`, `gfx950`)
   - library master switch AND feature switch (`*_USE_AITER` AND `*_USE_AITER_<OP>`)
   - **priority chains**: `if A_enabled: ... elif B_enabled: ... else: torch`. When several flags default to true, the **first** branch wins. To force the second one, you must turn the first off.
   - per-layer attributes: layer kind, sparse vs dense, a compress ratio, hash-routed layers. Two layers of the same model can take different paths.
   - shape gates: token-count thresholds, `num_tokens <= X` for multi-stream overlap, batch size against the largest captured graph.
4. **Tabulate.** Give one row per op: `step | resolved symbol | file:branch`. Add a "not investigated" list. Unknown rows are as useful as known ones, because they show where the map is weak.
5. **Confirm dynamically.** Capture a short, warm trace with a known graph mode. Group kernel names into buckets (GEMM, attention, MoE, norm/quant, RoPE/cache, comms). Check that each bucket matches the table. A kernel you did not predict usually means a gate you resolved wrong.

## Things the static walk misses

- **Platform divergences.** A model can keep a reference path on one platform while a fused kernel exists only on another, or turn off extra streams on one platform. Search the model file for platform checks.
- **Fallbacks hidden by `try/except ImportError`.** A missing optional library changes the path without any error.
- **Tuned-config lookup.** The same kernel can run with a default config when the shape or device name has no tuned entry. The symbol in the trace is the same, but it runs much slower.
- **Graph capture.** Under full-graph replay, small kernels look cheap. Under eager they look expensive. Record the mode.

## Skeleton of an MLA + MoE decode step (example only)

`norm → fused QKV/LoRA GEMM → q/kv norm → RoPE + KV-cache insert (+quant) → MLA decode attention → O-proj GEMM(s) → norm → router GEMM + top-k → fused MoE → all-reduce`. Each arrow is one row in the table.

Sources: [vLLM model executor](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models), [sglang models](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models), [PyTorch profiler](https://pytorch.org/docs/stable/profiler.html).

---
Related: [attention-backend-selection-rocm](attention-backend-selection-rocm.md) · [v1-engine-graphs-and-profiling](v1-engine-graphs-and-profiling.md) · [sglang rocm-dispatch-and-gates](../sglang/rocm-dispatch-and-gates.md) · [bottleneck-classification](../../kernel-optimization/method/bottleneck-classification.md)
