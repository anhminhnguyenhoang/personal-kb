---
name: attention-backend-selection-rocm
description: How vLLM V1 picks an attention backend on ROCm (enum registry, platform priority list, per-backend validation), the aiter env-var AND rule, and how to confirm the pick from logs.
source: https://github.com/vllm-project/vllm (vllm/platforms/rocm.py, vllm/v1/attention/backends/registry.py, vllm/_aiter_ops.py, vllm/envs.py)
---

# vLLM attention backend selection on ROCm

## Two layers

```
Layer 1  AttentionBackendEnum         NAME -> "fully.qualified.BackendClass"  (string, lazy import)
Layer 2  RocmPlatform priority list   ordered list of enum members, filtered by validate_configuration()

 --attention-backend NAME given?
      yes ─► validate only NAME ─► "Using NAME backend (selected via --attention-backend)."
      no  ─► walk priority list in order, keep the first one that validates
             ─► "Using X backend out of potential backends: [...]"
             or "Found incompatible backend(s) [...] ... Overriding with X ..."
```

The enum value is a class path string, resolved only when the backend is used. A runtime `register_backend` can override it, so tests and plugins can swap a backend in without editing the enum.

## Priority lists (shape of the logic; verify per version)

| Layer kind | Priority order |
| --- | --- |
| sparse MLA (layer has a top-k indexer) | `ROCM_AITER_MLA_SPARSE` only |
| MLA, aiter MLA enabled | `ROCM_AITER_MLA` → `TRITON_MLA` → `ROCM_AITER_TRITON_MLA` |
| MLA, aiter MLA disabled | `TRITON_MLA` |
| MHA/GQA | `ROCM_ATTN` → `ROCM_AITER_FA` (if aiter MHA on) → `ROCM_AITER_UNIFIED_ATTN` (if aiter installed and arch supported) → `TRITON_ATTN` → … |

- **The choice is made per layer, not per model.** `use_mla` and `use_sparse` come from each layer's attention type. A model with indexer layers can run a sparse MLA backend on those layers and a dense MLA backend on the rest, in the same forward pass.
- Newer versions take more inputs. For example, a KV-connector flag removes `ROCM_ATTN` from the list. Read `_get_backend_priorities` in the version you run.
- Vision-encoder (ViT) attention uses a separate chooser. On gfx9 it picks aiter FA when aiter is on, then `flash_attn` when it is installed (fp16/bf16), then Triton FA on RDNA, then SDPA.

## The aiter env-var AND rule

```
is_mla_enabled() = VLLM_ROCM_USE_AITER  AND  VLLM_ROCM_USE_AITER_MLA
is_mha_enabled() = VLLM_ROCM_USE_AITER  AND  VLLM_ROCM_USE_AITER_MHA
         ...       (same shape for MoE, linear, rmsnorm, fp8/fp4 BMM, ...)
```

- The master switch `VLLM_ROCM_USE_AITER` defaults to **off**. Most sub-switches default to **on**. Until you turn the master on, aiter does nothing, even when every sub-flag reads `True`.
- A separate check, `is_aiter_found_and_supported()`, tests only platform, arch, and that the library is installed. It does not read env vars. Some backends, such as unified attention, depend only on this check. They can still enter the priority list when the master switch is off.
- vLLM caches env values as class attributes when it imports the module. If a test changes `os.environ` after import, call `refresh_env_variables()` to reload them.

## Before you profile

Grep the server log for `Using ... backend` and `Found incompatible backend(s)`. The log does not show the per-layer pick. If the model mixes layer kinds, also check the model code for which layers set `use_sparse` or `use_mla`.

Sources: [vllm/platforms/rocm.py](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py), [vllm attention backends](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends), [vLLM env vars docs](https://docs.vllm.ai/en/latest/configuration/env_vars.html), [ROCm vLLM docs](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html).
Paths drift; verify before use.

---
Related: [rocm-code-map](rocm-code-map.md) · [model-forward-map-method](model-forward-map-method.md) · [sglang rocm-dispatch-and-gates](../sglang/rocm-dispatch-and-gates.md)
