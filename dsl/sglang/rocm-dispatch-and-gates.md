---
name: rocm-dispatch-and-gates
description: How sglang decides, on ROCm, which attention backend and which aiter or Triton kernel a layer runs, and where the AMD-specific code lives.
source: https://github.com/sgl-project/sglang (python/sglang/srt/layers/attention/, python/sglang/srt/environ.py, python/sglang/srt/server_args.py)
---

# sglang on ROCm: dispatch and gates

Use this note to answer "which kernel runs on AMD for op X?" in sglang.

## The three gates

```
          is_hip()  ──── torch.version.hip is not None
             │
             ▼
   SGLANG_USE_AITER=1 ? ──no──► Triton / torch fallback
             │yes
             ▼
   arch check (is_gfx95_supported: gcnArchName contains "gfx95")
             │
             ▼
   aiter kernel, possibly a gfx950-only variant (fp8 prefill, mxfp4 fused quant)
```

- **Platform gate.** `is_hip()` is true when PyTorch is a ROCm build. `is_cuda_alike()` is true for CUDA or HIP.
- **Library gate.** Each file that calls aiter repeats one line near the top: `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip`. The default is off. If you do not set it, most AMD paths fall back to Triton.
- **Arch gate.** A gfx95 substring check turns on CDNA4-only paths. The quick all-reduce path checks for `gfx94` or `gfx95`.

To find every AMD branch, grep for `_use_aiter`, `is_hip`, and `gfx9`. Do not trust a fixed list of files.

## Attention backend selection

```
CLI --attention-backend / --prefill-attention-backend / --decode-attention-backend
        │ (unset?)
        ▼
server_args: per-model-arch defaults  (sets backend, page_size, kv_cache_dtype)
        │
        ▼
factory = ATTENTION_BACKENDS[name]; factory(runner)   (registry filled by @register_attention_backend)
        │
        ▼
optional wrapper (hybrid linear-attention models wrap the chosen backend)
```

- Each backend registers a factory by name, using a decorator in `attention_registry.py`. On ROCm the names that matter are `aiter`, `triton`, `wave`, `torch_native`, and the sparse-MLA backends (`nsa` and similar).
- Prefill and decode can use **different** backends. Speculative-decoding drafts can use a third.
- `server_args` sets defaults per model architecture. For some MLA families it uses a different `page_size` on HIP than on CUDA. Always read the startup log or the resolved `server_args` before you profile. Do not assume the default.
- The `aiter` backend imports MHA prefill, paged/ragged decode, MLA decode/prefill, and the Triton `unified_attention` from aiter. The import is inside `try/except ImportError`, so a missing aiter install can fail silently into a different path.
- Feature flags inside a backend often need **both** an env var and an arch check. For example, aiter fp8 prefill attention needs `SGLANG_AITER_FP8_PREFILL_ATTN` (default on) and gfx95.

## Where AMD code lives (paths drift)

| Concern | Where to look |
| --- | --- |
| platform helpers, FP8 max constant | `srt/utils/common.py` (`is_hip`, `is_gfx95_supported`) |
| env defaults | `srt/environ.py` (`SGLANG_USE_AITER`, `SGLANG_USE_AITER_UNIFIED_ATTN`, …) |
| attention backends | `srt/layers/attention/aiter_backend.py`, `attention_registry.py` |
| norms | `srt/layers/layernorm.py` (aiter `rmsnorm2d_fwd`, `rmsnorm2d_fwd_with_add`) |
| fused norm + quant | `srt/layers/communicator.py` (gfx95 mxfp4 / fp8-group variants) |
| MoE | `srt/layers/moe/` (see [moe-dispatch-on-rocm](moe-dispatch-on-rocm.md)) |
| ROCm linear helpers | `srt/layers/rocm_linear_utils.py` (aiter `tgemm.mm`, fused RoPE+concat) |
| all-reduce | `srt/distributed/device_communicators/quick_all_reduce.py` |

## FP8 flavor

On HIP, sglang sets its FP8 E4M3 max to 224 for gfx942. This is a conservative clamp below the FNUZ format maximum of 240. Elsewhere it uses `torch.finfo(float8_e4m3fn).max` (448). Code picks `float8_e4m3fnuz` or `float8_e4m3fn` through an `is_fp8_fnuz()` helper. Check this first when FP8 numbers differ between MI300 and MI350 or NVIDIA.

## What usually stays generic on ROCm

The sampler, many quant-scale helpers, and some indexer/compressor helpers stay on Triton unless you set an aiter env var. In a trace, a Triton kernel next to aiter kernels is not always a bug.

Sources: [sglang repo](https://github.com/sgl-project/sglang), [sglang AMD GPU docs](https://docs.sglang.ai/platforms/amd_gpu.html), [ROCm/aiter](https://github.com/ROCm/aiter).
Paths drift; verify before use.

---
Related: [moe-dispatch-on-rocm](moe-dispatch-on-rocm.md) · [profiling-endpoints](profiling-endpoints.md) · [model-forward-map-method](../vllm/model-forward-map-method.md)
