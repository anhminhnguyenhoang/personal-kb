---
name: rocm-code-map
description: Where vLLM keeps AMD-specific code (platform class, arch flags, device-name map, aiter custom-op shim, csrc/rocm kernels) and why each layer exists.
source: https://github.com/vllm-project/vllm (vllm/platforms/rocm.py, vllm/_aiter_ops.py, csrc/rocm/)
---

# vLLM ROCm code map

```
vllm/platforms/rocm.py     arch flags, device-name map, quant whitelist, backend priorities, custom-AR gate
        │
vllm/_aiter_ops.py         every aiter callable wrapped as torch.ops.vllm.rocm_aiter_*  (+ fake impl)
        │
layers/... + v1/attention/backends/rocm_*.py, mla/rocm_aiter_*.py      callers
        │
csrc/rocm/  → vllm._rocm_C   ROCm-only C++/HIP: paged attention, skinny GEMMs
csrc/*.cu   shared kernels with HIP branches (fused norm+quant, RoPE+KV insert, ...)
```

## Platform layer

- **Arch flags without starting CUDA.** At import, vLLM reads `gcnArchName` through amdsmi and stores booleans: `on_gfx9`, `on_mi3xx` (gfx942 or gfx950), `on_gfx942`, `on_gfx950`, `on_gfx1x`. Plain booleans are safe inside `torch.compile`. Reading device properties inside a compiled region is not.
- **Device-name map.** A PCI-id → name table produces strings such as `AMD_Instinct_MI300X`. Tuned-config files use the **same** strings in their names (`E=..,N=..,device_name=AMD_Instinct_MI300X.json`). A new SKU or VF id that is not in the map has no matching config. vLLM then falls back to a default config and gives no error. See [moe-and-quant-selection](moe-and-quant-selection.md).
- **Quant whitelist.** `supported_quantization` lists the methods allowed on ROCm (fp8 variants, quark, mxfp4, compressed-tensors, gptq/awq, …). A method missing from this list fails at config time, even when a kernel exists.
- **Custom all-reduce** is on only for `gfx94*` and `gfx95*`.
- **Kernel extension.** `import vllm._rocm_C` sits inside `contextlib.suppress(ImportError)`. A build without it still runs, but uses slower fallbacks.

## The aiter shim pattern

Each aiter function is registered as a torch custom op with a **fake (meta) implementation**:

```python
direct_register_custom_op(op_name="rocm_aiter_fused_moe",
    op_func=_impl,          # imports aiter lazily, translates enums, calls the kernel
    fake_impl=_fake,        # returns a correctly shaped empty tensor
    mutates_args=[], dispatch_key=current_platform.dispatch_key)
```

The pattern exists for three reasons:
- `torch.compile` and graph capture can trace through the op without running aiter.
- vLLM imports aiter only on first use, so a CUDA build never imports it.
- Callers see one stable signature. String names (`"silu"`, `"per_128x128"`) are converted to aiter enums in one place.

Use the same pattern when you add a new vendor kernel to a compiled model.

Some files skip the shim and `import aiter` directly, for example scaled-mm linear, quark MoE, MX experts, and Mori EP. To list them, grep for `^\s*(import|from) aiter` under `vllm/`.

## ROCm C++ kernels

`csrc/rocm/` holds only a few files: the ROCm paged-attention family, skinny GEMMs for small-M decode, and the torch bindings. Kernels shared with CUDA, such as fused norm+quant and RoPE+KV-cache insert, live in `csrc/` with HIP branches. A separate `csrc/quickreduce/` holds an AMD all-reduce.

## Divergences to expect

Model code sometimes keeps a slower ROCm path while a fused CUDA kernel is missing. Examples are a BF16 reference einsum in place of an FP8 fused one, and extra CUDA streams turned off on ROCm, which loses overlap. Search the model file for `is_rocm()` before you compare ROCm and CUDA traces.

Sources: [vllm/platforms/rocm.py](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/rocm.py), [vllm/_aiter_ops.py](https://github.com/vllm-project/vllm/blob/main/vllm/_aiter_ops.py), [vllm csrc/rocm](https://github.com/vllm-project/vllm/tree/main/csrc/rocm), [PyTorch custom ops](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html).
Paths drift; verify before use.

---
Related: [attention-backend-selection-rocm](attention-backend-selection-rocm.md) · [moe-and-quant-selection](moe-and-quant-selection.md)
