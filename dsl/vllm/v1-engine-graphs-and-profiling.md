---
name: v1-engine-graphs-and-profiling
description: "vLLM V0 vs V1: V0 is now an alias of V1. Covers the V1 process topology, CUDAGraphMode and its dispatcher, how graph mode changes trace interpretation, and the profiler endpoint chain."
source: https://github.com/vllm-project/vllm (vllm/engine/llm_engine.py, vllm/v1/, vllm/config/compilation.py, vllm/v1/cudagraph_dispatcher.py)
---

# vLLM V1 engine: graphs and profiling

## V0 vs V1

On current `main`, `vllm/engine/llm_engine.py` only aliases `vllm.v1.engine.llm_engine.LLMEngine`. `async_llm_engine.py` aliases `AsyncLLM` in the same way. **The V0 code is gone.** `VLLM_USE_V1=0` has no effect. Old guides that compare V0 and V1 describe behavior that no longer exists. The differences that still matter, when you read old numbers:

| | V0 (historical) | V1 |
| --- | --- | --- |
| scheduler / executor | in the engine process | engine core in a separate process, connected by IPC |
| workers | per-rank processes | one OS process per TP/PP rank |
| graphs | decode-only capture | `CUDAGraphMode` (piecewise + full) with torch.compile |
| profiler | engine-driven | in each worker process; one trace per rank |

## Process topology

```
HTTP ─► AsyncLLM ──IPC (zmq)──► EngineCore process(es) ─► GPUWorker × (TP·PP)   ◄── torch.profiler runs here
```

Rule: anything that happens per worker happens once per rank. A JIT `cpp_extension.load` or a Triton/aiter compile on the first call runs in every worker after the model loads. The first request after a restart can time out. Always warm up before a timed run or a profile.

## CUDAGraphMode

| Mode | Meaning |
| --- | --- |
| `NONE` | Eager. You see every launch. |
| `PIECEWISE` | Capture the compiled pieces between non-capturable ops (attention stays outside). |
| `FULL` | One graph for the whole forward pass. |
| `FULL_DECODE_ONLY = (FULL, NONE)` | Full graph for uniform-decode batches, eager otherwise. |
| `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` | V1 default. Full graph for uniform decode, piecewise for mixed or prefill. |

A tuple mode means `(mode for uniform-decode batches, mode for mixed batches)`. The dispatcher runs once per step:

```
if graphs off or num_tokens > max capture size: NONE
pad batch -> descriptor
if FULL allowed and descriptor captured for FULL: FULL          # uniform decode
elif PIECEWISE allowed and relaxed descriptor captured: PIECEWISE
else NONE
```

"Uniform decode" means every request adds the same number of query tokens: 1, or k+1 with speculative decoding. A batch over the largest captured size runs eager. That gives a cliff in latency and in trace shape. Some features force `NONE`, for example certain expert-parallel all-to-all modes.

**Profiling trap.** An eager trace adds host launch overhead to every kernel. Small kernels such as all-reduce, norms, and quant ops then look far more expensive than they are under graph replay. **State the graph mode next to every per-kernel number.** Do not use percentages from an eager trace for a deployment that uses graphs.

## Profiler chain

```
POST /start_profile ─► AsyncLLM.start_profile ─► engine_core.profile(True) ─► each GPUWorker.profile(True)
```

- Choose the profiler (`torch` or `cuda`) in the profiler config **when the server starts**. The worker checks the type at init. An unknown value fails at startup, not at the request.
- Each worker writes its own trace, named with the DP/PP/TP rank. Turn stack capture off for large TP. It makes the trace much larger and the flush slower.
- `cudagraph_mode` accepts the enum name as a string in config files.

Sources: [vllm/engine/llm_engine.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/llm_engine.py), [vllm/config/compilation.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py), [vLLM CUDA graphs design doc](https://docs.vllm.ai/en/latest/design/cuda_graphs.html), [vLLM profiling docs](https://docs.vllm.ai/en/latest/contributing/profiling.html), [vLLM V1 blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
Paths drift; verify before use.

---
Related: [model-forward-map-method](model-forward-map-method.md) · [sglang profiling-endpoints](../sglang/profiling-endpoints.md) · [measurement](../../kernel-optimization/method/measurement.md)
