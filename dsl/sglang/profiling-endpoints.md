---
name: profiling-endpoints
description: How to capture torch-profiler or ROCm RPD traces from a running sglang server, which request fields and env vars apply, and how trace files are named per rank.
source: https://github.com/sgl-project/sglang (python/sglang/srt/entrypoints/http_server.py, python/sglang/srt/utils/profile_utils.py, python/sglang/srt/environ.py)
---

# sglang profiling endpoints

## Flow

```
client ──POST /start_profile {json}──► HTTP server ─► tokenizer manager ─► every scheduler rank
                                                                           │ torch.profiler.start()
          ... run a small workload ...                                     │ (or RPD tracer)
client ──POST /stop_profile──────────► ... ─► each rank exports a trace, then waits at a barrier
```

- Endpoints: `/start_profile` and `/stop_profile`, which accept GET or POST. The `Engine` class has in-process `start_profile()` / `stop_profile()`.
- A profile can stop on its own: set `num_steps` (and optionally `start_step`) in the request. Then you do not need `/stop_profile`.

## Request fields

| Field | Effect |
| --- | --- |
| `output_dir` | Output directory. Falls back to `SGLANG_TORCH_PROFILER_DIR`, then `/tmp`. |
| `activities` | Default `["CPU","GPU"]`. Also accepts `MEM` (memory history), `RPD` (ROCm), `CUDA_PROFILER` (cudaProfilerStart/Stop hook). |
| `with_stack`, `record_shapes` | Per request. In some versions these override the `SGLANG_PROFILE_*` env vars. |
| `profile_by_stage` | Separate traces for prefill and decode. |
| `merge_profiles` | Merge the per-rank traces. |
| `num_steps`, `start_step` | Window of forward steps to capture. |

**Trap:** env vars such as `SGLANG_PROFILE_WITH_STACK` exist in `environ.py`, but older code paths read only the request fields. Pass `with_stack` and `record_shapes` in the JSON body. Do not rely on the env var.

## ROCm RPD

With `"activities":["RPD"]`, sglang uses the rocmProfileData tracer instead of torch.profiler. Rank 0 creates a `trace.rpd` SQLite file, every rank traces into it, and at stop rank 0 converts it to a Chrome trace (`rpd-<ts>-TP-<rank>.trace.json.gz`). RPD records HIP API and kernel activity at lower overhead than torch.profiler with stacks.

## File naming

Each rank writes `<prefix><profile_id>-TP-<r>[-DP-<d>][-PP-<p>][-EP-<e>].trace.json.gz`. A DP, PP, or EP suffix appears only when that parallel size is above 1. Every rank waits at a barrier after export. Wait until all files exist before you read them.

## Minimal capture

```bash
export SGLANG_TORCH_PROFILER_DIR=/path/to/traces
# start the server, warm it up with a few requests, then:
curl -X POST localhost:30000/start_profile -H 'Content-Type: application/json' \
  -d '{"activities":["CPU","GPU"],"with_stack":false,"record_shapes":false,"num_steps":5}'
# send a small, fixed workload (a few prompts, low concurrency)
```

Rules:
- Warm up first. JIT builds and graph capture on the first requests distort the trace.
- Use `with_stack=false` for large tensor-parallel runs. Stack capture makes traces large and flushing slow.
- Graph mode changes what the trace shows. Eager traces show each launch with its host overhead. Graph replay hides that overhead. Record which mode you captured. sglang chooses graph mode through server args, not through an env var.

Sources: [sglang benchmark and profiling docs](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html), [sglang repo](https://github.com/sgl-project/sglang), [ROCm/rocmProfileData](https://github.com/ROCm/rocmProfileData).
Paths drift; verify before use.

---
Related: [rocm-dispatch-and-gates](rocm-dispatch-and-gates.md) · [vllm v1-engine-graphs-and-profiling](../vllm/v1-engine-graphs-and-profiling.md) · [measurement](../../kernel-optimization/method/measurement.md)
