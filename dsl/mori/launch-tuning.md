---
name: launch-tuning
description: "Tuning MoRI-EP launch geometry: MANUAL vs AUTO mode, the per-arch JSON rule tables and their keys, MoRI's two-phase tuner, why dispatch, combine, and each buffer mode need their own geometry, and why aiter callers do not see the tables."
source: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md ; https://github.com/ROCm/mori/blob/main/python/mori/ops/tuning_config.py ; https://github.com/ROCm/mori/tree/main/python/mori/ops/tuning_configs
---

# MoRI-EP launch tuning

The knobs are launch geometry, not tiles: `block_num`, `warp_per_block`, and for
inter-node `rdma_block_num` (and queue pairs per peer). The metric is bandwidth on the
**slowest rank**.

## MANUAL vs AUTO

`MORI_EP_LAUNCH_CONFIG_MODE` (default `MANUAL`):

```
MANUAL → per-call override  >  EpDispatchCombineConfig value  (class default 80 blocks × 8 warps)
AUTO   → JSON rule for (arch, model, kernel_type, ep_size, phase, shape) if present
         else hard-coded per-kernel-family fallback
         per-call block/warp overrides are IGNORED
```

If a sweep "has no effect", check this variable first. Three different "defaults" exist
(the class default, the AUTO fallback, and what a framework passes). Record which one a
number came from before comparing numbers across sources.

## Rule tables

Files: `python/mori/ops/tuning_configs/{arch}_{model}_{kernel}_ep{n}_{phase}.json`.

| Phase | Rule key |
|---|---|
| dispatch | `dtype, num_tokens, hidden_dim, topk` |
| combine | `dtype, num_tokens, hidden_dim, topk, zero_copy, quant_type` |

- `topk` was added later; old rules without it act as wildcards. It matters because two
  models with the same hidden size but different top-k move different bytes per rank.
  General rule: **key on every field that changes bytes per rank**.
- Combine has two extra key fields because the buffer mode and wire codec are different
  kernels. Example from MoRI's shipped MI308X table (EP8, 4096 tokens, hidden 7168, bf16,
  no codec): external buffer best at 72 blocks × 16 warps, 258 GB/s; zero-copy best at
  80 blocks × 4 warps, 333 GB/s. A geometry tuned for one mode is wrong for the other.
- Only the geometry fields are read at runtime. Bandwidth and latency fields are records
  for humans.
- Tables exist only for the arch/model pairs someone tuned. On a missing model, AUTO
  runs the fallback, which is a reasonable free default but not a tuned result.

## MoRI's tuner: calibrate, then sweep

`tools/batch_intranode_tuning.sh` (and an inter-node variant) sweeps the geometry per
(EP size, dtype, codec, zero-copy) group across token counts.

1. **Calibrate:** full search (about 75 configs) on two token counts, one small and one
   large.
2. **Quick sweep:** a narrow scope (about 10 configs) over all token counts, about 6x
   faster. Safe when calibration shows the optimum sits inside the quick scope.

The same two-phase idea fits any expensive sweep: spend the full budget on a few
representative points to bound the space, then sweep narrowly.

## Geometry rules of thumb

- At large token counts, block counts that scale with the CU count tend to win.
- At small (decode-size) token counts there is not enough work per block; fewer blocks
  win, because extra blocks add overhead without payload.
- Tune **dispatch and combine separately**. Their bottlenecks differ: dispatch issues
  posted remote writes (throughput-bound), while combine in some designs does remote
  reads, which are latency-bound and need more warps in flight to hide xGMI latency.
- The low-latency kernel types are for small batches; do not expect them to win at
  throughput shapes. Measure both on the target shape.

## aiter callers do not use the tables

aiter's `MoriAll2AllManager` passes fixed geometry and does not set AUTO. A tuned JSON rule
helps direct MoRI callers in AUTO mode; to help SGLang/vLLM traffic that goes through
aiter, the aiter side has to change too.

## Measurement hygiene for comms ops

- Call `reset()` (or `call_reset=True`) every iteration, as production does.
- Keep host syncs and harness copies out of the timed region. A zero-copy run that still
  copies the expert output into the registered buffer inside the timer measures the very
  copy zero-copy removes.
- Report the slowest rank, and interleave A/B runs in one session.

Sources: [MORI-EP guide](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md) (launch modes, tuner, JSON format) ·
[`tuning_config.py`](https://github.com/ROCm/mori/blob/main/python/mori/ops/tuning_config.py) (rule keys, `topk` rationale) ·
[`tuning_configs/`](https://github.com/ROCm/mori/tree/main/python/mori/ops/tuning_configs) (shipped tables) ·
[`dispatch_combine_v2` README](https://github.com/ROCm/mori/blob/main/python/mori/ops/dispatch_combine_v2/README.md) (per-phase geometry) ·
[aiter `all2all.py`](https://github.com/ROCm/aiter/blob/main/aiter/dist/device_communicators/all2all.py)
Verify specifics against the primary source. Paths drift; verify before use.

---
Related: [ep-dispatch-combine](ep-dispatch-combine.md) · [measurement](../../kernel-optimization/method/measurement.md)
