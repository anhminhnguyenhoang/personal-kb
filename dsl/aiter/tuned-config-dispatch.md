---
name: tuned-config-dispatch
description: "How aiter resolves a live call to a kernel through per-shape tuned CSVs: the lookup key, padded-M buckets, fallback order, config-file env overrides and merging, and how to prove a tuned row actually engaged."
source: https://github.com/ROCm/aiter/blob/main/aiter/tuned_gemm.py ; https://github.com/ROCm/aiter/blob/main/aiter/jit/core.py ; https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py
---

# Tuned-config dispatch and engagement

Every tuned aiter op works the same way. It builds a key from the live call, looks the
key up in a CSV, and runs the `libtype` + kernel id on the matching row. If no row
matches, it runs a default. **A miss is silent**: the op still works, just slower.

## Dense GEMM lookup (`aiter.tuned_gemm`)

```
key = (gfx, cu_num, padded_M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)

for gl in [exact M, fine bucket, coarse bucket]:      # get_padded_m(M, N, K, gl)
    row = csv.get(key with padded_M)
    if row and row.libtype == "flydsl" and (flydsl not installed or name unknown):
        continue                                      # dropped silently
    if row: log "is tuned on cu_num = ..."; return row
default: hipblaslt/asm if B is preshuffled, skinny for small-M default shapes, else torch
```

| Key field | Comes from | Typical miss |
|---|---|---|
| `gfx` | `get_gfx()` | CSV tuned on another arch |
| `cu_num` | `get_cu_num()` | Same arch, different SKU (CU count) |
| `padded_M` | Bucketed M (fine: round to 16/32/64/128 by range; coarse: next power of 2) | None. Buckets let one row serve a range of M |
| `bias` | `bias is not None` | Synthesized `bias=True` rows while the model calls with no bias |
| `scaleAB` | Scales passed or not | Scaled GEMM rows in the bf16 CSV |
| `bpreshuffle` | `B.is_shuffled` attribute | Tag lost by a `.view()` or rebuild |

## Fused MoE lookup (`aiter.fused_moe`)

Same pattern, wider key: `(gfx, cu_num, token bucket, model_dim, inter_dim, expert,
topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)`.
A row names a **stage-1 and a stage-2 kernel** plus `block_m` and `ksplit`. Older CSVs
have no `gfx` column; the merge back-fills it from `cu_num`. That back-fill cannot tell
apart two archs with the same CU count, so regenerate on the target instead of relying on it.

## Config files and overrides

```
AITER_CONFIG_<FAMILY> set?
├─ yes → use exactly the os.pathsep-joined list you gave (shipped file NOT auto-added)
└─ no  → shipped aiter/configs/<name>.csv
         + every aiter/configs/model_configs/*<name>*.csv (auto-merged)
merge: union of columns → back-fill gfx → de-dup on untuned-key + cu_num (+gfx)
       → duplicate shape? rewrite sources keeping lowest `us`, then RAISE
       → else merged copy written to a temp dir
```

The duplicate check runs over the whole family, so a per-model CSV that is fine alone
can collide with another model's CSV. Setting the env var also skips the
`model_configs/` scan.

| Family | Env override |
|---|---|
| bf16/fp16 dense GEMM | `AITER_CONFIG_GEMM_BF16` |
| fp8/int8 GEMM (plain, bpreshuffle, blockscale, ...) | `AITER_CONFIG_GEMM_A8W8*` (one per variant) |
| fp4 block-scale GEMM | `AITER_CONFIG_GEMM_A4W4` |
| batched GEMM | `AITER_CONFIG_BF16_BATCHED_GEMM`, `AITER_CONFIG_A8W8_BATCHED_GEMM` |
| fused MoE | `AITER_CONFIG_FMOE` (and a grouped-MoE variant) |

Each quant format has its **own** CSV and env var. A bf16 override does nothing for fp8
or fp4 GEMM. Read `aiter/jit/core.py` for the current list.

## Prove engagement before you trust a delta

1. Run with `AITER_LOG_TUNED_CONFIG=1`. Hits log `... is tuned on cu_num = N in <file>,
   libtype is ...`; misses log `not found tuned config ... will use default config`.
   Hit count must be > 0 for the shapes that dominate GPU time.
2. Check the kernel trace (`rocprofv3 --kernel-trace`). The intended kernel name must
   appear, not a Triton or `torch` fallback.
3. Check the framework gate (`SGLANG_USE_AITER=1` / `VLLM_ROCM_USE_AITER=1`) and, on
   vLLM, that the op stayed an opaque custom op through `torch.compile`.
4. Only then A/B end to end.

Two hooks that look like GEMM tuning but **do not reach** aiter's path: PyTorch
TunableOp and `HIPBLASLT_TUNING_FILE`. Both hook PyTorch's own `addmm` dispatch, which
aiter bypasses by calling hipBLASLt directly.

## Portability

`solidx` and `kernelName` belong to one aiter + ROCm + hipBLASLt build. A CSV is not a
portable artifact. Re-tune after any upgrade. A `flydsl` row also needs the FlyDSL
package on the serving machine.

Sources: [`aiter/tuned_gemm.py`](https://github.com/ROCm/aiter/blob/main/aiter/tuned_gemm.py) ·
[`aiter/jit/core.py`](https://github.com/ROCm/aiter/blob/main/aiter/jit/core.py) ·
[`aiter/fused_moe.py`](https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py) ·
[`aiter/ops/gemm_op_common.py`](https://github.com/ROCm/aiter/blob/main/aiter/ops/gemm_op_common.py)
Paths drift; verify before use.

---
Related: [architecture](architecture.md) · [tuning-workflow](tuning-workflow.md) · [pitfalls](pitfalls.md)
