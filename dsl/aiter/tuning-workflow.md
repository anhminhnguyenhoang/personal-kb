---
name: tuning-workflow
description: "The aiter optimization loop: profile to find the dominant op, capture live shapes, race backends per shape with an accuracy gate, deploy the CSV by env, prove engagement, A/B end to end, and only then consider writing a kernel."
source: https://github.com/ROCm/aiter/tree/main/csrc/gemm_a16w16 ; https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen ; https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# Tuning workflow

On a serving stack, the first aiter lever is the **per-shape DB**, not kernel source.
It is reversible, needs no code edit, and is same-math for bf16 (parity-safe).

```
 1 profile ──► 2 capture ──► 3 reduce+order ──► 4 race ──► 5 deploy ──► 6 prove ──► 7 A/B
 (rocprofv3     (live shapes   (bucket M,          (per shape,   (env var,     (hit log +   (same session,
  kernel stats)  only)          biggest first)      err-gated)    no edit)      trace)       e2e)
                                                                                    │
                                    plateau + profile shows headroom ◄──────────────┘
                                    ⇒ author a kernel (CK / HIP / Triton / FlyDSL / asm)
```

## Steps

1. **Profile.** `rocprofv3 --kernel-trace --stats` on the real workload. Rank kernels by
   total GPU time. Work on the top rows only; a big win on a tiny op is noise.
2. **Capture live shapes.** `AITER_TUNE_GEMM=1` makes every dense-GEMM call append its
   real key (including `bias`) to the untuned CSV. Never hand-write the untuned file: a
   wrong `bias` or quant field gives a CSV that never matches.
3. **Reduce and order.** Collapse M to the dispatcher's buckets. Sort by FLOPs,
   largest first, so a partial run already covers the prefill shapes that dominate time.
   A partial DB cannot regress: uncovered shapes keep the default.
4. **Race.** The multi-backend dense tuner (`csrc/gemm_a16w16/gemm_a16w16_tune.py`)
   times every candidate of each libtype and keeps the fastest that passes the error gate.

   | Flag | Meaning |
   |---|---|
   | `-i` / `-o` | Untuned input, tuned output CSV |
   | `--libtype` | Subset to race (`all` by default) |
   | `--with-hipblaslt` | Also race hipBLASLt (many solutions per shape: slow) |
   | `--errRatio` | Max fraction of mismatched elements (default 0.05 for GEMM) |
   | `--mp` | Parallel tuning processes (one per GPU) |

   Other families have their own tuners: `csrc/ck_gemm_*/` for scaled GEMM,
   `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` for fused MoE (looser default
   error gate, because quantized MoE stages carry more error).
5. **Deploy.** Point `AITER_CONFIG_<FAMILY>` at the new CSV. Restart the process: tuned
   tables are cached for the process lifetime, and tables baked into C++ need
   `AITER_REBUILD=1`.
6. **Prove engagement.** See [tuned-config-dispatch](tuned-config-dispatch.md#prove-engagement-before-you-trust-a-delta).
7. **A/B end to end.** Same session, interleaved runs. Accept only if the gain clears
   the noise band and accuracy holds. Quantized variants (fp8, fp4, MLA fp8 KV) also need
   a task-accuracy check, not only kernel tolerance.

## Tuning cost control

- A hipBLASLt race can try over a thousand solutions per shape. Over many large shapes it
  spawns enough helper processes to exhaust host memory and distort timing. Bucket M,
  cap `--mp`, and restrict `--libtype` while iterating.
- Capture small M explicitly. Decode GEMMs (M = 1..8) use the skinny path; tuning only
  prefill M misses them.

## Tune or author?

| Profile signal | Action |
|---|---|
| Dominant op, zero tuned hits | Tune the DB |
| Generic Triton or `torch` kernel in the trace for a hot shape | Coverage gap: tune that shape, or check the arch has a native path |
| Tuned and engaged, still far from roofline | Library ceiling: author a kernel in a DSL, then wire it in as a tuned row or a call-site rebind |
| Decode GEMM, bandwidth-bound | Make sure the skinny path is selected before anything else |

An authored kernel only counts if it reaches the live call site **and** moves e2e time.
Commit tuned CSVs per the repo's collision rules: one row per shape across the family.

Sources: [`csrc/gemm_a16w16/`](https://github.com/ROCm/aiter/tree/main/csrc/gemm_a16w16) ·
[`csrc/ck_gemm_moe_2stages_codegen/`](https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_moe_2stages_codegen) ·
[`gradlib/`](https://github.com/ROCm/aiter/tree/main/gradlib) ·
[ROCm workload optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
Paths drift; verify before use.

---
Related: [tuned-config-dispatch](tuned-config-dispatch.md) · [pitfalls](pitfalls.md) · [measurement](../../kernel-optimization/method/measurement.md) · [the-loop](../../kernel-optimization/method/the-loop.md)
