---
name: the-loop
description: The 10-step optimization loop (hypothesis -> correctness -> measure -> inspect IR/ISA -> one lever -> re-verify -> explain -> keep/revert -> record), and the document map. Read first.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

# Optimization loop and how to read these notes

These notes are a long-form checklist for optimizing GPU kernels.

Every section ties a general optimization concept to a lever you can
change, verify, measure, and explain.

Use it as a menu of considerations. Do not apply every item blindly.
Start from the operation contract and the measured bottleneck.

## The Loop (one-page summary)

If you take only one thing from these notes, take this loop:

```text
1. State the hypothesis.
2. Verify correctness baseline.
3. Measure baseline with stable timing.
4. Inspect generated IR / ISA / resources (static probes).
5. Change one lever (see the lever catalog).
6. Re-verify correctness.
7. Re-measure with the same harness.
8. Explain why it moved (per-iter ISA diff).
9. Keep or revert.
10. Record the result.
```

Do not batch several levers unless you are explicitly doing a coarse
search and plan to isolate the winner afterward. If stuck and no clear winning change with simple levers, extend to multiple levers in smart combinations and use judgement and do not give up and revert losses easily. **If correctness
fails, do not report speed as a win.**

## How to read this document

The notes follow this structure:

1. Define the problem (contract, shapes, layouts, dtypes, boundaries).
2. Establish baselines (correctness, performance, hygiene).
3. Classify the bottleneck (arithmetic intensity, profiler, IR/ISA).
4. Choose the algorithmic mapping.
5. Decompose the work (grid, block, wave, thread).
6. Optimize the memory hierarchy (global, LDS, registers, caches).
7. Pick matrix instructions and operand layouts.
8. Pipeline and schedule.
9. Optimize the epilogue.
10. Tune the compiler.
11. Inspect ISA and resources.
12. Autotune.
13. Apply operation-specific checklists (GEMM / conv / attention / …).
14. Recognize failure modes.
15. Report results.
16. Apply decision heuristics.
17. Read the empirical case studies.

If you came here looking for a specific knob, jump to the knob
catalog: it enumerates the performance levers by family (algorithmic
variant, tile geometry, MFMA atom, pipeline, epilogue, LDS layout,
occupancy, preshuffle, persistent / Stream-K, quantization,
attention micro-levers, chiplet swizzle, compiler flags, runtime /
launch, dispatcher policy, benchmark hygiene, static probes).

If you came here looking for **what is available on gfx950 / CDNA4**
(MI350X / MI355X), jump to the target architecture reference: the
full MFMA atom catalog, LDS specs and bank-conflict rules, CDNA4-only
instructions (`v_permlane32_swap_b32`, `ds_read_tr16_b{64,128}`,
`ds_read_tr_b8`, scaled / MX MFMA), VGPR / AGPR / occupancy caps,
chiplet swizzle parameters, buffer-descriptor flags, fp8 /
quantization support, and gfx950 compiler caveats.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#the-loop).

## Lessons from worked examples

- **Start from the fastest correct reference, not a blank file.** A known-fast implementation already solves operand layout, LDS addressing and softmax precision. Writing from scratch spends days re-solving them. Rebuild the reference unchanged, prove it matches, then change one thing; keep from-scratch work for simple memory-bound kernels (elementwise, reductions).
- **Fix a "done" bar before you start.** Compute the roofline target time for each stage, and stop when the kernel is within ~10% of it or every profiler signal sits at its algorithm-derived reference ([profiling-tiers](profiling-tiers.md)). Without a bar the loop never ends, or it ends at the first small win.

---
Next: [define-the-problem](define-the-problem.md) · jump to [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md) for the lever menu · [reproducible-commands](../../dsl/ck-dsl/reproducible-commands.md).
