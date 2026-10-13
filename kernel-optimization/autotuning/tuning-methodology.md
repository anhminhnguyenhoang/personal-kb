---
name: tuning-methodology
description: How to run a per-shape autotuning campaign that actually reaches production. Tuner tiers, exact lookup keys and silent misses, capturing real shapes, search-space pruning, accuracy gates, and when to re-tune.
source: https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html ; https://github.com/ROCm/hipBLASLt
---

# Autotuning methodology

An autotuner races candidate configs per shape and stores the winner.
The search is the easy part. Most failed campaigns tune the wrong shapes,
or tune a table that the deployed code never reads.

```text
capture live shapes ─► bucket ─► prune ─► race (+ accuracy gate) ─► store table
                                                                      │
             verify the deployed path hits the table ◄────────────────┘
                              │
                     end-to-end A/B in one session
```

## 1. Know which tier the deployed path reads

| Tier | What it tunes | Reaches production when |
|---|---|---|
| author-time | config list inside one kernel (tiles, stages, waves, atom size) | that kernel is what the deployed path dispatches |
| library offline | the library's solution index per shape | the deployed path calls that library entry point |
| dispatch table | a per-shape table read by a dispatcher at run time | the dispatcher sits on the live call path |

A frontend that bypasses a library's tuning hook ignores that hook's
tuned file. Find the tier that is on the live call path first.

## 2. The lookup key must match the live call exactly

A tuned entry is found by a key such as
`(CU count, M, N, K, bias, input dtype, output dtype, scale mode, layout/preshuffle)`.
One field that differs from the live call (commonly a bias flag tuned
`true` while the live call is `false`) gives a 100% miss. Nothing fails;
the default config just runs.

- Capture shapes from the real workload, warm, with the same flags, not
  from a synthetic script.
- After deploying, prove the table is hit: a dispatcher log line, or the
  tuned kernel's name in the profiler's kernel list. Zero hits means a key
  mismatch.

## 3. Shrink the search space

- **Bucket the varying dimension.** Batch-like M changes per call. Round
  it to a small set of buckets and tune those, not every M.
- **Start from the architecture tuning guide's defaults**, then search
  around them: tile sizes that are multiples of the matrix atom, enough
  workgroups to fill every CU and chiplet, a `waves_per_eu` hint only if
  it does not spill.
- **Prune statically before timing.** Drop configs that spill, that fall
  below the occupancy you need, or that launch fewer workgroups than CUs.
  A resource report costs well under a second per config (see
  [isa-resource-inspection](../levers/isa-resource-inspection.md)).
- **Prefer a per-shape selector over one config tuned to the average.**
  One config rarely wins every shape
  (see [decision-heuristics](../failure-modes/decision-heuristics.md)).

## 4. Race fairly and gate on accuracy

- Time each candidate on device time, with warmup and cache-busting input
  rotation; compare min or p10 (see [measurement](../method/measurement.md)).
- Reject any candidate whose error against the reference is above the
  op's tolerance. A faster wrong config must never enter the table
  (see [numerical-stability](../levers/numerical-stability.md)).
- Very large candidate pools are slow and can exhaust host memory or
  processes. Prune first, then parallelize across GPUs.

## 5. Store, version, and re-tune

- Keep tuned tables outside the installed package, versioned per
  (workload, dtype, hardware SKU). Load them through configuration, not by
  editing installed files.
- Re-tune when any key input changes: shape set, dtype, library or
  compiler version, or SKU. CU count is part of the key; parts of one
  family can have different CU counts.
- Persist in-process autotune caches. The first call per shape pays the
  full search, so keep that out of timed or captured regions.

## 6. Prove the win end to end

Run the full workload with and without the tuned table, back to back in
one session, several repeats. Accept the change only if the bands do not
overlap. Summing per-kernel gains misses overlap and caching effects.

Sources: [MI300X workload tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html) (GEMM tuning, TunableOp, kernel tuning knobs) · [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt)

## Lessons from worked examples

- **Calibrate wide, then sweep narrow.** Run the full search space on two representative sizes. If the winners fall inside a small sub-range, sweep only that sub-range across all sizes. This cuts tuning cost several-fold (about 6× in one public guide) with little loss. ([MoRI EP guide](https://github.com/ROCm/mori))

---
Related: [op-specific-checklists](op-specific-checklists.md) · [measurement](../method/measurement.md) · [reporting-template](../failure-modes/reporting-template.md)
