---
name: diagnostic-decision-tree
description: the one-page diagnostic decision tree (build smoke -> occupancy -> intrinsic -> targeted bench -> ISA -> rocprof) plus a complementary symptom-to-action table. Intentionally lossy — points at the next section to read.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## Appendix: One-Page Diagnostic Decision Tree

```text
new kernel / regression
   │
   ▼
1) Build smoke (config-sweep probe, build only)
      │
      ├─ SPEC-FAIL → fix coupled config fields (spec validation)
      ├─ BUILD-FAIL → reduce / isolate failing variant; verify IR coverage
      └─ all build  → 2)
   │
   ▼
2) occupancy probe
      │
      ├─ spill > 0     → reduce VGPR pressure (atom, accumulators, unroll)
      ├─ waves < 4     → check LDS budget; try a leaner pipeline; smaller tile_k
      ├─ limited LDS   → recheck swizzle (Section 6.4a); smaller tile or async
      └─ healthy       → 3)
   │
   ▼
3) intrinsic-count probe
      │
      ├─ no MFMA       → wrong atom selected; check Section 7
      ├─ no async DMA  → double-buffered async pipeline not selected (or unsupported shape)
      ├─ s.barrier ≫   → too many phases; collapse (Section 8.3)
      └─ healthy       → 4)
   │
   ▼
4) targeted-bench probe against baseline (Triton / CK Tile / AITER)
      │
      ├─ regression    → 5)
      └─ near-best     → ship & document
   │
   ▼
5) ISA-inspection probe
      │
      ├─ buffer_store_short ≫ → epilogue is scalar; vectorize (Section 9.2)
      ├─ ds_read non-tr ≫     → swizzle isn't paired with reader; pick the right LDS layout
      ├─ waitcnt patterns ≫   → barrier/wait timing not aligned to async DMA
      └─ unclear              → 6)
   │
   ▼
6) rocprofv3 with metrics.txt
      │
      ├─ memory_stall > 40 %  → prefetch / async LDS (Section 8.2)
      ├─ lds_stall > 20 %     → run the LDS-conflict analyzer
      ├─ bandwidth > 80 %     → memory-bound, increase reuse (Section 6.3)
      └─ compute_util > 70 %  → optimize MFMA packing (Section 7.4)
```

This decision tree is intentionally lossy: it points at the next
section to read, not at the final answer. The runbook itself is the
canonical reference; this tree is just the dispatcher.

### Symptom-to-action table (complementary view)

When you have a known symptom and want the typical first action:

```text
Symptom: correct but slow, low MFMA count
Likely:  wrong atom / tile, scalar path, missing vectorization
Action:  inspect LLVM / ISA; check atom selection
         and the inner loop

Symptom: fast but incorrect only on padded / tail shapes
Likely:  invalid pointer load, bad descriptor valid, vector crosses tail
Action:  test tiny adversarial shapes; inspect buffer-rsrc
         sentinel path

Symptom: intermittent wrong answers in async path
Likely:  missing `s_waitcnt` / barrier, workspace lifetime
Action:  add / check `s_waitcnt(vmcnt=0)`, stream sync,
         launcher keep-alive

Symptom: atom change improves ISA but regresses runtime
Likely:  VGPR / LDS occupancy loss, or epilogue bottleneck
Action:  inspect resources via the occupancy probe; try cshuffle vs
         direct epilogue alternatives

Symptom: direct conv close but not within tolerance
Likely:  K-packed lane order or accumulator-reset bug
Action:  compare per-lane small reference; inspect fold order
        

Symptom: low MfmaUtil + high VALUBusy + low MemUnitStalled
Likely:  compute-throttled by VALU / LDS plumbing, not HBM-bound
Action:  per-iter ISA histogram; fix the plumbing
         (epilogue / register-residency / lane layout), not the
         launch knobs

Symptom: removing a per-iter cost regressed throughput
Likely:  the cost moved to a worse place (cross-lane permute,
         VGPR pressure, occupancy drop)
Action:  per-iter ISA diff before/after the change
         — never declare a structural change a win without it

---
Each node maps to a lever section in [02-levers/](../levers) or to [dsl-probe-workflow](../../dsl/ck-dsl/dsl-probe-workflow.md). Heuristics behind it: [decision-heuristics](../failure-modes/decision-heuristics.md).

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#diagnostic-decision-tree).
