---
name: decision-heuristics
description: lever-direction heuristics (if X regresses, try Y) and the anti-pattern list (chasing flags before ISA, trusting one run, promoting from a smoke set, removing a cost without checking where it moved).
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 16. Decision Heuristics

### 16.1 Lever-direction heuristics

- If the output dimension is smaller than MFMA width, consider direct
  / specialized mapping (e.g. a 4-channel direct conv with the `4x4x4` atom).
- If an epilogue uses scalar stores, vectorize it before deeper
  changes.
- If global-to-LDS uses register intermediate, try direct async copy
  (a double-buffered async pipeline).
- If barriers dominate, increase work per barrier or reduce stages.
- If wider tiles slow down, inspect extra LDS passes and register
  pressure.
- If smaller workgroups speed up, occupancy was likely limiting.
- If smaller workgroups slow down, data sharing / barrier amortization
  may dominate.
- If swizzle does not help through register staging, it may still
  matter only with the matching async distribution.
- If LDS epilogue slows down, direct vector stores may already be
  good enough.
- If compiler flags produce wrong answers, remove them even if
  another codebase uses them safely.
- If the HIP-debug backend disagrees with LLVM-direct on a kernel,
  one of the two is missing an op lowering — file the issue,
  don't ignore it.

### 16.2 Anti-patterns (what to avoid)

- Chasing compiler flags before inspecting IR / ISA.
- Trusting one benchmark run.
- Comparing kernels with different math or masks.
- Comparing compile + launch time to warm launch time.
- Ignoring failed correctness because the speed number is attractive.
- Adding compatibility shims for unshipped experimental branches
  instead of fixing the builder.
- Promoting a kernel variant from a smoke set without a full-cohort
  sweep.
- Removing one cost without checking whether it moved to a worse
  place in the per-iter ISA composition.

## Lessons from worked examples

- **Co-evolve levers when the first one regresses.** Moving to an atom whose output layout matches the next stage, keeping the intermediate in registers, and using the matching transposed LDS reader only win together; any two of the three lose to the original. A first-lever regression is often an exposed downstream cost, so look at what it uncovered before reverting.
- **The dispatcher is a lever.** One variant rarely wins every shape (masked vs unmasked, small vs large batch). Add a per-shape selector instead of tuning one variant to the average; it closes the long tail.
- **Know the bandwidth ceiling before tuning further.** Sustained streaming bandwidth can sit far below the marketed peak (e.g. ~44% for a skinny bf16 GEMM on MI355X, where the vendor kernel reached 42%); measure the ceiling for your access pattern. If your kernel and the vendor kernel both sit there, remaining gains come from moving fewer bytes (fusion, lower-precision weights), not from the GEMM body. Check the byte total and the floor time first.
- **Do not add a bandwidth lever to a compute-throttled kernel.** Lower-precision storage and larger reuse tiles cut bytes, but if memory stalls are near zero the extra dequantize work lands on the already-busy vector ALU and the result is neutral or worse.
- **When launches dominate, remove host syncs before you capture a graph.** If the timeline shows thousands of tiny kernels per step, graph replay can remove most of the host cost. Capture needs fixed shapes and no device-to-host reads in the hot path; one scalar read per loop iteration both stalls the stream and blocks capture. Move shape decisions to the host or pad to fixed buckets first. The same rule holds for JIT paths that specialize on shape: a dimension that grows each step recompiles per bucket and can end slower than eager.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#decision-heuristics).

---
The one-page decision tree: [diagnostic-decision-tree](../method/diagnostic-decision-tree.md).

