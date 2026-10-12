# Kernel optimization

A DSL-neutral method for tuning GPU kernels. DSL-specific mappings live in [`../dsl/`](../dsl/README.md). Many notes end with a `Lessons from worked examples` section.

## 00-method

- [the-loop](00-method/the-loop.md) — The 10-step optimization loop from hypothesis to record, plus the document map. Read first.
- [define-the-problem](00-method/define-the-problem.md) — Pin the operation contract, shapes, layouts, dtypes, tolerances, and boundary conditions before choosing how the kernel runs.
- [establish-baselines](00-method/establish-baselines.md) — Layered correctness gates, vendor baselines, benchmark hygiene, and metadata to record. Never report speed without correctness.
- [measurement](00-method/measurement.md) — Benchmark discipline against an autotuned baseline: use min or p10, not median; cold versus hot L2; change one thing at a time.
- [bottleneck-classification](00-method/bottleneck-classification.md) — Estimate arithmetic intensity, read profiler counters, use cheap static probes first, and sort kernels into compute, memory, sync, or launch bound.
- [diagnostic-decision-tree](00-method/diagnostic-decision-tree.md) — One-page diagnostic tree and a symptom-to-action table. It points to the next note to read.

## 10-levers

- [algorithmic-mapping](10-levers/algorithmic-mapping.md) — Pick the kernel before the knobs: instance tables for GEMM, convolution, attention, reduction, transpose, gather, and the implicit-GEMM register cost.
- [work-decomposition](10-levers/work-decomposition.md) — Grid, block, wave, and thread mapping; chiplet swizzle; wave64 block sizes; MFMA lane mapping and the K-pack bug.
- [chiplet-xcd-remap](10-levers/chiplet-xcd-remap.md) — How a library's PID remap helpers map onto the HipKittens chiplet algorithm, the safe defaults, and the split-K gap.
- [matrix-instructions](10-levers/matrix-instructions.md) — MFMA and WMMA atom choice, operand layout matching across chained atoms, accumulator strategy, and fewer MFMAs through K-pack.
- [memory-hierarchy](10-levers/memory-hierarchy.md) — Registers, LDS, caches, and HBM; coalescing, async loads to LDS, double buffering, swizzle, and bank conflicts.
- [pipelining-scheduling](10-levers/pipelining-scheduling.md) — Software pipelines, ping-pong, async copy and wait placement, barriers, and scheduling hints that gfx950 may drop.
- [epilogue](10-levers/epilogue.md) — Epilogue ops, direct versus LDS epilogue trade-offs, and output validation. Vectorizing the epilogue is often the largest win.
- [compiler-build](10-levers/compiler-build.md) — Build type, common HIP flags, DSL compiler hazards, and alias semantics. Flags rarely close gaps above 1%.
- [isa-resource-inspection](10-levers/isa-resource-inspection.md) — Static IR and ISA analysis: what to count and extract (VGPR, SGPR, LDS, spills, occupancy), with a worked example.

## 20-autotuning

- [op-specific-checklists](20-autotuning/op-specific-checklists.md) — Checklists for GEMM, convolution, attention, reduction, and fused ops: questions to ask before and during tuning.

## 30-failure-modes

- [failure-modes](30-failure-modes/failure-modes.md) — Correctness failures, performance failures, and benchmark failures, with their causes.
- [decision-heuristics](30-failure-modes/decision-heuristics.md) — If X regresses, try Y; and the anti-patterns: flags before ISA, one run, smoke-set promotion, unchecked cost moves.
- [reporting-template](30-failure-modes/reporting-template.md) — Experiment log table, final summary template, minimum result record, and done criteria.
