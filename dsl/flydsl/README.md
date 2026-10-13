# FlyDSL

FlyDSL is a Python DSL for AMD GPU kernels ([ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)). It uses CuTe-style layouts and an MLIR Fly → ROCDL → HSACO pipeline. The general method is in [`../../kernel-optimization/`](../../kernel-optimization/README.md).

Reading order:

- [layout-algebra](layout-algebra.md) — CuTe layouts in FlyDSL: shape:stride, the four divides, TV layouts, swizzle as a composed layout, preshuffle as a view.
- [kernel-authoring-primitives](kernel-authoring-primitives.md) — Boundary types, copy/MMA atoms, the tiled G→LDS→MFMA→global chain, buffer tensors, LDS allocators, control flow.
- [compilation-pipeline](compilation-pipeline.md) — AST rewrite to HSACO: the pass list, what the JIT cache misses, IR/ISA dumps, ATT line info, which hints reach LLVM.
- [pitfalls](pitfalls.md) — Trace-time vs runtime traps, loop bound rules, unit and operand-order bugs, hangs, an isolation workflow.
- [lds-and-prefetch-idioms](lds-and-prefetch-idioms.md) — Ping-pong LDS as a layout mode, swizzle/padding layouts, G→LDS atoms, loop-carried register prefetch and its ISA check.
- [hot-loop-scheduling](hot-loop-scheduling.md) — How to tell if a hot-loop `sched_*` schedule works: the measure, diagnose, re-measure loop and the roofline gate.
- [operator-recipes](operator-recipes.md) — Preshuffle GEMM tile algebra and data paths, two-stage fused MoE (gate modes, split-K, combine modes), wave64 block reduction.
- [varlen-idioms](varlen-idioms.md) — Varlen/jagged kernel idioms: device scalar reads, runtime base offsets, early exit, bounded buffers, bias epilogue.
