# FlyDSL

FlyDSL is a Python DSL for AMD GPU kernels ([ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)). It uses CuTe-style layouts and an MLIR Fly → ROCDL → HSACO pipeline. The general method is in [`../../kernel-optimization/`](../../kernel-optimization/README.md).

Reading order:

- [layout-algebra](layout-algebra.md) — CuTe layouts in FlyDSL: shape:stride, the four divides, TV layouts, swizzle as a composed layout, preshuffle as a view.
- [kernel-authoring-primitives](kernel-authoring-primitives.md) — Host-built tiled atoms, less common atoms, the example-04 operand chain, API stability.
- [compilation-pipeline](compilation-pipeline.md) — What the JIT cache key misses, which dump file answers which question, which hint lanes reach LLVM.
- [pitfalls](pitfalls.md) — Unrolled-alias prefetch, MFMA operand order, strided torch views, hangs, an isolation workflow.
- [lds-and-prefetch-idioms](lds-and-prefetch-idioms.md) — Ping-pong LDS as a layout mode, swizzle/padding layouts, G→LDS atoms, loop-carried register prefetch and its ISA check.
- [hot-loop-scheduling](hot-loop-scheduling.md) — How to tell if a hot-loop `sched_*` schedule works: the measure, diagnose, re-measure loop and the roofline gate.
- [operator-recipes](operator-recipes.md) — Preshuffle GEMM tile algebra and data paths, two-stage fused MoE (gate modes, split-K, combine modes), wave64 block reduction.
- [varlen-idioms](varlen-idioms.md) — Varlen/jagged kernel idioms: device scalar reads, runtime base offsets, early exit, bounded buffers, bias epilogue.
