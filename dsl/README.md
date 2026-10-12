# Kernel DSLs

DSL-specific notes. The general method lives in [`../kernel-optimization/`](../kernel-optimization/README.md).

- [CK DSL (ck_dsl)](ck-dsl/README.md) — The Composable Kernel Python DSL: primitive map, commands, probe workflow, and knob catalog.
- [hot-loop-scheduling](flydsl/hot-loop-scheduling.md) — How to tell if a FlyDSL hot-loop schedule works: the measure, diagnose, re-measure loop and the roofline gate.
- [varlen-idioms](flydsl/varlen-idioms.md) — Verified FlyDSL layout-API idioms for varlen and jagged kernels: scalar reads, early exit, bounded buffers, bias epilogue.
