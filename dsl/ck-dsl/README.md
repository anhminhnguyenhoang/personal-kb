# CK DSL (ck_dsl)

ck_dsl is the Composable Kernel Python DSL in ROCm/rocm-libraries.
The general method from its optimization runbook now lives in [`../../kernel-optimization/`](../../kernel-optimization/README.md).
This folder keeps only the ck_dsl-specific parts.

Upstream: [optimization_runbook.md @ 1efeadd](https://github.com/ROCm/rocm-libraries/blob/1efeadd2f64d70c3f0d5cd53b1b880da718a47a7/projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md) (branch users/vanantha/ck-dsl-prototype). Paths and symbols drift; verify before use.

## Notes

- [primitive-map](primitive-map.md) — Where each general kernel-optimization concept lives in ck_dsl: instances, helpers, builder calls, probes.
- [reproducible-commands](reproducible-commands.md) — Exact environment and harness invocations: the validation block, probe quick-shots, profiler attach, LDS and occupancy tools.
- [dsl-probe-workflow](dsl-probe-workflow.md) — The static probe sequence from build smoke to profiler confirm, with per-step cost, programmatic use, and output fields.
- [knob-catalog-and-sweep](knob-catalog-and-sweep.md) — Every ck_dsl performance lever by family, sweep discipline, the dispatcher as a lever, and variant naming.
