---
name: reporting-template
description: the experiment-log table, final-summary template, minimum result record, and done-criteria for an optimization. Never report speed if correctness fails.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 15. Reporting Template

Use this table for experiment logs:

| Variant | Hypothesis | Correct | Time | Throughput | VGPR | SGPR | LDS | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | reference | yes | | | | | | |
| v1 | change one lever | yes/no | | | | | | |

Use this final summary:

```text
Best correct variant:
  name:
  shape:
  latency:
  throughput:
  correctness:
  resources:

Main bottleneck now:
  evidence:

Rejected ideas:
  - idea: reason

Next experiments:
  1.
  2.
  3.
```

For published numbers, attach the manifest + HSACO + a static ISA / resource
analysis (or equivalently, the occupancy-probe and
ISA-inspection-probe reports). See `measured_results.md` for the
last documented validation pass.

### 15.1 Minimum Result Record

For every experiment, record:

```text
kernel/spec name
shape
dtype/layout
GPU and ROCm version
baseline commit/config
variant description
correctness status
max/mean error
latency median
latency spread
TFLOPS or GB/s
VGPR/SGPR/LDS if inspected
notable ISA changes
notes
```

If correctness fails, do not report speed as a win.

### 15.2 Done Criteria For An Optimization

An optimization is done when:

- it has a one-sentence hypothesis;
- correctness passes representative and adversarial shapes;
- benchmark improvement is stable across repeated runs;
- generated IR / ISA confirms the intended primitive changed;
- resource usage is recorded;
- docs or comments explain the new invariant if it is non-obvious;
- unsupported configurations are rejected by validation.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#reporting-template).

---
What to measure: [establish-baselines](../00-method/establish-baselines.md). What to inspect: [isa-resource-inspection](../10-levers/isa-resource-inspection.md).
