---
name: isa-resource-inspection
description: the static analysis layer (IR analysis, code-object ISA analysis, per-variant report), what to count, what to extract (VGPR/SGPR/LDS/spill/occupancy), and how to interpret — with a worked intrinsic-count diff.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 11. ISA And Resource Inspection

A static analysis layer on top of the compiler output answers most
questions without a GPU run. It should:

- Count MFMA / async / raw-buffer ops, barriers, and waitcnts in the
  LLVM IR.
- Disassemble the code object (`llvm-objdump -d`) for an opcode tally,
  and read `llvm-readelf --notes` for VGPR / SGPR / AGPR / LDS / spill.
- Join compile, IR, ISA, and benchmark data into one per-variant
  report, and sort variants into comparison rows.

Interactive probes on top of this give three views: an intrinsic-count
view (IR-level, custom intrinsic table), an ISA-inspection view (opcode
mix with VALU/SALU sub-buckets), and an occupancy view (readelf notes
plus an occupancy estimate).

### 11.1 What To Count

- MFMA / WMMA instructions.
- Global loads / stores.
- LDS reads / writes.
- Barriers.
- Waits.
- Scalar ALU instructions.
- Vector ALU instructions.
- Branches.
- Atomics.
- Conversion instructions (`v_cvt_*`, `cvt_pk_*`).

### 11.2 What To Extract

- VGPR count.
- SGPR count.
- LDS bytes.
- Scratch / spill usage.
- Occupancy metadata.
- Code object target.
- Workgroup size.
- Waves per workgroup.

### 11.3 How To Interpret

- High VGPR with low occupancy may need fewer accumulators or a
  smaller tile.
- High barriers with little compute per phase suggests bigger
  per-barrier work or fewer phases.
- Many scalar stores indicate an epilogue problem.
- Many address instructions indicate offset math should be hoisted /
  precomputed.
- Register spills usually invalidate performance conclusions until
  fixed (the occupancy probe flags `spill > 0`).
- If MFMA count is higher than expected, inspect loop unrolling and
  instruction choice.

Example diff (from running the intrinsic-count probe on the
32x32-MFMA lever of a tiled attention kernel):

```text
baseline_mw16 → mfma32_transposed:
  mfma.f32.16x16x32.bf16:   17 → 0   (-17, replaced)
  mfma.f32.32x32x16.bf16:    0 → 17  (+17)
  ds.swizzle:               32 → 0   (-32, replaced)
  ds.bpermute:               0 → 67  (+67)
  fmul (LLVM IR):           53 → 113 (+60)
```

The diff makes the algorithmic change visible: the `transposed`
variant swaps an `ds.swizzle`-based softmax butterfly for an
`ds.bpermute`-based one. The next question — "is the bpermute path
faster on this shape?" — is for the targeted-bench probe.

## Lessons from worked examples

- **The per-iteration opcode histogram is the unit of design comparison.** Latency compresses differences. Counting opcodes per loop iteration (MFMA count and size, LDS writes, accumulator↔VGPR moves, cross-lane permutes, waits, barriers) makes a structural difference explicit. Count the reference's loop too; it shows what it does differently faster than inferring from counters.
- **A change that does not move the hot-loop ISA changed nothing.** If the before/after loop bodies are identical, the "win" is noise (e.g. removing an LDS allocation the compiler had already eliminated). The same test catches a documented knob that no code reads: the binary and the lowered IR come out byte-equal with the flag on and off.
- **Count layout conversions, not only loads.** A register-layout conversion that moves data across lanes lowers to an LDS write, a barrier, and an LDS read. Two conversions where one would do doubles that LDS traffic, so count them in the ISA. ([ROCm workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html))

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#isa-resource-inspection).

---
Interactive probes on top of this: [dsl-probe-workflow](../../dsl/ck-dsl/dsl-probe-workflow.md). Probe output fields: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md).

