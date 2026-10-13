---
name: pipelining-scheduling
description: software pipeline (prologue/steady/epilogue, ping-pong, early-V), async copy and wait placement, waits/barriers, scheduling hints (compile-time only, no ISA encoding), and instruction-level balance.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 8. Pipelining And Scheduling

### 8.1 Software Pipeline

- Separate prologue, steady-state loop, and epilogue. A software-pipeline helper
  that runs ping-pong staging keeps this structure explicit.
- Prefetch next tile while computing current tile.
- Use ping-pong buffers (typically two LDS halves).
- Use more stages only if latency is not hidden.
- Keep pipeline state small.
- Ensure waits occur as late as safely possible.
- Avoid barrier before useful independent compute.
- Order async issues to give *all* dependent compute a chance to
  overlap. The "early-V" schedule in a unified-attention worked example moved the V async
  copy issue from after QK to before QK so V overlaps with both QK
  and softmax, not just softmax.

### 8.2 Async Copy

- Prefer direct global-to-LDS instructions if available
  (an async tile loader that issues buffer-load-to-LDS).
- Track outstanding async operations.
- Place waits just before consumers (`s_waitcnt vmcnt(0)`).
- Ensure boundary lanes write zeros or that suppressed loads do not
  leave stale LDS.
- Understand whether async instruction writes lane-contiguous or
  arbitrary addresses (lane-contiguous on AMD; see Section 6.3).
- If arbitrary swizzle is needed, verify the intrinsic supports it.

### 8.3 Waits And Barriers

- Count `s_waitcnt` and `s_barrier` with the ISA-inspection probe.
- Distinguish global memory waits from LDS waits.
- Use barriers only for cross-thread visibility.
- `s_waitcnt` does NOT replace a barrier when another thread reads
  your LDS write.
- Avoid barrier after direct global store unless needed.
- Collapse multiple waits where possible.
- Avoid compiler-generated over-waiting by separating phases.
- `lgkmcnt` is a 4-bit field on gfx9 (max 15). The immediate `20336`
  (`0x4F70`) decodes to vmcnt 16, expcnt 7, lgkmcnt 15, not lgkmcnt 16;
  decode immediates against the ISA field layout.

### 8.4 Scheduling Hints

- Try compiler scheduling flags only after correctness is stable.
- On AMD, experiment with `sched_group_barrier(mask, count, sync_id)`,
  `s_setprio`, and `s_waitcnt`, ideally driven by one schedule-policy
  object.
- Compare ISA with and without hints (the ISA-inspection probe,
  the intrinsic-count probe).
- Validate correctness after every scheduler flag.
- Some flags can break generated kernels even when they work for
  hand-written HIP.

`sched_barrier` / `sched_group_barrier` / `iglp_opt` are compiler
scheduler directives with no ISA encoding on any arch (gfx942 and
gfx950 alike). They appear only as `;` comments in the assembly, so a
count of 0 in the final ISA is expected, not a sign they were dropped.
Judge a hint by the instruction order it produces.
([LLVM AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html))

### 8.5 Instruction-Level Balance

- Interleave MFMA, LDS reads, and global loads.
- Avoid long runs of memory instructions followed by long runs of
  MFMA if latency is exposed.
- Avoid immediate use of a just-loaded value if independent work
  exists.
- Group enough MFMA between waits.
- Watch scalar ALU address arithmetic (the ISA-inspection probe's
  `salu` sub-bucket grows if the address math is not hoisted).

## Lessons from worked examples

- **Prefetch only helps a kernel that is waiting per tile.** Extra buffering and `vmcnt` tuning hide per-tile latency. If the kernel is already at the system-wide outstanding-load / bandwidth ceiling, a second in-flight tile changes nothing (bit-exact, same time). Quick test: best ≈ median with <1% spread on the simple loop means there is no per-tile stall to hide.
- **Scheduling-tier knobs do not compose into a structural win.** Pipeline variant, persistent grid and occupancy hints stacked on a bandwidth-bound kernel give noise. Persistent CTAs have nothing to amortize when there are only tens of tiles. Spend the effort on tile geometry and bytes moved first.

- **A deeper buffer needs a pipeline that reads it, and one K-tile can break a 2-stage loop.** Allocating a third stage buffer is a no-op unless the pipeline actually cycles through it. A tile depth equal to the whole reduction (a single K-tile) can silently mis-accumulate in a 2-stage double-buffer epilogue. Cosine similarity hid a 123% relative error, so also check mean signed error.
- **LDS size caps pipeline depth.** LDS use grows as tile bytes × stages. With 64 KB LDS (CDNA3), a GEMM tile typically fits 2 stages, and two fused GEMMs fit 1. With 160 KB (CDNA4), 3 to 4 stages fit. Size the stages before you tune the schedule. ([ROCm workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html))

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#pipelining-scheduling).

---
Pipeline knobs (mem/compv3/compv4): [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md). Async DRAM->LDS contract: [memory-hierarchy](memory-hierarchy.md). gfx950 compiler caveats: [target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md).

