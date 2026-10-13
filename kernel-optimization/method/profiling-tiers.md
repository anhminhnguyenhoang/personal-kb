---
name: profiling-tiers
description: Four profiling tiers on AMD CDNA (timeline, PMC counters, PC sampling, thread trace), what question each answers, how to read Speed-of-Light and the roofline, and how to tell occupancy-bound from latency-bound.
source: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html ; https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html ; https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/ ; Williams, Waterman, Patterson, "Roofline", CACM 2009
---

# Profiling tiers: from "which kernel" to "which instruction"

Each tier answers one question. Run them in order. Go down a tier only
when the tier above gives a clear answer.

```text
Tier 0  roofline target      what is the best possible time?        (paper math, no GPU)
Tier 1  timeline             which kernel / stage costs the time?    rocprofv3 --sys-trace / --kernel-trace
Tier 2  PMC counters         which subsystem stalls?                 rocprofv3 --pmc  |  rocprof-compute SoL
Tier 3  PC sampling          which instruction does the wave sit on? rocprofv3 --pc-sampling-*
Tier 4  thread trace (ATT)   per-wave, per-instruction timeline      rocprofv3 --att (needs the trace decoder)
```

## Tier 0: compute the target before you profile

```text
ops        = algorithmic FLOPs (or bytes, for copy / gather)
bytes      = minimum HBM traffic (each operand read once, output written once)
AI         = ops / bytes
ceiling    = min(peak_FLOPs_for_this_opcode, peak_BW * AI)
target_us  = ops / ceiling
```

- Use the peak for the real matrix opcode and dtype. An fp8 kernel judged
  against the fp32 roof looks bad for no reason.
- Compute one target per stage. Stages of one op (main kernel, reduce,
  epilogue kernel) often have different AI.
- The target is a bar for "done", not a promise. Sustained peaks sit below
  datasheet peaks; measure them (Tier 2 roofline) when you can.

## Tier 1: timeline

- Rank kernels by share of total device time. Work on the stage that is
  far from its own target, not on the largest absolute bar if that bar is
  already at its roof.
- Many short kernels with gaps between them is a launch problem, not a
  kernel problem. Fix it with fusion, graph replay or a persistent kernel.
- Confirm the exact kernel name here. Use it for the kernel filter in
  later tiers (`kernel_include_regex` in the rocprofv3 input file). A
  loose filter collects the wrong kernel.

## Tier 2: counters

Two ways in:

| Tool | Gives | Use when |
|---|---|---|
| `rocprofv3 --pmc` | raw and derived counters per dispatch | you know which counters you want |
| `rocprof-compute profile` + `analyze` | System Speed-of-Light (each engine as % of peak), memory chart, empirical roofline (`--roof-only`) | first look at an unknown kernel |

A counter set must fit in one hardware pass, or the job fails. Split a
wide set into small groups and run each group as its own pass. A
useful split:

| Group | Counters | Tells you |
|---|---|---|
| timing | `SQ_WAVES`, `GRBM_GUI_ACTIVE`, `VALUBusy`, `SALUBusy` | wave count, active cycles, ALU busy |
| memory | `LDSBankConflict`, `MemUnitStalled`, `FetchSize`, `WriteSize` | LDS pressure, HBM bytes |
| mix | `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_LDS`, `SQ_INSTS_MFMA` | instructions per wave |
| L2 | `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR`, `TCC_HIT_sum`, `TCC_MISS_sum` | VMEM mix, L2 hit rate |

Read counters against the algorithm, not against absolute thresholds:

- **Per-wave instruction count ÷ algorithmic minimum.** Divide each
  `SQ_INSTS_*` by `SQ_WAVES`. Count what the algorithm needs (loads, MFMAs,
  softmax ops). A class far above its minimum (e.g. VALU 3× or more) is
  data shuffling: lane re-packing, address math, conversions.
- **`FetchSize` ÷ minimum bytes.** Above ~1.2× with a high L2 hit rate
  means the kernel re-reads data (redundant or predicated loads), not
  that HBM is slow.
- **`LDSBankConflict` with `MemUnitStalled`.** Their ratio says whether
  memory stalls are on the LDS side or the VMEM side.

Read the Speed-of-Light table this way: the engine with the highest %
of peak is the closest roof. If no engine is near its peak, the kernel
is not throughput-bound. It is occupancy-bound or latency-bound.

```text
                 ┌─ near compute roof ──────────────► COMPUTE-BOUND
roofline point ──┼─ near bandwidth roof ────────────► BANDWIDTH-BOUND
                 └─ far from both ─┬─ few waves/CU (VGPR, LDS, grid) ► OCCUPANCY-BOUND
                                   └─ waves OK, stalls high ─────────► LATENCY-BOUND
```

Occupancy-bound and latency-bound look the same on the roofline. Only
waves-per-CU and the stall share separate them. A "bandwidth-bound"
verdict against the HBM roof may be wrong if the working set lives in
L2 or the last-level cache; check the cache roofs too.

## Tier 3: PC sampling

The profiler samples the program counter of resident waves at a fixed
interval (host-trap method; some Instinct parts also support a
stochastic, hardware-based method). Bucket the samples by instruction
class from the disassembly: `s_waitcnt`, `v_mfma*`, `ds_*`, VMEM
(`buffer_*`, `global_*`), other VALU, SALU, branches.

| MFMA share | `s_waitcnt` share | Reading | Next step |
|---:|---:|---|---|
| high (>50%) | low (<10%) | arithmetic-bound | larger or denser matrix opcode |
| 20–50% | 10–20% | balanced | tune load / MFMA interleave |
| low (<20%) | high (>25%) | wait-bound | cut staging traffic; do not add MFMA throughput |
| low | low | memory-bound | check `FetchSize` and L2 hit rate |

A kernel can sit at its roofline with a low MFMA share. Decode-shaped
attention and short-K GEMMs often do. Read the histogram before you
decide that "more MFMA" is the fix.

Take enough dispatches for coverage; each sample is one PC value. Use a
shorter interval for very short kernels.

## Tier 4: thread trace

ATT gives per-wave, per-instruction issue and stall timing. It needs
the trace decoder library. If the decoder is missing, Tier 2 instruction
mix plus Tier 3 histograms plus static ISA answer almost every question
(see [bottleneck-classification](bottleneck-classification.md)).

## Rules that apply to every tier

- Profiling perturbs time. Take timing from an untraced run, counters
  from a profiled run. Never read speed from a PMC or sampling run.
- Run several iterations and drop the cold first dispatch. Use the median.
- Sweep at least three problem sizes (short / mid / long reduction, small
  / large batch). The bottleneck often changes with size.
- After a change, check that the targeted signal moved in the predicted
  direction, not only the wall time.

## When to stop

Stop when one of these holds:

1. measured time is within ~10% of the Tier 0 target;
2. the wait share is low (<15%) and the binding engine is near its peak;
3. every signal sits at its algorithm-derived reference.

The thresholds above are working heuristics, not vendor limits.
Verify specifics against the primary source.

Sources: [rocprofv3 how-to](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html) · [PC sampling how-to](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html) · [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) · [rocprof-compute performance model](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/performance-model.html) · Williams et al., "Roofline: an insightful visual performance model", CACM 52(4), 2009

---
Related: [bottleneck-classification](bottleneck-classification.md) · [diagnostic-decision-tree](diagnostic-decision-tree.md) · [measurement](measurement.md)
