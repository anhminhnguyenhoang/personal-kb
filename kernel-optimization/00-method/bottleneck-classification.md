---
name: bottleneck-classification
description: estimate arithmetic intensity, rocprof PMC counters + bottleneck decision tree, the static-probe tier (cheap GPU-free signal first), and the compute/memory/sync/launch-bound signal lists.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 3. First Bottleneck Classification

### 3.1 Estimate Arithmetic Intensity

- For conv, derive `tflops` and `gbps` from the problem metrics.
- For GEMM variants, derive FLOPs from the problem shape.
- For paged attention compute the working bytes per head per token
  from the problem dims; a good runner prints both `TFlops` and
  `GB/s` per launch.
- Compare to the hardware balance point. Look up the architecture's
  peak FLOP/s for the relevant dtype and peak HBM bandwidth from the
  vendor spec sheet, then compute `balance = peak_flops / peak_bw`
  in FLOP / byte. Below the balance point → memory-bound; above →
  compute-bound.

### 3.1a ROCm Profiler Metrics (Hardware Counters)

Use `rocprof` / `rocprofv3` to collect hardware performance counters.
These are critical for identifying *real* bottlenecks, not theoretical
ones.

#### Essential Profiling Setup

Create `metrics.txt`:

```text
pmc: SQ_WAVES
pmc: SQ_WAVE_CYCLES
pmc: SQ_BUSY_CU_CYCLES
pmc: SQ_INSTS_MFMA
pmc: SQ_ACTIVE_INST_MFMA
pmc: TCC_EA_RDREQ_32B
pmc: TCC_EA_RDREQ_64B
pmc: TCC_EA_WRREQ_32B
pmc: TCC_EA_WRREQ_64B
pmc: TCC_HIT
pmc: TCC_MISS
pmc: SQ_WAIT_INST_LDS
pmc: TCP_PENDING_STALL_CYCLES
pmc: LDS_BANK_CONFLICT
```

Then run profiling against the kernel runner:

```bash
sudo rocm-smi --setperflevel high
sudo rocm-smi --setsclk 7

rocprofv3 -i metrics.txt -o output.csv --stats --kernel-trace -- \
    <kernel-runner> "$HSACO" "$MANIFEST" \
    --shape "..." --warmup 5 --iters 100
```

The `--kernel-trace` flag adds VGPRs, AGPRs, LDS bytes, occupancy
percentages directly to the CSV. Pair with a
register-usage profiler for a higher-level view per kernel config.

#### Critical Metrics and Interpretation

| Metric | Meaning | Action threshold |
|---|---|---|
| `DurationNs` | wall-clock per launch | minimize |
| `TCC_EA_*` | HBM read/write transactions | >80 % peak → bandwidth-bound |
| `TCC_HIT / TCC_MISS` | L2 hit rate | <60 % hit → enlarge tiles |
| `SQ_INSTS_MFMA / SQ_ACTIVE_INST_MFMA` | MFMA issue rate | >70 % → compute-bound |
| `SQ_BUSY_CU_CYCLES` ratio | occupancy | <0.3 → fix VGPR/LDS pressure |
| `TCP_PENDING_STALL_CYCLES` | memory latency stall | >40 % → add prefetch stages |
| `SQ_WAIT_INST_LDS` | LDS stall | >20 % → swizzle / pad |
| `LDS_BANK_CONFLICT` | LDS bank conflict count | >0 → analyze with the LDS-conflict analyzer |

#### Bottleneck Decision Tree (Hardware-Driven)

```python
# After collecting rocprof metrics:
PEAK_GFLOPS = ...   # peak FLOP/s for the active dtype, from spec sheet
PEAK_BW_GBS = ...   # peak HBM bandwidth, from spec sheet

compute_util = (measured_gflops / PEAK_GFLOPS) * 100
memory_util  = (measured_bandwidth_gb_s / PEAK_BW_GBS) * 100

if occupancy < 0.5:
    bottleneck = "OCCUPANCY_BOUND"
    # Action: reduce VGPR (use a leaner pipeline), reduce LDS (smaller tile_k)
elif memory_stall_pct > 40:
    bottleneck = "MEMORY_LATENCY_BOUND"
    # Action: increase prefetch stages, use async LDS loads
elif memory_util > 70:
    bottleneck = "MEMORY_BANDWIDTH_BOUND"
    # Action: increase tile sizes (more data reuse)
elif compute_util > 70:
    bottleneck = "COMPUTE_BOUND"
    # Action: optimize MFMA packing (e.g., K32 folding)
elif lds_stall_pct > 20:
    bottleneck = "LDS_BOUND"
    # Action: enable XOR swizzle, add LDS padding
else:
    bottleneck = "BALANCED_OR_LAUNCH_BOUND"
```

| Symptom | Likely Cause | Lever |
|---|---|---|
| Occupancy < 0.5 | VGPR or LDS too high | leaner pipeline, smaller K tile, set a waves-per-EU hint |
| Memory stall > 40 % | Memory latency | double-buffered async pipeline with async tile loads, more stages |
| Bandwidth > 80 % peak | Bandwidth-bound | enlarge the M / N tile for reuse |
| MFMA util < 50 % | Memory bottleneck | check stall metrics first |
| LDS stall > 20 % | Bank conflicts | LDS padding or XOR swizzle (Section 6.4a) |
| L2 hit rate < 60 % | Poor locality | enlarge tile sizes |
| Arithmetic intensity < 10 FLOP/byte | Memory-bound | enlarge blocking factor |

#### When the ATT decoder is unavailable

If `rocprof-trace-decoder` is not installed on the host, `rocprofv3
ATT` and the `kernel-trace-analysis` skill cannot run. The equivalent
diagnostic signal comes from one `rocprofv3` PMC pass with the
counter set above, plus static ISA via `llvm-objdump` (or
the ISA-inspection probe), plus the kernel-stats header for VGPR / LDS
/ occupancy. This combination answers every row of the bottleneck
decision tree without ATT. In a unified-attention worked example,
this fallback identified a single-digit `MfmaUtil`, an
above-threshold `LDSBankConflict`, and a near-zero `MemUnitStalled`,
and ruled HBM out of the gap on the first profiling pass.

### 3.1b Static Inspection First — The Static Probe Tier

Before paying for a full rocprof run, exercise static probes:

| Question | Static probe | Cost |
|---|---|---|
| Will it fit at the expected occupancy? | the occupancy probe | ~0.5 s/variant (compile + readelf) |
| Are the right intrinsics emitted? | the intrinsic-count probe | ~0.05 s/variant (lower only) |
| What is the opcode mix? | the ISA-inspection probe | ~0.5 s/variant (compile + objdump) |
| Does HIP-debug agree with LLVM-direct? | the lowering-compare probe | ~10 s/variant (hipcc) |
| Best variant of a sweep? | the config-sweep probe + your run function | depends on the run function |
| Best vs Triton on production shapes? | the targeted-bench probe | ~0.5 s per (shape, backend) |

These give you a static, GPU-free signal in well under a second per
variant. Use rocprof to confirm, not to discover.

### 3.2 Compute-Bound Signals

- High MFMA issue density (`SQ_ACTIVE_INST_MFMA` > 70 %).
- High VALU utilization (the ISA-inspection probe shows `valu` >>
  `vmem_load`).
- Low memory stalls.
- Performance scales with MFMA count reduction.
- Performance sensitive to accumulation order / K packing.
- VGPR pressure throttles occupancy
  (the occupancy probe reports `limited_by == "VGPR"`).
- Scheduling hints change performance.
- K dimension large enough to amortize memory.

### 3.3 Memory-Bound Signals

- High global load/store throughput.
- Performance scales with vectorization and coalescing.
- Performance improves with caching, preshuffle, prepack, or LDS reuse.
- Low MFMA utilization.
- Many scalar/vector memory instructions per MFMA
  (the ISA-inspection probe shows `vmem_load` and `vmem_store`
  dominating).
- Sparse/gather/paged access dominates.
- Output epilogue is slow due to stores.

### 3.4 Synchronization-Bound Signals

- Many barriers in the disassembly. The ISA-inspection probe reports
  the `barrier` count and the `waitcnt` count plus the most common
  encoded operands.
- Barriers inside hot loops.
- Low occupancy and frequent LDS phases.
- Performance improves when fusing iterations between barriers.
- A double-buffered async software pipeline collapses barriers
  by overlapping global → LDS DMA with compute.

### 3.5 Launch-Bound Signals

- Tiny kernel latency.
- Graph replay improves throughput.
- Batched problem with many small independent kernels.
- CPU overhead visible.
- Kernel time close to launch latency (~5-15 µs on AMD).
- Fusion or persistent kernels help.
- Load the HSACO once and repeat launches, and / or chain multi-stage
  launches on one stream.
- Keep long-lived workspaces alive across launches (a workspace pool).
- Warm up and iterate to amortize the per-launch HIP module load
  before measuring.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#bottleneck-classification).

---
Then pick a lever family: [algorithmic-mapping](../10-levers/algorithmic-mapping.md), [memory-hierarchy](../10-levers/memory-hierarchy.md). Static probes how-to: [dsl-probe-workflow](../../dsl/ck-dsl/dsl-probe-workflow.md).
