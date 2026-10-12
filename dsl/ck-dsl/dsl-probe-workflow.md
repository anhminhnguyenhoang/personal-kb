---
name: dsl-probe-workflow
description: "Runbook S18: the static-inspection probe sequence (build smoke -> occupancy -> intrinsic -> ISA -> lowering parity -> targeted bench -> rocprof confirm), its per-step cost, programmatic use, and output-field meanings."
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 18. DSL Probe Workflow

This section is the runbook's how-to for the static-inspection probes
under `utilities/tools/dsl_probes/`. Skip rocprof, get a signal in
under one second, narrow the variant list, then pay the rocprof cost.

### 18.1 Standard Diagnostic Sequence

For a new kernel or a regression:

1. **Build smoke**. Build every variant with `probe_config_sweep.py`
   `--only-build` and look for `SPEC-FAIL` or `BUILD-FAIL`. A common
   trap is that two spec fields are coupled (e.g. `use_mfma_32x32`
   requires `block_m_per_warp=32`); the spec dataclass raises
   `ValueError` and the probe reports it as `SPEC-FAIL` without
   killing the sweep.
2. **Resource check**. Run `probe_occupancy.py` over every variant.
   Look for:
   - `spill > 0` (always fix first; it invalidates perf comparisons).
   - `limited_by` distribution. If half the variants are `VGPR` and
     half are `LDS`, the optimal tile sits at the boundary.
   - waves/CU dropping below 4 — usually means you need `pipeline =
     "lean"` or a smaller tile.
3. **Intrinsic check**. Run `probe_intrinsic_counts.py` over the two
   variants you most expect to differ. Confirm:
   - The MFMA atom changed when you flipped `use_mfma_32x32`.
   - `raw.ptr.buffer.load.lds` is non-zero when you set
     `pipeline="compv4"` (async DMA active).
   - `ds.read.tr16.b64` or `ds.read.tr16.b128` appears when you used
     a transpose-LDS path (`b.ds_read_tr16_*`).
   - `s.barrier` count went down after a pipeline change.
4. **ISA check (only if intrinsic check is inconclusive)**. Run
   `probe_isa_inspect.py` to see the post-codegen instruction mix.
   This is where you confirm that vector stores actually emit
   `buffer_store_dwordx{2,4}` (and not `buffer_store_short`), that
   the cshuffle epilogue is using wide stores, and that
   `s_sched_barrier` was silently removed on gfx950 (it always is —
   see Section 8.4).
5. **Lowering parity check (only if HIP debug parity is failing)**.
   Run `probe_lowering_compare.py` — if the HSACO sizes diverge >2×
   or the VGPR/LDS deltas are large, one backend is missing an op
   lowering.
6. **Best-of-sweep on production shapes**. Once the variant list is
   narrowed by static signal, run `probe_targeted_bench.py` on the
   production-trace shape set with `candidate_fn` (your best variant)
   and `baseline_fn` (Triton / AITER / CK Tile).
7. **rocprof confirm**. Pick the top 1-2 variants from step 6, run
   `rocprofv3 -i metrics.txt -- python probe_rocprof_single.py …`,
   then parse with `analyze_lds_conflicts.py`, `compare_rocprof_stats.py`.

This sequence costs roughly:

- step 1: 100 ms × N variants → typically <1 s
- step 2: 500 ms × N → typically 5-10 s
- step 3: 50 ms × N
- step 4: 500 ms × N
- step 5: 10 s × N (hipcc dominates)
- step 6: 1-5 s × N × M shapes
- step 7: 10-60 s per kernel (rocprof)

So a full sweep over 10 variants × 8 shapes is around 5 minutes
including rocprof, versus hours if you go straight to rocprof
without filtering.

### 18.2 Programmatic Use

Every probe has a Python entry point that doesn't require argv:

```python
from probe_occupancy import probe_occupancy, ARCH_GFX950
from probe_intrinsic_counts import probe_intrinsic_counts, count_intrinsics
from probe_isa_inspect import probe_isa_inspect
from probe_lowering_compare import probe_lowering_compare
from probe_config_sweep import probe_config_sweep
from probe_targeted_bench import bench_shapes, time_cuda_event

# Example: feed a custom kernel + spec to probe_occupancy
from ck_dsl.instances.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled,
)
spec = UnifiedAttention2DTiledSpec(
    head_size=64, block_size=32, num_query_heads=64, num_kv_heads=8,
    dtype="bf16", use_sinks=True, sliding_window=0, has_softcap=False,
    num_warps=4, tile_size=64,
)
kdef = build_unified_attention_2d_tiled(spec)
probe_occupancy([("my_variant", kdef, spec.num_warps)], arch=ARCH_GFX950)
```

### 18.3 Probe Outputs and What They Mean

| Output field | From | Meaning |
|---|---|---|
| `vgpr_count` | `probe_occupancy.py` | private VGPRs per lane, allocated in 16-VGPR slots |
| `agpr_count` | `probe_occupancy.py` | MFMA accumulator VGPRs (gfx9x0 only) |
| `sgpr_count` | `probe_occupancy.py` | scalar regs per wave |
| `vgpr_spill_count` | `probe_occupancy.py` | nonzero = spills, usually a perf bug |
| `lds_size` | `probe_occupancy.py` | static LDS bytes per workgroup |
| `waves_per_cu` | `probe_occupancy.py` | coarse occupancy estimate |
| `limited_by` | `probe_occupancy.py` | `VGPR`, `AGPR`, `LDS`, `WAVES_PER_EU_HINT`, `MAX_WAVES_PER_CU` |
| `mfma` (cat) | `probe_isa_inspect.py` | count of `v_mfma_*` instructions |
| `vmem_load` / `vmem_store` | `probe_isa_inspect.py` | `buffer_*` + `global_*` count |
| `waitcnt` patterns | `probe_isa_inspect.py` | top 6 most-frequent encoded operand strings |
| `intrinsics.mfma.f32.*` | `probe_intrinsic_counts.py` | per-atom intrinsic count in lowered IR |
| `intrinsics.raw.ptr.buffer.load.lds` | `probe_intrinsic_counts.py` | async DRAM→LDS count |
| `intrinsics.ds.bpermute` / `ds.swizzle` | `probe_intrinsic_counts.py` | cross-lane reduction primitive |
| `structural.fmul` / `fadd` | `probe_intrinsic_counts.py` | post-lowering scalar arithmetic count |

---
Cheap-filter rationale: [bottleneck-classification](../../kernel-optimization/00-method/bottleneck-classification.md) (S3.1b). Probe entries in the knob catalog: [knob-catalog-and-sweep](knob-catalog-and-sweep.md) (S12.1.Q). Commands: [reproducible-commands](reproducible-commands.md).
