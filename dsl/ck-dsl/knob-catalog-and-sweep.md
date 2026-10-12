---
name: knob-catalog-and-sweep
description: "Runbook S12: the MASTER knob catalog (S12.1.A-Q — every ck_dsl perf lever grouped by family: variant, tile geometry, MFMA atom, pipeline, epilogue, LDS, occupancy, preshuffle, persistent/stream-K, quant, attention-2D micro-levers, chiplet, flags, runtime, dispatcher, hygiene, probes), sweep discipline, the dispatcher as a lever, and variant naming."
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 12. Autotuning Strategy

### 12.1 Knob Catalog (Master List)

This is the master enumeration of every performance lever exposed by
`ck_dsl` and the surrounding workflow. Knobs are grouped by lever
family. For each knob: where it lives, what it controls, what direction
it usually moves perf, and when **not** to flip it.

To discover all knobs for a specific kernel: open the spec dataclass
under `instances/<kernel>.py` and read the `@dataclass` field list.
Every field is a knob (the validator in `__post_init__` documents the
constraints).

#### 12.1.A Algorithmic variant (choose the kernel before the knobs)

| Lever | Where | Direction |
|---|---|---|
| GEMM family member | `instances/gemm_universal.py` / `batched_gemm.py` / `grouped_gemm.py` / `streamk_gemm.py` / `gemm_multi_d.py` / `gemm_multi_abd.py` / `mfma_gemm.py` / `flatmm.py` / `block_scale_gemm.py` / `mx_gemm.py` | small / decode shapes → `flatmm`; many small problems → `grouped_gemm` or `persistent`; tail-balance → `streamk_gemm`; fused chain → `gemm_multi_d` / `gemm_multi_abd` |
| Conv family | `conv_implicit_gemm.py` / `conv_implicit_gemm_auto.py` / `conv_direct_grouped.py` (16c, 4c) / `img2col.py` | tiny K or C*R*S → direct conv; 3×3 hero shapes → implicit GEMM; explicit im2col → if downstream stage is plain GEMM |
| Attention family | `attention_unified.py` (scalar oracle) / `attention_tiled_2d.py` / `attention_tiled_3d.py` (split-KV) | prefill → 2D; long-context decode → 3D; sliding-window — see §17.4 final policy |
| FMHA family | `fmha_mfma.py` / `fmha_varlen.py` / `fmha_head_grouping.py` / `fmha_paged_prefill.py` / `fmha_splitkv_decode.py` / `fmha_fwd_fp8.py` / `fmha_bwd.py` / `fmha_appendkv.py` / `sage_attention.py` / `sparse_attention.py` | choose based on KV layout (paged vs varlen), GQA, dtype, sparse pattern |

#### 12.1.B Tile and block geometry

| Knob | Spec | Default | What it does | When to change |
|---|---|---|---|---|
| `tile_m`, `tile_n`, `tile_k` | GEMM / Conv | 64 / 64 / 64-128 | Per-CTA macro tile in M / N / K | Larger ⇒ more reuse and bigger MFMA hot loop; bounded by LDS budget and VGPR pressure (run `probe_occupancy.py`) |
| `warp_m`, `warp_n`, `warp_k` | GEMM / Conv | 2 / 2 / 1 | Warp grid inside the CTA | `warp_m × warp_n × warp_k × wave_size` = block_size |
| `warp_tile_m`, `warp_tile_n`, `warp_tile_k` | GEMM / Conv | 32 / 32 / 16 | The MFMA atom shape (see §12.1.C) | One of the validated `_F16_WARP_TILE_SHAPES_GFX950` / `_BF16_WARP_TILE_SHAPES_GFX950` sets |
| `block_size` | GEMM / Attention | derived | Threads per CTA (must equal `num_warps × 64`) | Override only for autotune experiments |
| `num_warps` | Attention | 1 | wave64 warps per CTA — `BLOCK_M = num_warps × block_m_per_warp` | 1 (decode) / 2-4 (prefill) / 8 (Triton-style large prefill); ∈ {1,2,4,8} |
| `tile_size` | Attention | `block_size` | T = KV tokens per K-loop iter; multi-block decomposition | Larger ⇒ fewer outer iters, more LDS; requires `T × head_size ≥ num_warps × 64 × 8` and `T % block_size == 0` |
| `block_m_per_warp` | Attention | 16 | M rows per warp (16 = one MFMA atom; 32 = two stacked atoms or one 32×32) | 32 only when MFMA is 32×32 |
| `block_q` | DirectConv | 16 / 4 | Q output rows per CTA | bigger ⇒ more LDS for input rows |
| `block_groups` | DirectConv | 8 / 16 | Conv groups per CTA | Sweep ±1: 16c best was BLOCK_GROUPS=4, not 8 (§17.1) |
| `groups` | conv_implicit_gemm | 1 | Grouped conv via descriptor `unmerge("group")` | `> 1` for ResNeXt / depthwise-style |
| `k0_k1_split` | conv_implicit_gemm | False | K0/K1 inner-dim split for `CoalescedTileLoader` | When C is the natural contiguous dim |
| `n_acc_slots` (derived) | DirectConv | KH | Circular accumulators for row streaming | Conv 3×3 → 3 slots |

#### 12.1.C MFMA atom and K-pack

| Knob | Spec | Effect |
|---|---|---|
| MFMA atom shape (via `warp_tile_*`) | GEMM / Conv | `16x16x16`, `16x16x32`, `32x32x8`, `32x32x16`, `4x4x4`. Larger K-pack ⇒ half the K-loop trips. (§7.1, §7.4) |
| `use_mfma_32x32` | Attention 2D | switch QK / PV atom from 16×16×32 to 32×32×16. **Headline structural win for attention** — see §17.4. Requires `block_m_per_warp=32` and `tile_size % 32 == 0` |
| `use_transposed_qk_32x32` | Attention 2D | orient softmax with one query column per lane; eliminates a cross-lane reduction. Requires `use_mfma_32x32` |
| `kpack` (bf16/f16 atoms) | `MfmaAtom` factory | True ⇒ pick the larger-K variant of a given (M, N) atom (`16x16x32` over `16x16x16`); halves K-loop count |
| `MfmaAtom.f16_*` / `bf16_*` / `fp8_*` / `bf8_*` | `helpers/atoms.py::MFMA_*_ATOMS` | The catalog every kernel picks from |

Critical caveat: lane-layout matching between chained atoms matters as
much as atom shape (§7.2). 16×16×32's C-output doesn't match its A-input
→ QK→PV chain pays 288 cross-lane permutes per iter; 32×32×16 natively
matches → zero re-pack. See §17.4 for the quantitative reduction.

#### 12.1.D Pipeline / scheduling

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `pipeline` | GEMM `TraitSpec` | `"mem"` / `"compv3"` / `"compv4"` | `mem` = single-buffer; `compv4` = double-buffered async DMA + MFMA overlap. Compv3 / compv4 trade LDS for latency hiding |
| `scheduler` | GEMM `TraitSpec` | `"intrawave"` / `"interwave"` | Where the scheduler injects waits. Intrawave keeps producers and consumers in one wave; interwave splits them |
| `async_dma` | conv_implicit_gemm | False / True | Enable direct global → LDS DMA (`raw_ptr_buffer_load_lds`). Pairs with `pipeline="compv4"` |
| `unroll_k` | conv_implicit_gemm | False | Python-time unroll of the K loop. Bigger code, fewer waits |
| `use_early_v_schedule` | Attention 2D | False | Issue current-V async copy before QK so V overlaps QK + softmax (§8.1, §17.4). Use only on no-SW prefill |
| `prefetch distance` (implicit) | `pipeline` choice | — | More stages ⇒ better latency hiding but more LDS |
| `helpers/pipeline.py::SoftwarePipeline.run_ping_pong` | helper | — | Manual prologue / steady / epilogue staging when you author your own kernel |
| `helpers/schedule.py::SchedulePolicy` | helper | — | Emit `sched_group_barrier(mask, count, sync_id)` + `s_setprio` hints |

Caveat: explicit `s_sched_barrier` / `s_sched_group_barrier`
intrinsics are silently dropped by the LLVM backend on gfx950
(§3.1a / §8.4). Verify with `probe_isa_inspect.py` (`sched_barrier`
sub-bucket should be 0).

#### 12.1.E Epilogue

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `epilogue` | GEMM `TraitSpec` | `"default"` / `"cshuffle"` | `cshuffle` = LDS-staged + wide `buffer_store_dwordx{2,4}`; `default` = direct per-lane stores |
| `epilogue` | conv_implicit_gemm | `"default"` / `"cshuffle"` | Same trade-off |
| `vec_in_acc` | `helpers/epilogues.DirectEpilogue` | — | True when the accumulator's per-lane elements are already contiguous (4×4 direct conv) |
| `CShuffleEpilogue.from_grid(...)` | `helpers/epilogues.py` | — | Builds the cshuffle distribution from a `WarpGrid` |
| `FusedEpilogue` chain | `helpers/fuse.py` | — | `BiasAdd`, `ReLU`, `SiLU`, `GELU`, `Cast`, `Clamp`, `ResidualAdd`, `ResidualMul`, `Scale`, `tanh`, `sigmoid` |

Caveat: the "in-between" naive LDS epilogue (LDS layout that doesn't
match either CK's wide-store distribution or a direct contiguous
pattern) loses to both extremes. Either match the library mapping or
stay direct (§9.3, §17.4 register-PV regression analogue).

#### 12.1.F LDS layout

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `lds_k_pad` | conv_implicit_gemm | None | K-pad to break bank conflicts (`+8` sync default; `0` async default) |
| `lds_layout` | conv_implicit_gemm | None | Explicit `LdsLayout` (helpers/layouts.py) — padding, packed-async, transpose-reader |
| `LdsLayout` swizzle | `helpers/layouts.py` | — | XOR swizzle (zero LDS waste, higher ALU cost) vs padding swizzle (small LDS waste, lower ALU). Architecture-specific rule §6.4a |
| `TransposeLdsReader` | `helpers/layouts.py` | — | Use `ds_read_tr16_b{64,128}` for transposed BF16/F16 loads |
| `pad_m` / `pad_n` / `pad_k` | GEMM `TraitSpec` | False | Pad operands to tile boundaries (avoids tail scalar path) |
| `Q_lds`, `K_lds`, `V_lds`, `P_lds` sizing | Attention 2D (implicit via shape) | — | The §17.4 case showed 16 KiB allocated for `P_lds` was the structural cost; `use_register_pv` removes it |

#### 12.1.G Register / occupancy

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `waves_per_eu` | GEMM, conv, Attention 2D | None | `"amdgpu-waves-per-eu"` hint. 2-3 forces VGPR budget down, more waves/CU. None ⇒ LLVM heuristic |
| `wave_size` | GEMM, DirectConv | 64 | wave64 is the only path the helpers support today |
| `kernel.attrs["max_workgroup_size"]` | DSL IR | derived | `"amdgpu-flat-work-group-size"` emitted from block_size |
| `use_agpr_alloc_zero` | Attention 2D | False | Force VGPR-only MFMA — avoids AGPR↔VGPR copies on accumulator-touching paths. Already a no-op when the existing R4 path has zero AGPR moves (§17.4) |
| `IRBuilder.param(noalias, readonly, writeonly, align, dereferenceable)` | DSL IR | — | Alias + alignment hints for the LLVM backend |

#### 12.1.H Operand layout & preshuffle

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `dtype_a`, `dtype_b`, `dtype_c`, `dtype_acc` | GEMM `DataSpec` | f16/f16/f16/f32 | Per-operand precision |
| `layout` | GEMM `DataSpec` | `"RCR"` | A row / B col / C row major (only "RCR" today) |
| `preshuffle_b` | GEMM `TraitSpec` | False | B operand pre-shuffled by host via `host_preshuffle_layout`; per-lane B-load uses one `buffer_load_dwordx4` per K-tile instead of strided scalar loads |
| `batched` | GEMM | False | Reads `block_id_z` as batch index, picks up `stride_{a,b,c}` args |

#### 12.1.I Persistent / Stream-K / Split-K

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `persistent` | GEMM `TraitSpec`, `StreamKGemmSpec` | False | Persistent CTA loops over many macro tiles via `persistent_tile_for_each` (workspace counter) |
| `num_cus` | `StreamKGemmSpec` | 304 | Target CU count for Stream-K partition |
| `blocks_per_cu` | `StreamKGemmSpec` | 1 | Persistent dispatch density |
| `reduction` | `StreamKGemmSpec` | `Atomic` | `StreamKReductionStrategy.{Atomic, Reduction, AtomicWithFixup}` |
| split-K | GEMM (not yet a spec field; available primitives) | — | `b.global_atomic_add_f32` for atomic split-K; `helpers/streamk.py` for the Stream-K macro tile decode |
| Grouped GEMM persistent | `instances/grouped_gemm.py` | per-group launches | v2 persistent variant is a documented follow-up |

#### 12.1.J Quantization

| Knob | Spec | Values | Effect |
|---|---|---|---|
| `kv_storage_dtype` | Attention 2D | None / `"fp8e4m3"` | FP8 K/V cache — halves KV HBM bytes; dequant happens on load (sync) or in-register (`use_fp8_mfma_qk`) |
| `use_fp8_mfma_qk` | Attention 2D | False | Native fp8 MFMA for QK (K stays raw fp8 in LDS; in-register dequant). bf16 math preserved |
| `use_fp8_mfma_pv` | Attention 2D | False | Native fp8 MFMA for PV (P quantized to fp8 before PV) |
| `mantissa_dtype` | `BlockScaleGemmSpec` | — | FP8 / BF8 element + per-block scale |
| `MxMantissaDType` | `MxGemmSpec` | — | E8M0 shared exponent; MX (microscaling) |
| `QDType` | `helpers/quant.py` | `"i8"` / `"fp8e4m3"` / `"bf8e5m2"` | Quant dtype for smoothquant, MoE smoothquant, add-rmsnorm-rdquant |
| `helpers/codebook.py` | helper | — | i4 packed weight unpack (`codebook_lookup_i4_pair_to_{bf8,fp8}`) |
| `helpers/i4_dequant.py` | helper | — | i4 dequant primitives |

Caveat: FP8 KV is a HBM-bandwidth lever, not a compute lever. If the
PMC says `MemUnitStalled < 1 %` the kernel isn't HBM-bound and FP8 KV
will be neutral or negative (§17.4 worked example).

#### 12.1.K Attention-2D micro-levers (UA 2D specific)

Every flag below is exposed on `UnifiedAttention2DTiledSpec`. They are
all validated in `__post_init__` with explicit constraints —
read the validator before combining flags. Cohort impact is from §17.4.

| Flag | Effect | Best paired with |
|---|---|---|
| `use_register_pv` | Keep P in registers; remove `P_lds` (16 KiB / WG) and the 64 `ds_write_b16` per iter | Only wins when paired with `use_mfma_32x32 + use_transposed_pv_tr_read`; **net regression alone with 16×16 atom** |
| `use_transposed_qk_32x32` | Orient softmax with one query column per lane | `use_mfma_32x32` |
| `use_transposed_scalar_state` | Single m / l per query lane (not per acc reg) | `use_transposed_qk_32x32` |
| `use_transposed_invariant_hoist` | Hoist row invariants out of per-reg/per-tile score loop | `use_transposed_qk_32x32` |
| `use_transposed_mask_once` | Compute mask invariants once per KV iter | `use_transposed_qk_32x32` |
| `use_transposed_half_local_pv` | Each 32-lane half consumes K rows it owns; matching half-local V `ds_read_tr16_b64` | `use_transposed_qk_32x32`. Strongest individual lever after R4 |
| `use_mfma32_skip_legacy_qreg` | Skip dead 16x16 Q register gather and drain barrier | `use_mfma_32x32` |
| `use_transposed_mask_limit` | Collapse causal + prefix masks into one compare | Full R4_s1mask stack |
| `use_grouped_kv2_softmax` | 2 KV tiles per acc update | Smoke OK, full cohort regressed (§17.4 "did not help") |
| `use_fast_paged_kv_desc` | Specialized paged-KV byte descriptor for the hot R4 shape (bf16, h64kv8, HD=64, BS=32, T=64, nw=4) | Use only on supported shape class |
| `use_early_v_schedule` | Issue V async before QK; V overlaps QK + softmax | No-SW prefill only |
| `use_agpr_alloc_zero` | Force VGPR-form MFMA | Currently redundant on R4 (already 0 AGPR moves) |

#### 12.1.L Multi-XCD / chiplet grid swizzle

| Knob | Spec | Default | Effect |
|---|---|---|---|
| `chiplet_swizzle` | GEMM / conv | False | Remap WGIDs so contiguous stripes land on the same XCD (L2 reuse) |
| `chiplet_wgm` | GEMM / conv | 8 | Super-tile WGM grouping |
| `chiplet_num_xcds` | GEMM / conv | 8 | MI300X / MI325X / MI350X have 8 XCDs |
| `chiplet_chunk_size` | GEMM / conv | 64 | XCD round-robin chunk size |
| `helpers/grid.py::chiplet_transform_chunked` | helper | — | Pure helper if you author your own kernel |
| Constants `NUM_XCDS_MI300X / MI325X / MI350X` | `helpers/grid.py` | — | All 8 today |

#### 12.1.M Compiler flags

Default flag list from `runtime/comgr.py` is `["-O3"]`. Per-spec
overrides via `compile_kernel(kdef, options=[...])` or
`build_hsaco_from_llvm_ir(..., options=[...])`.

| Flag | Safe? | Effect |
|---|---|---|
| `-O3` | ✅ default | LLVM optimization level |
| `-DNDEBUG` | ✅ | Disable C++ assertions on device |
| `-fno-offload-uniform-block` | ✅ | Required for some launch / perf assumptions |
| `-mllvm -amdgpu-function-calls=false` | ✅ | Force inline |
| `-mllvm -amdgpu-early-inline-all=true` | ✅ | Early inlining |
| `-mllvm --lsr-drop-solution=1` | ✅ | LSR pass tweak |
| `-mllvm -enable-post-misched=0` | ⚠️ risky | Safe in CK; **miscompiled MLIR-generated kernels** in our experience |
| `--offload-arch=gfx950 / gfx942 / gfx90a` | ✅ | Target ISA |
| `compile_kernel(kdef, isa="amdgcn-amd-amdhsa--gfx950")` | ✅ | DSL-level ISA override |

Compiler flags very rarely close large gaps (§10.2). A full flag stack
moved measured throughput by under 1 % on direct-conv kernels.

#### 12.1.N Runtime / launch

| Knob | Where | Effect |
|---|---|---|
| `KernelLauncher` | `runtime/launcher.py` | One-HSACO, repeated launches — amortizes HIP module load |
| `PipelineLauncher` | `runtime/launcher.py` | Multi-stage chained launches on one stream |
| `WorkspacePool` | `runtime/launcher.py` | Keep long-lived torch workspaces alive across launches |
| `no_fence` context manager | `runtime/launcher.py` | Skip per-call sync inside an event-timed loop (graph-style) |
| `time_launches(fn, warmup, iters, stream)` | `runtime/launcher.py` | The canonical HIP-event timer |
| `StreamConfig` | `runtime/launcher.py` | Mirror of CK Tile `stream_config` |
| `resolve_stream(stream=0)` | `runtime/torch_module.py` | Substitute torch's current stream to keep allocator coherent |
| `pack_args` vs `pack_args_kernelparams` | `runtime/torch_module.py` | AMDGPU kernarg buffer vs the safer `kernelParams` path |
| HIP graph capture | torch | Amortizes launch overhead — pair with `no_fence` and many iters |
| `rocm-smi --setperflevel high && --setsclk 7` | shell | Lock clocks to avoid thermal / DVFS noise during measurement |

#### 12.1.O Dispatcher / selector policy (per-shape branching)

A single kernel variant rarely wins every shape (§12.3, §17.4). The
selectors in `attention_unified.py` are the canonical dispatch surface;
override them via monkey-patch for sweeps, then promote a stable
policy back into the source.

| Selector | What it picks |
|---|---|
| `_select_2d_num_warps(problem)` | 1 (decode) / 2-4 (prefill) / 8 (Triton-style large prefill) |
| `_select_2d_tile_size(problem)` | T = block_size (decode), 64 (prefill), 32 (sliding-window) |
| `_select_2d_block_m_per_warp(problem)` | 16 / 32 |
| `_select_2d_waves_per_eu(problem)` | None / 2 / 3 |
| `_enable_mfma_32x32(problem)` | True for bf16 long-prefill no-SW |
| `_enable_transposed_qk_32x32(problem)` | gated on dtype, head_size, seq lens, sinks, softcap, sliding_window |
| `_enable_register_pv(problem)` | currently hard-disabled; lift carefully |
| `_enable_fp8_mfma_qk(problem)` | requires `kv_storage_dtype = "fp8e4m3"` |
| `use_2d_kernel(problem)` | 2D vs 3D split-KV — depends on `num_2d_prgms` vs target |
| `select_2d_config / select_3d_config` | top-level Attention dispatch |

Best UA 2D policy (§17.4): branch on `sliding_window` to route
no-sliding-window shapes to one kernel variant (early-V schedule,
default tile size) and sliding-window shapes to a different variant
(smaller tile size matched to the useful work per tile). Two kernel
variants behind one dispatcher entry point, materially better than
either single variant.

#### 12.1.P Benchmark hygiene

| Knob | Where | Effect |
|---|---|---|
| `--attempts ≥ 5` (fresh process) | `sweep_bench.py` | Catch bimodal latency and first-run JIT effects |
| `--warmup ≥ 5`, `--iters ≥ 20` | `time_launches`, harnesses | Discard JIT / module-load cost before measuring |
| Discard first run | benchmark policy | Cold-cache vs warm-cache effect can be > 2× (§2.3) |
| Median + spread (`(max-min)/median × 100 %`) | `benchmark/summary.py::summarize_runs` | Standard quoted statistics |
| Salt kernel symbol with shape hash | per-spec | Avoid HSACO module-cache aliasing across specializations (§12.2) |
| Full cohort vs smoke set | sweep harness | Smoke runs miss shape-dependent variance — see §12.2 caveat |

#### 12.1.Q Static probes (cheap pre-bench filter)

These are not perf knobs but per-iter signal generators that determine
which knob to flip next. See §18 for the full workflow.

| Probe | Output | What it tells you |
|---|---|---|
| `probe_occupancy.py` | VGPR / AGPR / SGPR / LDS / waves-per-CU / limiter | Is the variant occupancy-bound? Which resource? |
| `probe_intrinsic_counts.py` | LLVM AMDGCN intrinsic histogram | Did the atom switch actually emit the new intrinsic? Is async DMA active? |
| `probe_isa_inspect.py` | Post-codegen opcode histogram with VALU/SALU sub-buckets | Are stores vectorized? Are waitcnts excessive? |
| `probe_lowering_compare.py` | LLVM-direct vs HIP-debug HSACO compare | Does the HIP-debug backend agree on this kernel? |
| `probe_config_sweep.py` | Build status + HSACO size across overrides | Variant build matrix |
| `probe_targeted_bench.py` | Per-shape latency vs baseline (Triton, etc.) | Final latency table per shape |
| `probe_rocprof_single.py` | One-process kernel runner for rocprof | Clean rocprof attach window |

### 12.2 Sweep Discipline

- Sweep one family at a time.
- Cache compiled variants (`sweep.py` keys by spec hash).
- Record compile failures (`BuildRecord.error`).
- Record correctness failures.
- Record resource usage (`probe_occupancy.py`).
- Use median and variance (`benchmark/summary.py::summarize_runs`).
- Keep best per shape, not only the global best
  (`sweep.py::pick_best`).
- Avoid overfitting to one shape.
- **Salt cached kernel symbols with specialization constants.** When
  two kernel specs differ only in compile-time constants (e.g.
  `num_seqs`, binary-search trip count), the runtime HSACO/module
  cache may alias them under the same display symbol — and serve
  the wrong cached blob. Append a short shape hash to the kernel
  name before compile. **§17.4** documents the failure mode and the
  fix.
- **Always run the full cohort, not a smoke set.** Several levers in
  the §17.4 pass (grouped-KV2 online softmax, register-P proxy)
  looked promising on small smoke runs and regressed materially on
  the full production-shape sweep. The smoke set under-samples the
  shape-dependent variance.

### 12.3 The Dispatcher Is A Perf Lever

A single kernel variant rarely wins every shape. Per-shape branches
in the selector close the long-tail without compromising the hero
configuration. **§17.4**'s final policy branches on `sliding_window`
to route no-sliding-window shapes to one variant (early-V schedule,
default tile size) and sliding-window shapes to a different variant
(smaller tile size matched to the useful work per tile). Two kernel
variants behind one entry point, materially better than the best
single-variant pick. When designing a selector, budget explicit cases
for: long-prefill no-SW, short-prefill, decode, sliding-window,
single-batch, plus the FP8 KV variant if applicable.

The DSL tuning stack:

- `gemm_universal.all_dispatcher_configs(...)` — pre-baked GEMM
  config catalog.
- `sweep.py::build_all_instances(specs)` — parallel build with
  content-hash caching, emits a JSON run-plan.
- `sweep_bench.py` — bench `>=3` fresh-process attempts per
  (kernel, shape), reports median + spread.
- `helpers/autotune.py::Autotuner`, `AutotuneCache`,
  `AutotuneConfig`, `AutotuneResult`.
- `probe_config_sweep.py` — interactive in-process sweep over a
  spec dataclass.

### 12.4 Variant Naming

Use names that encode the hypothesis:

- `v1_scalar` — scalar baseline.
- `v2_mfma` — added MFMA atom.
- `v3_h_pipeline` — added H-row pipeline.
- `v4_async_lds` — switched to async DRAM→LDS.
- `v5_vector_store` — wide direct epilogue.
- `v6_k32` — K=32 MFMA fold.
- `v7_wg4_nf4` — block_groups=4, n_fold=4.
- `v8_swizzle_xor` — XOR LDS swizzle.
- `v9_lds_epilogue` — CShuffle epilogue.

The DSL spec dataclasses produce one canonical name per spec via
`spec.kernel_name()` — use it as the canonical variant ID in your
sweep CSVs.

---
The lever sections explained: [algorithmic-mapping](../../kernel-optimization/levers/algorithmic-mapping.md) through [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md). Op checklists: [op-specific-checklists](../../kernel-optimization/autotuning/op-specific-checklists.md).
