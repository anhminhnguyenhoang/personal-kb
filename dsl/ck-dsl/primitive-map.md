---
name: primitive-map
description: Where each general kernel-optimization concept lives in ck_dsl (instances, helpers, builder calls, probes).
source: CK DSL Optimization Runbook (ROCm/rocm-libraries, ck_dsl)
---

# ck_dsl primitive map

Each section maps a DSL-neutral note in ../../kernel-optimization/ to the ck_dsl code that implements it. Paths are relative to the ck_dsl package and drift; verify before use.

## the-loop

- Loop step 4 "inspect IR / ISA / resources" → `utilities/tools/dsl_probes/` — static-inspection probes written for the runbook (see bottleneck-classification).
- Loop step 5 "change one lever" → §12.1 Knob Catalog — every performance lever exposed by `ck_dsl`, grouped by family.
- Extra DSL-only sections → §18 DSL Probe Workflow (when and how to use the probes) and §19 Reproducible Commands (venv / PYTHONPATH / harness invocations).
- Runbook companion docs → `runbook_mapping.md` (section-by-section DSL-primitive table), `runbook_compliance.md` (pass results per section), `measured_results.md` (last validation pass).
- gfx950 / CDNA4 reference → §21 Target Architecture Reference (see gfx950).
- Skill docs → `utilities/skills/` (`gemm-optimization`, `lds-optimization`, `kernel-trace-analysis`, `prefetch-data-load`, `capture-kernel-trace`, `empirical-case-studies`, `kernel-launch-guide`, `bisect-perf-regression`).
- Probe directory → `utilities/tools/dsl_probes/` — the static-inspection probes introduced for the runbook.

## define-the-problem

- Operation contract per family → spec dataclasses: `UniversalGemmSpec` (GEMM); `ConvProblem` + `ImplicitGemmConvSpec` (implicit-GEMM conv); `DirectConvProblem` + `DirectConv16cSpec` / `DirectConv4cSpec` (direct grouped conv); `UnifiedAttentionProblem` + `UnifiedAttention2DSpec` / `*3DSpec` / `*ReduceSpec` (paged attention); `Reduce2DSpec` / `LayerNorm2DSpec` / `RMSNorm2DSpec` / `ElementwiseSpec` / `Transpose2DSpec` (small ops); `FmhaVarlenSpec` / `FmhaHeadGroupingSpec` / `FmhaPagedPrefillSpec` / `FmhaSplitKvDecodeSpec` / `FmhaFwdFp8Spec` / `FmhaBwdSpec` / `FmhaAppendKvSpec` (FMHA) — the spec name is the contract.
- Forward vs backward → `FmhaBwdSpec` is the only backward kernel; everything else in `instances/` is forward.
- Fusion boundary → `helpers/fuse.py` (`compile_fn`, `explain_fn`, `FusedEpilogue`, `_PATTERN_TABLE`); docs in `dsl_docs/fusion/overview.md`.
- Atomics → `b.global_atomic_add_f32`, emitted by split-K GEMM and Stream-K.
- Workspace / side effects → `WorkspacePool`, `*_workspace_bytes`.
- Named dimensions → problem dataclass fields: `ConvProblem.N/Hi/Wi/C/K/R/S/sH/sW/pH/pW/dH/dW`; `UnifiedAttentionProblem.total_q/num_seqs/num_query_heads/num_kv_heads/head_size/block_size/max_seqlen_q/max_seqlen_k`.
- Compile-time vs runtime shapes → spec fields (compile-time) vs manifest fields / kernel args (runtime).
- Boundary-shape parity harness → `examples/` (e.g. `examples/ck_tile_parity.py`).
- Layouts → `UniversalGemmSpec.layout = "RCR"` / `"CRR"`; conv NHWC (`dsl_docs/instances/convolution.md`); KV paging as (block, slot).
- Coordinate-transform DAG → `transforms.py` with `pass_through`, `pad`, `pad_dynamic`, `embed`, `unmerge`, `merge`, `indirect`; walkthrough in `architecture/TRANSFORM_DAG.md`.
- Alignment / contiguity descriptor → `TensorDescriptor` in `transforms.py`; legacy `TensorView` / `TileWindow` in `helpers/tensor_view.py`.
- Dtype fields in spec → e.g. `UnifiedAttention2DTiledSpec.dtype="bf16"`, `kv_storage_dtype="fp8e4m3"`.
- Dtype catalog → `core/ir.py:34-97` (`I1`..`I64`, `BF16`, `F16`, `F32`, `FP8E4M3`, `BF8E5M2`).
- Quantization helpers → `helpers/quant.py` (`quant_max_abs`, `quantize_scalar_f32`, `dequantize_scalar_to_f32`, `quant_ir_type`, `ir_to_qdtype`; `QDType = Literal["i8","fp8e4m3","bf8e5m2"]`).
- MX block-scale → `helpers/mx_scale.py` + `MxGemmSpec`.
- Tolerance policy → `examples/ck_tile_parity.py` (per-op tolerances; layer/rms-norm history in `notes/PROPOSALS_IMPLEMENTATION_REPORT.md::F2`).
- Bug-signature tolerances → `utilities/skills/empirical-case-studies.md` (Case Study 3).
- Padding → `transforms.pad`, `transforms.pad_dynamic` (returns `(offset, valid)`).
- OOB-safe buffer descriptor → `b.buffer_rsrc(ptr, num_bytes)` (`flags=0x00027000`).
- Masks → `helpers/attention.py` (`causal_mask`, `sliding_window_mask`, `apply_softcap_log2`).
- Split-K / persistent / Stream-K → `instances/streamk_gemm.py` + `helpers/streamk.py` + `helpers/persistent.py`.
- Paged-attention indirection → `transforms.indirect(...) + unmerge(...)`.

## establish-baselines

- Correctness gate 1 (static tests) → `pytest test/test_ck_dsl.py` — 286 tests, ~1.7 s.
- Correctness gate 2 (docs / symbols / build smoke) → `python python/ck_dsl/dsl_docs/development/verify_dsl_docs.py` — 50 checks, ~2 s.
- Correctness gate 3 (per-kernel manifest verify) → `python -m ck_dsl.run_manifest <hsaco> <manifest>.json --verify`.
- Correctness gate 4 (small ops vs torch) → `python python/ck_dsl/examples/ck_tile_parity.py --op all` — 20 cases, seed=0.
- Correctness gate 5 (extended kernels) → `python python/ck_dsl/examples/parity_extended_kernels.py --op all`.
- Correctness gate 6 (attention) → `python python/ck_dsl/examples/attention/parity_unified_attention.py`.
- Correctness gate 7 (lowering audit) → `python python/ck_dsl/examples/hip_lowering_parity.py` — LLVM vs HIP-debug lowering over every shipped spec.
- NumPy fp32 reference → `run_manifest.py`; attention reference → `ref_paged_attn` in `examples/attention/parity_unified_attention.py`.
- CK Tile C++ baseline → `lower_kernel_to_cktile` (`core/lower_cktile.py`); only `UniversalGemmSpec` and `ImplicitGemmConvSpec` lower, the rest raise `NotImplementedError`.
- Triton baseline → AITER `unified_attention`, located via the `AITER_PATH` env var.
- Torch eager baseline → `examples/ck_tile_parity.py::_bench_torch`.
- FMHA correctness oracle → `_fmha_warp_body.py` (`fmha_warp_fwd_inner_body`); specs still on the warp-scalar body: paged_prefill, splitkv_decode, bwd, varlen (bf16 pre-F1), fp8 QK/PV.
- Production FMHA body → `mfma_attention_fwd_inner_body` in `helpers/mfma_attention.py`.
- HIP-event timing → `runtime/launcher.py::time_launches` (warmup, per-launch sync, iteration loop).
- Triton-safe timing → `utilities/tools/dsl_probes/probe_targeted_bench.py::time_cuda_event` (`time_launches` syncs between iterations and perturbs Triton's autotuner).
- Median + spread → `benchmark/summary.py::summarize_runs`.
- Module-load amortization → `KernelLauncher` (one HSACO) / `PipelineLauncher` (multi-stage).
- Graph-replay timing → `time_launches` `no_fence` context manager.
- Metadata recording → `helpers/manifest.py::make_*_manifest` → `write_artifact`; `manifest.notes` holds the library SHA; `compile_kernel(options=...)` (default `-O3`); `analyze_hsaco(hsaco_path).resources`; artifact `timings` dict (`ir_build`, `ir_lower_llvm`, `comgr_bc`, `reloc`, `exe`, `total`).

## bottleneck-classification

- Arithmetic-intensity metrics → `ConvFp16Problem::metrics()` (returns `tflops`, `gbps`); `.flops` on `UniversalGemmSpec` / `BatchedGemmSpec` / `GroupedGemmSpec`; the manifest runner prints `TFlops` and `GB/s` per launch.
- Kernel runner for profiling → `python -m ck_dsl.run_manifest "$HSACO" "$MANIFEST" --shape ... --warmup 5 --iters 100` under `rocprofv3`.
- Register usage per config → `utilities/tools/utils/profile_register_usage.py`.
- LDS conflict analysis → `analyze_lds_conflicts.py` (see memory-hierarchy).
- Occupancy probe → `probe_occupancy.py` — compile + readelf, ~0.5 s/variant; reports `limited_by`.
- Intrinsic-count probe → `probe_intrinsic_counts.py` — lower only, ~0.05 s/variant.
- ISA-inspection probe → `probe_isa_inspect.py` — compile + objdump, ~0.5 s/variant; buckets `valu`, `vmem_load`, `vmem_store`, `barrier`, `waitcnt`, `salu`.
- Lowering-compare probe → `probe_lowering_compare.py` — HIP-debug vs LLVM-direct, ~10 s/variant (hipcc).
- Config-sweep probe → `probe_config_sweep.py` + a user `run_fn`.
- Targeted-bench probe → `probe_targeted_bench.py` — best vs Triton on production shapes, ~0.5 s per (shape, backend).
- Probe directory → `utilities/tools/dsl_probes/`.
- Occupancy / VGPR levers (table "DSL Lever" column) → `pipeline="lean"`, smaller `tile_k`, `kernel.attrs["waves_per_eu"]`.
- Memory-latency lever → `pipeline="compv4"` + `AsyncTileLoader`, more stages.
- Bandwidth / L2 lever → enlarge `tile_m`, `tile_n`.
- Bank-conflict lever → `LdsLayout` padding or XOR swizzle.
- Barrier collapse → `pipeline="compv4"` software pipeline.
- Launch-bound levers → `KernelLauncher`, `PipelineLauncher`, `WorkspacePool`, `time_launches(..., warmup, iters)`.

## diagnostic-decision-tree

- Step 1 build smoke → `probe_config_sweep --only-build`; SPEC-FAIL comes from validation in spec `__post_init__`.
- Step 2 → `probe_occupancy`; levers `pipeline="lean"`, smaller `tile_k`.
- Step 3 → `probe_intrinsic_counts`; "no async DMA" means `pipeline != "compv4"`.
- Step 4 → `probe_targeted_bench` against Triton / CK Tile / AITER.
- Step 5 → `probe_isa_inspect`.
- Step 6 LDS stall → `analyze_lds_conflicts.py`.

## algorithmic-mapping

GEMM family:
- Plain GEMM → `instances/gemm_universal.py::UniversalGemmSpec`.
- Batched / strided batched GEMM → `instances/batched_gemm.py::BatchedGemmSpec` (layouts in the spec).
- Grouped GEMM → `instances/grouped_gemm.py::GroupedGemmSpec`.
- Persistent GEMM → `helpers/persistent.py` + `instances/streamk_gemm.py`.
- Stream-K → `instances/streamk_gemm.py::StreamKGemmSpec`.
- Split-K → `b.global_atomic_add_f32` available; spec field WIP.
- Preshuffled GEMM → `helpers/preshuffle.py::PreshuffleBSpec`.
- Block-scaled GEMM → `instances/block_scale_gemm.py::BlockScaleGemmSpec`.
- Sparse GEMM → not yet implemented.
- Quantized GEMM → block-scaled / MX GEMM + `helpers/quant.py`.
- Fused-epilogue GEMM → `instances/gemm_multi_d.py`, `gemm_multi_abd.py`, `helpers/fuse.py`.
- MFMA reference GEMM → `instances/mfma_gemm.py`.
- FlatMM (small decode) → `instances/flatmm.py`.
- Batched N-D contraction → `instances/batched_contraction.py`.
- Shape-fits-MFMA check → `helpers/atoms.py` `MfmaAtom` factory.
- Tiny shapes → `flatmm` or `mfma_gemm` instead of `gemm_universal`.
- Grouped scheduling vs padding → `GroupedGemmSpec`.
- Contiguity → `layout="RCR"` etc.

Convolution:
- Direct conv → `instances/conv_direct_grouped.py::DirectConv16cSpec`, `DirectConv4cSpec` (streaming-row 3-acc circular pipeline documented at lines 41-66).
- Implicit GEMM → `instances/conv_implicit_gemm.py::ImplicitGemmConvSpec`; auto-unrolled → `instances/conv_implicit_gemm_auto.py`.
- im2col + GEMM → `instances/img2col.py`.
- Winograd / FFT → not yet implemented.
- Depthwise / small-channel → `DirectConv4cSpec`.
- Small-channel MFMA atoms (`4x4x4`, `16x16x16`, `16x16x32`) → `helpers/atoms.py::MFMA_F16_ATOMS`.
- VGPR floor check → `probe_occupancy.py`.

Attention:
- Unified prefill + decode (scalar oracle) → `instances/attention_unified.py`.
- 2D MFMA tiled → `instances/attention_tiled_2d.py::UnifiedAttention2DTiledSpec`.
- Split-KV decode segment + reduce → `instances/attention_tiled_3d.py`.
- FMHA forward (MFMA) → `instances/fmha_mfma.py`; varlen → `fmha_varlen.py`; head grouping (GQA / MQA) → `fmha_head_grouping.py`; paged prefill → `fmha_paged_prefill.py` (warp-scalar inner); splitkv decode → `fmha_splitkv_decode.py` (warp-scalar inner); fp8 forward → `fmha_fwd_fp8.py`; backward → `fmha_bwd.py` (warp-scalar inner); append KV → `fmha_appendkv.py` (DMA).
- Sage attention → `instances/sage_attention.py`; block-sparse (Jenga / VSA) → `instances/sparse_attention.py`.
- Valid atom for head dim → `helpers/attention.py::mfma_16x16x16_for_dtype`.
- Online softmax → `helpers/attention.py::OnlineSoftmaxState` + `warp_xor_reduce_max` / `_sum`.
- 2D vs 3D choice → `helpers/attention.py::select_2d_config` / `select_3d_config` / `use_2d_kernel`; sweep overrides `_select_2d_num_warps` / `_select_2d_tile_size` via `utilities/tools/dsl_probes/probe_config_sweep.py`.

Reductions / normalization:
- Row reduce → `instances/reduce.py::Reduce2DSpec`; LayerNorm → `layernorm2d.py::LayerNorm2DSpec`; RMSNorm → `rmsnorm2d.py::RMSNorm2DSpec`; fused add + RMSNorm + rdquant → `add_rmsnorm2d_rdquant.py`; smoothquant → `smoothquant.py`, `moe_smoothquant.py`; topk softmax → `topk_softmax.py`; pooling → `pooling.py::Pooling2DSpec`.
- Vector-width sweep → `sweep_row_chunks` (`vec`).
- Block reduce → `helpers/reduction.py::block_lds_reduce` (sum / max / min / prod).
- Fused scale / bias / activation → `FusedEpilogue` system.

Transpose / permute / copy:
- 2D transpose → `instances/transpose.py::Transpose2DSpec`; batched → `batched_transpose.py`; in-register sub-tile → `transpose_bc.py`; N-D permute → `permute_nd.py::PermuteSpec` (one thread per element, rank-N decompose); im2col copy → `img2col.py`.

Gather / scatter:
- Bounds-checked buffer descriptors → `buffer_rsrc`.
- MoE → `instances/fused_moe.py`, `moe_sorting.py`, `moe_gemm_fused.py`, `fused_moe_e2e.py`.

## work-decomposition

- Per-family grid size → `*_grid` helpers, e.g. `gemm_universal.gemm_universal_grid`.
- Chiplet swizzle → `helpers/grid.py::chiplet_transform_chunked` with `NUM_XCDS_MI300X/MI325X/MI350X == 8`.
- Persistent CTAs → `helpers/persistent.py::persistent_tile_loop`.
- Block size → spec fields `tile_m`, `tile_n`, `tile_k`, `warp_m`, `warp_n`; verified at lowering by `kernel.attrs["max_workgroup_size"]`.
- Wave-count sweep → `probe_config_sweep.py` with `num_warps ∈ {1, 2, 4, 8}` overrides.
- Rows per block → `block_m_per_warp` in the 2D attention spec.
- Wave ownership view → `helpers/geometry.py::WarpGrid` (tile + warp grid + bound `tid / lane / warp_* / block_*_off` SSA).
- Lane-to-MFMA mapping → `MfmaAtom.lane_to_output(b, lane, i)`.
- Lane-contiguous loads → `helpers/loads.py::lane_contiguous_descriptor`.
- Compile-time decomposition → `IRBuilder.static_for` / `unroll` / `static_if` (Python-time only).

## memory-hierarchy

- Coalesced sync DRAM → VGPR → LDS loads → `helpers/loads.py::CoalescedTileLoader`.
- Vector loads → `b.buffer_load_vN_f16`, `b.global_load_vN_f16` (N ∈ {1, 2, 4} dwords).
- Buffer-resource descriptor → `b.buffer_rsrc(ptr, num_bytes)` (`flags=0x00027000`, OOB lanes return zero).
- Accumulator → packed output → `b.vec_trunc_f32_to_f16`.
- Direct vector stores → `helpers/epilogues.py::DirectEpilogue`.
- LDS epilogue → `helpers/epilogues.py::CShuffleEpilogue`.
- Double buffering → `UniversalGemmSpec.pipeline = "compv4"`.
- Direct global-to-LDS async copy → `helpers/loads.py::AsyncTileLoader` (wraps `raw_ptr_buffer_load_lds_addr`, per-wave LDS base).
- LDS swizzle / padding → `helpers/layouts.py::LdsLayout` with `lds_k_pad` (defaults `+8` sync, `0` async); `LdsLayout(swizzle="xor")`.
- Transposed LDS loads → `helpers/layouts.py::TransposeLdsReader`, `b.ds_read_tr16_b64`, `b.ds_read_tr_b8`.
- LDS bank-conflict analysis → `utilities/tools/stage4_analyze/analyze_lds_conflicts.py` (rocprof counters + ISA inspection).
- Count `ds_read` / `ds_write` → `probe_isa_inspect.py`; compare variants → `probe_intrinsic_counts.py`.
- Store-width check → `probe_isa_inspect.py` (`buffer_store_dwordx2` / `_x4` vs `buffer_store_short`; `vmem_store` bucket).
- Occupancy / LDS footprint → `probe_occupancy.py`, `analyze_hsaco`.
- `sched_barrier` never appears in the ISA (any arch; it is a scheduler directive) → judge by instruction order.
- Register tracking → `probe_occupancy.py`.
- Read/write conflict asymmetry → `utilities/skills/empirical-case-studies.md` (Case Study 2).

## matrix-instructions

- Atom catalog → `helpers/atoms.py` `MfmaAtom` (f16: 16x16x16, 16x16x32, 32x32x8, 32x32x16, 4x4x4; bf16: 16x16x16, 16x16x32, 32x32x16; fp8 / bf8 + scaled / MX).
- Small-channel `4x4x4` use → `DirectConv4cSpec`.
- WMMA (RDNA / wave32) → not currently exposed.
- Emitted-intrinsic check → `probe_intrinsic_counts.py`.
- Lane mapping check → `MfmaAtom.lane_to_output(b, lane, i)`.
- 32×32 chain lever → `use_mfma_32x32` + `use_transposed_qk_32x32` (unified attention 2D).
- Online softmax state → `helpers/attention.py::OnlineSoftmaxState`.

## epilogue

- Bias add → `helpers/fuse.py::BiasAdd`.
- Activation → `helpers/fuse.py`, `EpilogueOp` chain.
- Residual add → `ResidualAdd`; scaling → `Scale`.
- Quant / dequant → `helpers/quant.py`.
- Type conversion → `vec_trunc_f32_to_f16`, `cvt_pk_f32_fp8`.
- MX row / column scale → `helpers/mx_scale.py`.
- Dropout → `helpers/rng.py::dropout_mask_pair_f32`.
- Store transpose → `instances/transpose*` + LDS reader formula.
- Direct epilogue → `helpers/epilogues.py::DirectEpilogue(vec_in_acc=True/False)`.
- Packed stores → `b.buffer_store_vN_f16`, `b.global_store_vN_f16`.
- LDS / shuffle epilogue → `helpers/epilogues.py::CShuffleEpilogue.from_grid(...)`.
- Store-vector count in CK-style epilogue → `STORE_VECS = BLOCK_Q × BLOCK_C8`.
- Per-iter ISA histogram check → `probe_isa_inspect.py` or `probe_intrinsic_counts.py`.

## pipelining-scheduling

- Ping-pong staging → `helpers/pipeline.py::SoftwarePipeline.run_ping_pong(...)`.
- Async copy → `AsyncTileLoader` + `b.async_buffer_load_lds`.
- Wait placement → `b.s_waitcnt(vmcnt=0)`.
- The DSL's `s_waitcnt` constant `20336` (`0x4F70`) decodes to vmcnt 16, lgkmcnt 15 (lgkmcnt is 4 bits on gfx9).
- Wait / barrier counts → `probe_isa_inspect.py`.
- Scheduling hints → `helpers/schedule.py::SchedulePolicy` (`sched_group_barrier(mask, count, sync_id)`, `s_setprio`, `s_waitcnt`).
- Compare ISA with / without hints → `probe_isa_inspect.py`, `probe_intrinsic_counts.py` (`sched.barrier` / `sched.group.barrier` should be 0 on gfx950).
- Scalar ALU address math → `probe_isa_inspect.py` `salu` sub-bucket.

## compiler-build

- `NDEBUG` pass-through → `runtime/comgr.py`.
- Explicit target → `compile_kernel(kdef, isa="amdgcn-amd-amdhsa--gfx950")`.
- Default flags → `runtime/comgr.py` (`-O3`); per-spec overrides via `compile_kernel(kdef, options=[...])`.
- Conservative pass pipeline → `core/passes.py::optimize_kernel` (canonicalize → integer constant fold → CSE → DCE, up to 3 iterations; no vectorizer / fuser / scalar-elision).
- LLVM lowering → `core/lower_llvm.py` (only scalar-op optimization is `_lower_unrolled_for`: Phase 3 unroll + Phase 4a trailing-sync elide).
- `if`-elision guard → ck_dsl raises `TypeError` on `Value.__bool__`; use `IRBuilder.static_if(...)` for Python booleans and `IRBuilder.scf_if(...)` for runtime predicates.
- Alias attributes → `IRBuilder.param(noalias=True, readonly=True, writeonly=True, align=N, dereferenceable=N)`.

## isa-resource-inspection

- Static analysis layer → `ck_dsl.analysis`.
- `analyze_llvm_ir(text)` → `analysis/ir.py` — counts MFMA / async / raw-buffer / barriers / waitcnts in LLVM IR.
- `analyze_hsaco(hsaco_path)` → `analysis/isa.py` — `llvm-objdump -d` + `llvm-readelf --notes`; returns `IsaStats` + `ResourceInfo`.
- `parse_isa(text)` → `analysis/isa.py` — opcode tally.
- `parse_resources(text)` → `analysis/isa.py` — VGPR / SGPR / AGPR / LDS / spill.
- `VariantReport.from_artifact(...)` → `analysis/report.py` — joins compile + IR + ISA + bench.
- `compare_variant_reports([...])` → sorted comparison rows.
- Interactive probes → `utilities/tools/dsl_probes/`: `probe_intrinsic_counts.py` (analyze_llvm_ir-like, custom intrinsic table), `probe_isa_inspect.py` (analyze_hsaco-like, VALU/SALU sub-buckets), `probe_occupancy.py` (readelf notes + occupancy estimate; flags `spill > 0`).
- Worked diff → `probe_intrinsic_counts.py` on the `use_mfma_32x32` lever of `UnifiedAttention2DTiledSpec`.
- "Is it faster?" → `probe_targeted_bench.py`.

## op-specific-checklists

- Layout → `UniversalGemmSpec.layout`.
- Tile matches MFMA → `MfmaAtom.shape`.
- Vectorized loads → `probe_intrinsic_counts.py` (`global.load` / `buffer.load` vs MFMA count).
- Vectorized epilogue → `probe_isa_inspect.py` (`buffer_store_dwordx2` / `_x4` vs `buffer_store_short`).
- Warp-level vs block-level reduction → `block_lds_reduce` vs warp-only.
- Fusion VGPR cost → `probe_occupancy.py`.
- Fusion pipeline → `helpers/fuse.py`, routed through `FusionLegalizer.legalize(graph)` (LDS budget, dtype, vector width, atomic-region rules).

## failure-modes

- AST rewriter elision guard → `Value.__bool__`.
- Missing op lowering in HIP-debug → `probe_lowering_compare.py`.
- Scalar stores → `probe_isa_inspect.py` (`buffer_store_short` >> 0).
- Spills / low occupancy → `probe_occupancy.py` (`spill > 0`, `waves_per_cu < 4`).
- Bank conflicts → `analyze_lds_conflicts.py`.
- Cross-flag restriction audit → spec validators (spec `__post_init__`).
- Slot-0-only state init after atom switch → `REGS_PER_LANE` (changes with MFMA shape).
- HSACO cache aliasing → salt the kernel symbol with a shape hash before compile (§12.2).
- Launch timing vs Triton autotuner → `time_launches`; use `probe_targeted_bench.py::time_cuda_event` instead.

## reporting-template

- Resource reports for published numbers → `analyze_hsaco`, `probe_occupancy.py`, `probe_isa_inspect.py`.
- Last documented validation pass → `measured_results.md`.

## decision-heuristics

- Output dim smaller than MFMA width → `DirectConv4cSpec` + `4x4x4` atom.
- Register-intermediate global-to-LDS → `pipeline = "compv4"` (direct async copy).

## gfx950

- Full atom catalog → `helpers/atoms.py::MFMA_*_ATOMS`; factories `MfmaAtom.f16_4x4x4`, `f16_16x16x16`, `f16_16x16x32`, `f16_32x32x8`, `f16_32x32x16`, `bf16_16x16x16`, `bf16_16x16x32`, `bf16_32x32x16`, `fp8_*_f8`, `bf8_*_bf8`.
- `4x4x4` use → `DirectConv4cSpec`.
- Scaled (block-scale) MFMA → `BlockScaleGemmSpec`; `helpers/atoms.py::MFMA_FP8_ATOMS`.
- MX MFMA → `MxGemmSpec`; `helpers/mx_scale.py`.
- i4 unpack → `helpers/codebook.py` (`codebook_lookup_i4_pair_to_{bf8,fp8}`).
- 32×32 chain levers → `use_mfma_32x32`, `use_transposed_qk_32x32` (attention 2D path).
- `v_permlane32_swap_b32` → `permlane32.swap` intrinsic.
- `ds_bpermute_b32` → `b.ds_bpermute`; `ds_bpermute_b64` → `b.ds_bpermute_b64`.
- `ds_swizzle_b32` → `b.ds_swizzle_xor`.
- `ds_read_tr16_b64` / `ds_read_tr16_b128` / `ds_read_tr_b8` → `b.ds_read_tr16_b64` / `b.ds_read_tr16_b128` / `b.ds_read_tr_b8`.
- `s_setprio` → `b.s_setprio`.
- `s_sched_barrier` / `s_sched_group_barrier` → `b.sched_barrier`, `b.sched_group_barrier`; verify with `probe_isa_inspect.py` (`sched_barrier` sub-bucket is 0).
- Wave64 only → the only path the DSL helpers support today.
- Occupancy caps → `probe_occupancy.py` (limiters `VGPR`, `AGPR`, `LDS`, `WAVES_PER_EU_HINT`, `MAX_WAVES_PER_CU`).
- XCD constants → `helpers/grid.py::NUM_XCDS_MI300X / MI325X / MI350X` (8 / 8 / 8); swizzle `chiplet_transform_chunked`; defaults `chiplet_wgm=8`, `chiplet_num_xcds=8`, `chiplet_chunk_size=64`.
- Buffer resource → `b.buffer_rsrc(ptr, num_bytes)` (emits correct DW3).
- LDS-load dword choice → `AsyncTileLoader.choose_dwords` (accepts `{4, 3, 1}`, not 2).
- `s_waitcnt` `20336` = vmcnt 16, lgkmcnt 15 → cross-check in the lowered IR.
- fp8 / quant dtypes → `Type("fp8e4m3")`, `QDType="fp8e4m3"`; `Type("bf8e5m2")`, `QDType="bf8e5m2"`; `QDType="i8"`.
- MX scale → `MxGemmSpec`, `helpers/mx_scale.py`; block-scale mantissa → `BlockScaleGemmSpec`, `MantissaDType`, `QuantMode`.
- Native fp8 attention MFMA → `use_fp8_mfma_qk`, `use_fp8_mfma_pv` on `UnifiedAttention2DTiledSpec`.
- Default ISA target → `compile_kernel(...)` default `amdgcn-amd-amdhsa--gfx950`; `_DATALAYOUT` in `core/lower_llvm.py`.
- Other targets → `isa="amdgcn-amd-amdhsa--gfx942"` to `compile_kernel`; atoms must be in `_F16_WARP_TILE_SHAPES_*` / `_BF16_WARP_TILE_SHAPES_*`.
- Case studies → `utilities/skills/empirical-case-studies.md`.
