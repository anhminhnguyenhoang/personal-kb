---
name: compilation-pipeline
description: How a FlyDSL kernel becomes HSACO — AST rewrite, tracing, the Fly→ROCDL→LLVM pass list, JIT cache key and fingerprint (and what it misses), IR/ISA dumps, ATT line info, and which compile hints actually reach LLVM.
source: https://github.com/ROCm/FlyDSL (python/flydsl/compiler/backends/rocm.py, jit_function.py, ast_rewriter.py, utils/env.py, docs/architecture_guide.md, .claude/skills/llvm/SKILL.md); https://llvm.org/docs/AMDGPUUsage.html
---

# FlyDSL compilation pipeline

## 1. The stack

```
 @flyc.jit / @flyc.kernel  (Python)
   │ ① cache lookup  key = (target, compile_hints, per-arg cache signature)
   ▼
 ASTRewriter      for/if/while → scf.for / scf.if / scf.while ; range_constexpr → unrolled
   ▼
 Tracing          run rewritten Python in an MLIR context → fly + gpu + scf + arith + vector IR
   ▼
 Pass pipeline    fly-* → convert-fly-to-rocdl → gpu→rocdl → LLVM dialect → fatbin (HSACO)
   ▼
 JITCFunction     ExecutionEngine wrapper; pickled to the disk cache
   ▼
 Runtime          hipModuleLoadData / hipModuleLaunchKernel
```

## 2. Pass list (ROCm backend)

| # | Pass | Job |
|---|---|---|
| 1 | `fly-rewrite-func-signature` | pack DSL arg types into LLVM structs; drop fully-static args |
| 2 | `fly-canonicalize` | fold layout algebra (static shapes vanish here) |
| 3 | `fly-rocdl-expand-ops` | expand target ops |
| 4 | `fly-layout-lowering` | `crd2idx`, divides, products → `arith` |
| 5 | `fly-int-swizzle-simplify` | peel addends out of swizzle bit patterns so CSE can share them |
| 6 | `canonicalize` | |
| 7 | `fly-convert-atom-call-to-ssa-form` | memref-level copy/MMA atom calls → SSA |
| 8 | `fly-promote-regmem-to-vectorssa` | register memrefs → vector SSA (no alloca left) |
| 9 | `convert-fly-to-rocdl` | atoms → ROCDL intrinsics (MFMA, buffer rsrc, `ds_*`) |
| 10 | `gpu.module(convert-scf-to-cf, cse, convert-rocdl-fastmath-ops, convert-gpu-to-rocdl, fly-rocdl-cluster-attr)` | |
| 11 | `rocdl-attach-target{O=2 chip=… fast=… unsafe-math=… wave64=…}` | target + math flags |
| 12 | scf/cf/vector/arith/func → LLVM, `reconcile-unrealized-casts` | |
| 13 | `ensure-debug-info-scope-on-llvm-func{LineTablesOnly}` | only with debug info on |
| 14 | `gpu-module-to-binary{format=fatbin}` | HSACO |

Read the pipeline from `RocmBackend._pipeline_parts` in your checkout. It changes between
releases (for example, passes 3 and the fastmath pass are recent).

Address spaces: Generic 0, Global 1, Shared/LDS 3, Register 5 (→ `alloca`, then promoted),
BufferDesc 8 (internal; buffer fat pointer).

## 3. JIT cache: what invalidates and what does not

```
 key  = ( ("_target_", arch, wave size),
          ("_hints_", sorted(compile_hints)),           # only if non-empty
          (param, cache_sig(arg)) for each param )       # Constexpr: value; runtime scalar: type
 disk = <cache_dir>/<func>_<toolchain fingerprint>/<hash(key)>.pkl
```

The source hash covers:
- the traced function source and its scalar closure values;
- helpers it calls **from the same directory** as the root file, recursively;
- any `@flyc.jit` / `@flyc.kernel` it calls, from any directory;
- the FlyDSL toolchain fingerprint (Python sources, native `.so`, version, external LLVM);
- extra dirs listed in `FLYDSL_EXTRA_SOURCE_DIRS`.

It does **not** cover plain helpers imported from another directory or package, or C++ pass
changes without a rebuild of the `.so`. In those cases set `FLYDSL_RUNTIME_ENABLE_CACHE=0`
or delete the cache dir. The in-memory cache is always on.

| Env | Effect |
|---|---|
| `FLYDSL_RUNTIME_CACHE_DIR` | disk cache location (default under `~/.flydsl/cache`) |
| `FLYDSL_RUNTIME_RUN_ONLY=1` | load from cache only; raise on miss (air-gapped deploy) |
| `COMPILE_ONLY=1` | compile, skip the ExecutionEngine (no-GPU CI gate) |
| `ARCH` / `FLYDSL_GPU_ARCH` | override target arch |

## 4. Dump and inspect

```bash
FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=./dumps python k.py
#   NN_<pass>.mlir per stage …  NN_llvm_ir.ll  …  NN_final_isa.s
FLYDSL_DEBUG_PRINT_AFTER_ALL=1   # IR after every pass
FLYDSL_DEBUG_AST_DIFF=1          # AST before/after rewrite (did my loop become scf.for?)
```

What to check in which file:

| Question | File |
|---|---|
| Did `range` become `scf.for`, or unroll? | first `.mlir` (post-trace) |
| Did the layout math fold to constants? | after `fly-layout-lowering` |
| Is the hint on the function? | `*_llvm_ir.ll` (`"amdgpu-waves-per-eu"="N"`) |
| MFMA / `ds_read_b128` / `buffer_load_dwordx4` present? | `*_final_isa.s` |
| VGPR, AGPR, spills, LDS | `llc -mtriple=amdgcn-amd-amdhsa -mcpu=<gfx> -pass-remarks-analysis=kernel-resource-usage NN_llvm_ir.ll` |

The dumped `.ll` has no target triple or CPU. Always pass both to `llc`.

### Source lines for ATT traces

Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`. Tracing records Python `loc()`s. The DI-scope pass
(row 13) turns them into `DISubprogram` metadata, and `.debug_line` lands in the HSACO.
rocprofv3 ATT then maps ISA to Python lines. Without that pass, the locations are dropped in
the MLIR→LLVM translation. A `-g` flag alone does nothing. Check: `.file` / `.loc` in
`final_isa.s`.

## 5. Which knobs reach LLVM

| Lane | Reaches LLVM as | Status |
|---|---|---|
| `compile_hints["waves_per_eu"]` | `rocdl.waves_per_eu` → `"amdgpu-waves-per-eu"` fn attr | live |
| `compile_hints["fast_fp_math"]` | `rocdl-attach-target fast=true` + fastmath flags on traced ops | live |
| `compile_hints["unsafe_fp_math"]` | `unsafe-math=true` only; **no** op flags | partial |
| `compile_hints["llvm_options"]` | LLVM `cl::opt`s set in-process | live |
| kernel / value attrs (`passthrough`) | raw `gpu.func` attributes | live |
| anything via `gpu-module-to-binary opts=` | — | **dead on AMD**: ROCDL never reads it |
| `FLYDSL_COMPILE_OPT_LEVEL` | — | in cache key, no consumer (`O=2` fixed) |

- The hint dict has no whitelist. A misspelled key changes the cache key (a real recompile)
  but does nothing else. Always run a **negative control**: misspell the key on purpose. Your
  IR check must then fail.
- `waves_per_eu`: CDNA max is 8 waves/SIMD, gfx1250 max is 16. The same number means a
  different occupancy fraction per family.
- Target occupancy with `waves_per_eu`, not a register cap.
- `@autotune(Config(...))`: plain kwargs become `Constexpr` args. Only `waves_per_eu` is
  forwarded as a compile hint. Set other hints around the tuned call with
  `CompilationContext.compile_hints({...})`. Results are cached as JSON under
  `FLYDSL_AUTOTUNE_CACHE_DIR`.

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `python/flydsl/compiler/backends/rocm.py` (`_pipeline_parts`, `lower_compile_hints`), `python/flydsl/compiler/jit_function.py` (`_is_user_function`, fingerprint, `_dump_isa`), `python/flydsl/compiler/ast_rewriter.py`, `python/flydsl/utils/env.py`, `python/flydsl/autotune.py`, `docs/architecture_guide.md`, `docs/autotune_guide.md`, `.claude/skills/llvm/SKILL.md`; [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html) (`amdgpu-waves-per-eu`, `amdgpu-flat-work-group-size`); [rocprofv3 ATT](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/).
Paths drift; verify before use.

---
Related: [kernel-authoring-primitives](kernel-authoring-primitives.md) · [pitfalls](pitfalls.md) · [compiler-build](../../kernel-optimization/levers/compiler-build.md) · [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md)
