---
name: compilation-pipeline
description: "FlyDSL compile details the skills lack: what the JIT cache key and fingerprint miss, which dump file answers which question, and which compile-hint lanes reach LLVM."
source: https://github.com/ROCm/FlyDSL (python/flydsl/compiler/backends/rocm.py, jit_function.py, ast_rewriter.py, utils/env.py, docs/architecture_guide.md, .claude/skills/llvm/SKILL.md); https://llvm.org/docs/AMDGPUUsage.html
---

# FlyDSL compilation pipeline

## 1. Pipeline and passes

The stack (AST rewrite → tracing → Fly→ROCDL→LLVM passes → HSACO) and the key passes are in
[flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §1. The exact pass list changes between releases: read `RocmBackend._pipeline_parts`
in your checkout.

## 2. JIT cache: what invalidates and what does not

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
| `FLYDSL_RUNTIME_RUN_ONLY=1` | load from cache only; raise on miss (air-gapped deploy) |
| `COMPILE_ONLY=1` | compile, skip the ExecutionEngine (no-GPU CI gate) |
| `ARCH` / `FLYDSL_GPU_ARCH` | override target arch |

## 3. Dump and inspect

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

Source-line info for ATT traces (`FLYDSL_DEBUG_ENABLE_DEBUG_INFO`, the DI-scope pass): [flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §9.

## 4. Which knobs reach LLVM

| Lane | Reaches LLVM as | Status |
|---|---|---|
| `compile_hints["fast_fp_math"]` | `rocdl-attach-target fast=true` + fastmath flags on traced ops | live |
| `compile_hints["unsafe_fp_math"]` | `unsafe-math=true` only; **no** op flags | partial |
| `compile_hints["llvm_options"]` | LLVM `cl::opt`s set in-process | live |
| kernel / value attrs (`passthrough`) | raw `gpu.func` attributes | live |
| `FLYDSL_COMPILE_OPT_LEVEL` | — | in cache key, no consumer (`O=2` fixed) |

- The hint dict has no whitelist. A misspelled key changes the cache key (a real recompile)
  but does nothing else. Always run a **negative control**: misspell the key on purpose. Your
  IR check must then fail.
- `waves_per_eu`, the dead `opts=` lane, `maxnreg`, and `@autotune`: [flydsl-kernel-authoring](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills/flydsl-kernel-authoring) §9.
- `@autotune(Config(...))` forwards only `waves_per_eu` as a hint. Set other hints around the
  tuned call with `CompilationContext.compile_hints({...})`.

Sources: [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) — `python/flydsl/compiler/backends/rocm.py` (`_pipeline_parts`, `lower_compile_hints`), `python/flydsl/compiler/jit_function.py` (`_is_user_function`, fingerprint, `_dump_isa`), `python/flydsl/compiler/ast_rewriter.py`, `python/flydsl/utils/env.py`, `python/flydsl/autotune.py`, `docs/architecture_guide.md`, `docs/autotune_guide.md`, `.claude/skills/llvm/SKILL.md`; [LLVM AMDGPU usage](https://llvm.org/docs/AMDGPUUsage.html) (`amdgpu-waves-per-eu`, `amdgpu-flat-work-group-size`); [rocprofv3 ATT](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/).
Paths drift; verify before use.

---
Related: [kernel-authoring-primitives](kernel-authoring-primitives.md) · [pitfalls](pitfalls.md) · [compiler-build](../../kernel-optimization/levers/compiler-build.md) · [isa-resource-inspection](../../kernel-optimization/levers/isa-resource-inspection.md)
