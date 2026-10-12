---
name: compiler-build
description: build type, common AMD/HIP flags, DSL-specific compiler hazards (conservative pass pipeline; -enable-post-misched=0 miscompiles MLIR kernels; AST-rewriter if-elision trap), and alias/pointer semantics. Flags rarely close >1% gaps.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 10. Compiler And Build Settings

### 10.1 Build Type

- Use Release builds for performance.
- Ensure `NDEBUG` is set for C/C++ kernels when assertions affect
  device code (make sure the DSL's compile driver passes it through).
- Confirm target architecture explicitly
  (e.g. `amdgcn-amd-amdhsa--gfx950`).
- Save compile commands and binary metadata.
- Avoid comparing debug and release builds.

### 10.2 Common AMD/HIP Flags

The default flag set is usually `-O3`, with per-kernel overrides.

- `--offload-arch=gfx950` (or `gfx942`, `gfx90a`).
- `-O3`.
- `-DNDEBUG`.
- `-fno-offload-uniform-block` when required for launch assumptions
  or perf.
- `-mllvm -amdgpu-function-calls=false`.
- `-mllvm -amdgpu-early-inline-all=true`.
- `-mllvm --lsr-drop-solution=1`.
- `-mllvm -enable-post-misched=0` only if correctness is verified
  (this flag is **known** to miscompile some MLIR-generated kernels
  even when it works for hand-written HIP — see Section 10.3).

Compiler flags very rarely close large performance gaps. Rebuilding
a kernel with the full recommended flag stack including `-DNDEBUG`,
`-fno-offload-uniform-block`, `-mllvm -amdgpu-function-calls=false`,
`-mllvm -amdgpu-early-inline-all=true`, `-mllvm --lsr-drop-solution=1`,
`-mllvm -enable-post-misched=0` has been measured to move throughput
by under 1 % across multiple kernels and DSLs. If you see a multi-×
gap, do not blame compiler flags first; instrument the ISA, the
bottleneck class, and the kernel structure.

### 10.3 DSL-Specific Compiler Hazards

- A DSL may ship only a conservative pass pipeline: canonicalize →
  conservative integer constant fold (add/sub/mul/div/mod/and/or/zext/
  sext/cmp/select) → CSE on pure ops → DCE on unused pure ops, up to 3
  iterations. No vectorizer, no fuser, no scalar-elision pass. Loads /
  stores / MFMA / barriers are never moved.
- Its LLVM lowering may not vectorize, fuse, or elide scalar work
  either; the only scalar optimization may be unrolling plus trailing
  sync elision. Write vectorized, hoisted code in the source.
- Specific MLIR/FlyDSL trap (a related one):
  `-mllvm -enable-post-misched=0` is safe in CK's CMake-generated code
  but produces wrong outputs in MLIR-generated kernels for some
  shapes. The other CK-style flags
  (`amdgpu-function-calls=false`, `amdgpu-early-inline-all=true`,
  `lsr-drop-solution=1`, `-fno-offload-uniform-block`, `-O3`, fast/
  unsafe math) were safe. Always treat compiler flags as
  one-at-a-time experiments with a correctness check after each one.
- DSL AST-rewriter trap: in tracing-style DSLs (like FlyDSL) the AST
  rewriter classifies any `if` whose test contains an `ast.Name` as
  potentially dynamic and converts it into an `scf.if` dispatch. A
  `if loads_next is not None:` guard around a Python-time decision
  is silently rewritten and the body elided, dropping all in-loop
  LDS stores while still emitting all loads, MFMAs, and barriers (and
  producing wrong but plausible-looking outputs). A DSL can avoid
  this by raising `TypeError` on boolean conversion of a runtime
  value; then you MUST use a static-if construct for Python booleans
  and an `scf.if`-style construct for runtime predicates.

### 10.4 Alias And Pointer Semantics

- Preserve `noalias` metadata when safe — kernel parameter
  declarations should carry `noalias`, `readonly`, `writeonly`,
  `align=N`, and `dereferenceable=N` where they hold.
- Do not disable alias analysis casually.
- Mark pointers restrict / noalias in C++ when correct.
- Use buffer descriptors with proper bounds when possible.
- Avoid hidden aliasing between input / output / workspace.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#compiler-build).

---
Flag safety table: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md). gfx950 compiler caveats: [target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md). Anti-pattern (flags before ISA): [decision-heuristics](../failure-modes/decision-heuristics.md).
