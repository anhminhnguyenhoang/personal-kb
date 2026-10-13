---
name: tensilelite-kernels-and-selection
description: How TensileLite names GEMM problem types (Cijk), how logic YAMLs map shapes to solutions (Equality, GridBased, Range), and what each kernel parameter in a solution name controls.
source: https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/tensilelite, https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
---

# TensileLite: kernel naming and solution selection

TensileLite is the kernel generator and selector inside hipBLASLt. It generates assembly offline from parameterized solutions. At runtime it matches the problem against per-arch logic tables.

```
benchmark YAML ─► Tensile (bench sweep) ─► LibraryLogic (winners per size) ─► logic YAML per arch
       ─► TensileCreateLibrary (asm codegen → .co) ─► runtime: MasterSolutionLibrary.findTopSolutions(problem)
```

## Reading a problem-type name

`Cijk_Alik_Bljk_BBS_BH_Bias_HAS_SAV_UserArgs`:

| Piece | Meaning |
| --- | --- |
| `Cijk` | output indexed by free dims i, j and batch k |
| `Ailk` / `Alik` | A with the summation index `l` first or second. Shows if A is transposed. |
| `Bljk` / `Bjlk` | the same for B |
| `BBS` | types: A/B bf16, D bf16, compute fp32. Codes: H fp16, B bf16, S fp32, D fp64, I8, F8, B8 |
| `BH` | beta used, high-precision accumulate |
| `Bias`, `SAV`, `SAB` | bias epilogue, scale-alpha-vector, scale A/B |
| `UserArgs` | accepts runtime overrides (GSU, WGM, StaggerU) |

A kernel name adds the solution parameters to the type, for example `MT256x256x64_MI16x16x32_..._GSU..._WGM..`. Read the macro tile (`MT`) and the MFMA shape (`MI`) first.

## Selection

Each arch directory (for example aquavanjaram/gfx942 and gfx950, with CU-count variants) holds logic YAMLs. Each YAML holds a problem type, a list of solutions, and a matching table.

| Strategy | How it matches | Use |
| --- | --- | --- |
| **Equality** | exact `(M, N, batch, K)` → solution | known hot shapes, such as LLM projections |
| **GridBased** | nearest point on a multi-dimensional grid | default coverage |
| **Range** | recursive min/max splits per dimension | older or coarse coverage |

The runtime tries exact matches first, then the approximate libraries, and returns the top N. If it finds too few, it adds every solution that passes `isSolutionSupported`. **An unseen shape gets a nearest-neighbour guess.** That is why tuning your exact shapes often gives large gains. Code objects load lazily, one group at a time, so startup does not load the whole library.

## Solution parameters

| Param | Controls | Rule of thumb |
| --- | --- | --- |
| MacroTile (MT0×MT1) | output tile per workgroup | big tiles for big square problems. Use a small side when M or N is small (tail and wave quantization). |
| MatrixInstruction (MI) | the MFMA shape and its blocking | 16×16 has finer granularity. 32×32 covers more output per instruction and uses more VGPRs. |
| MIWaveTile, MIWaveGroup | MFMA tiles per wave, waves per WG along M/N | MT = MI × WaveTile × WaveGroup (per dim) |
| DepthU | K elements per main-loop iteration | larger means less loop overhead and more LDS |
| GlobalSplitU (GSU) | split-K across workgroups | for large K with too few output tiles. Needs workspace or atomics. |
| StreamK | split the K-iteration work evenly over CUs | when tiles ÷ CUs leaves a bad tail |
| WorkGroupMapping (WGM), WGM-XCC | WG → tile order, chiplet spread | L2 reuse. See [chiplet-xcd-remap](../../kernel-optimization/levers/chiplet-xcd-remap.md). |
| DirectToLds / DirectToVgpr | global→LDS without VGPRs / skip LDS | DTL for large tiles. DTV for small tiles that LDS slows down. |
| PrefetchGlobalRead / LocalRead | depth of the global and LDS prefetch | more depth hides more latency and uses more LDS or VGPRs |
| StaggerU | offset the start in K per WG | spreads DRAM channel and TLB pressure on large GEMMs |
| NonTemporal A/B/C/D | cache hints per tensor | streaming operands should not evict reused ones |
| LdsPad, LdsBlockSizePerPad | LDS padding | removes bank conflicts |

LDS limits occupancy directly. gfx942 has 64 KB LDS per CU and gfx950 has 160 KB. A 256×256 double-buffered tile can use more than 128 KB, which leaves one workgroup per CU. See [gfx950](../../hardware/amd-cdna/gfx950.md).

## Rough starting points (verify by sweeping)

| Shape | Start with |
| --- | --- |
| large square (M, N, K ≥ 4K) | MT 256×256 or 128×128, DepthU 64, PGR 2, DTL, WGM 4–16, StreamK |
| tall-skinny (M ≫ N or large K with small M·N) | a small MT side, GSU 4–16 or StreamK |
| small batched (M = N ≈ 128, large batch) | MT 16×16 or 16×32, larger DepthU, LocalSplitU, no GSU |
| fp8 / MX | the K-wide MFMA (16×16×128), so a smaller DepthU is enough |

Sources: [TensileLite source](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/tensilelite), [hipBLASLt docs](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/), [Stream-K paper (arXiv:2301.03598)](https://arxiv.org/abs/2301.03598), [AMD CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf).
Heuristics are secondhand. Verify specifics against the primary source.
Paths drift; verify before use.

---
Related: [api-concepts](api-concepts.md) · [benchmarking-and-tuning](benchmarking-and-tuning.md) · [work-decomposition](../../kernel-optimization/levers/work-decomposition.md)
