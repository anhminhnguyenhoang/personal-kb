---
name: work-decomposition
description: grid / block / wave / thread mapping. Chiplet swizzle for CU balance, wave64 block sizes, MFMA lane-to-output mapping, and the canonical K-pack lane-mapping bug (validate at 1e-3).
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 5. Work Decomposition

### 5.1 Grid Mapping

- Map large independent dimensions to grid axes.
- Fold small batch / head / group dimensions into the x-axis if it
  improves scheduling. A per-kernel grid-size helper should own this.
- Avoid putting too much work only in grid-z if it reduces scheduler
  interleaving.
- Balance blocks across CUs. A chiplet (XCD) swizzle of the block id
  (8 XCDs on MI300X / MI325X / MI350X) can help.
- Ensure enough CTAs for occupancy.
- Avoid too many tiny CTAs with high launch overhead.
- Avoid too few huge CTAs that underfill the GPU.
- Consider persistent CTAs for irregular work (a persistent tile loop).

### 5.2 Block Mapping

- Choose threads per block to match wave count and occupancy.
- Common wave64 choices: 64, 128, 256, 512, 1024.
- Block size follows from the tile and wave-grid config (tile M/N/K,
  warps in M/N) and should be checked against the kernel's max
  workgroup size at lowering.
- More waves per block can improve data sharing but reduce occupancy.
- Fewer waves per block can improve occupancy but reduce reuse.
- Sweep wave count via the config-sweep probe with
  `num_warps ∈ {1, 2, 4, 8}` overrides.
- Sweep groups / heads per block.
- Sweep spatial columns per block.
- Sweep rows per block (rows per warp in a 2D attention kernel).
- Keep hot-loop per-block work high enough to amortize barriers.

### 5.3 Wave Mapping

- Decide what each wave owns. Pack tile + warp grid + bound
  `tid / lane / warp_* / block_*_off` values into one immutable view.
- Map lanes to contiguous memory where possible.
- Map lanes to MFMA operand layout exactly. A
  lane-to-output helper per atom shape documents the lane mapping.
- Avoid lane mapping that forces many shuffles.
- Avoid wasted lanes on small dimensions.
- Consider multiple independent small problems per wave.
- For `4x4x4` MFMA, use lane batching.
- For `16x16x16` MFMA, align `lane % 16` and `lane / 16` to matrix
  dimensions.
- For `16x16x32` MFMA, understand K-lane packing before using it.

For `mfma_f32_16x16x32_f16` on AMD CDNA, lane `(c4 = lane / 16)`
holds K elements `[c4*8 : c4*8 + 8]`, not a flat concatenation of two
4-element halves. A common bug when folding two filter columns
`S=0, S=1` into K=32 is to pack `[S=0 ch 0..3, S=1 ch 0..3]` per lane.
The correct mapping is `[S=0 ch 0..7]` for `c4=0`, `[S=0 ch 8..15]`
for `c4=1`, `[S=1 ch 0..7]` for `c4=2`, `[S=1 ch 8..15]` for `c4=3`.
The wrong packing compiles, runs, and validates within `1e-2` but
fails at `1e-3` (`max_abs ~ 5e-3`, ~10 % of elements bad). Always
validate K-pack lane mapping at `1e-3`, not just at `1e-2`.

### 5.4 Thread Mapping

- Assign vectorized global loads to contiguous lanes
  (a lane-contiguous load descriptor).
- Assign stores to contiguous lanes.
- Separate compute lanes from store lanes only if it improves
  coalescing.
- Use inactive lanes deliberately, not accidentally.
- Avoid dynamic modulo/divide in hot loops when compile-time
  decomposition is possible. Static loop / unroll / if constructs
  that are resolved at Python (trace) time give this.
- Precompute per-thread offsets outside loops.

## Lessons from worked examples

- **Wider tiles can leave CUs idle.** Widening the N tile divides the CTA count. When CTAs fall below the CU count, the idle CUs cost more than the extra reuse gives (e.g. 256 → 128 CTAs on 256 CUs). On skinny-M, memory-bound shapes the narrowest tile that still issues full-width loads usually wins, because more CTAs stream more of the weight in parallel.
- **Match tile size to the useful work per tile.** With sparse masks (sliding window, causal edge) a tile sized to the LDS maximum spends part of every K-iteration on masked-out columns. A smaller tile removes that wasted compute, LDS reads and waits in proportion.
- **Gate padded work with an early exit.** If the grid is sized for all buckets (experts, groups) but only some are populated, time scales with allocated tiles, not active ones. Have each CTA read its bucket's header and return before the K-loop when it is empty. Cost with all tiles active is ≈0; with all inactive the kernel just exits. Compute the gate from the same post-remap tile offset the body uses for addressing, or a chiplet/persistent remap will make the gate skip a tile the body still computes.

- **Persistent visitor with zero host sync.** For ragged groups, build a per-group prefix of occupied tiles on device and read the total into a scalar register (`readfirstlane`) so every block strides the same loop. Map a linear tile id to its group by a compile-time-unrolled binary search over the prefix; a linear scan over many groups becomes the bottleneck. Empty groups have zero-width intervals and are skipped for free. Building the work list on the host forces a sync and erases the win end to end.
- **Size the grid from the queried CU count.** Launch at least one full wave of workgroups across every CU and chiplet, and preferably several so the tail is short. CU counts differ inside one product family (e.g. 304 on MI300X, 256 on MI355X). A hardcoded count leaves CUs idle on one part or quantizes badly on the other.
- **Fill a skinny grid with split-K sized to dispatch slots.** For decode-shaped work (tiny M), choose `splits = min(slots / (batch × groups), reduction_len / block)`, with `slots` about 2× the CU count. Round up to a power of two and cap at the largest split the reduce stage supports. Parameterize it per instantiation, because the group count changes with the parallelism layout.
- **Matrix-bound kernels do not need high occupancy.** A large accumulator tile at 1–2 waves per SIMD, with deep in-wave pipelining, can beat more waves with a smaller tile. Prefer fewer waves with zero spills over more waves that spill. A waves-per-EU hint is only a request: check the ISA for scratch after you set it.
- **A static persistent partition leaves a tail.** If each persistent CTA gets a fixed slice of tiles, uneven tile cost makes the last CTAs finish late. Use an atomic work counter or a Stream-K split of the last tiles.
- **No workspace means no split-K or Stream-K.** A library heuristic queried with a workspace limit of 0 returns only kernels that need no reduction scratch. Small-MN, large-K shapes then lose their best options. Give the heuristic a real workspace. ([hipBLASLt API reference](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/api-reference.html))
- **Size the grid per phase for remote-memory traffic.** Phases that read from a peer GPU are latency-bound and need many waves in flight to hide link latency. Phases that post writes saturate the link earlier, often near half the CUs. Tune block counts per phase, not one value for the whole op. ([MoRI dispatch/combine v2](https://github.com/ROCm/mori))

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#work-decomposition).

---
Next: [memory-hierarchy](memory-hierarchy.md). MFMA operand layout: [matrix-instructions](matrix-instructions.md). Chiplet knobs: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md); arch caps: [target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md).
