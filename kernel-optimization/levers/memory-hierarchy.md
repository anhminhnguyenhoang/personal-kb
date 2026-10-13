---
name: memory-hierarchy
description: physical storage taxonomy (registers vs LDS vs L1/L2 vs HBM — what's on-chip where, addressable vs cache vs scratchpad), global loads/stores (coalescing, buffer-rsrc OOB), LDS/shared (async DRAM->LDS, double-buffer, swizzle), LDS bank conflicts (XOR vs padding per-arch rule), registers, and caches.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 6. Memory Hierarchy

### 6.0 Physical Layout and Storage Taxonomy

Before the levers, the physical picture. On-chip storage comes in
**three fundamentally different categories** — confusing them leads to
mistuning. From closest-to-ALU outward:

```
GPU package
└── XCD (chiplet)               ← 8 per MI300X/MI350X package
    └── CU (Compute Unit)
        └── SIMD (×4 per CU)
            ├── VGPR / AGPR     ← per-lane registers (operands)
            └── SGPR            ← per-wave scalar registers
        ├── LDS                 ← per-CU software-managed scratchpad
        └── L1 (vL1)            ← per-CU hardware cache
    └── L2 cache                ← per-XCD hardware cache, shared across CUs
└── (Infinity Cache) → HBM      ← off-die stacked DRAM, on-package via interposer
```

The three categories and why they are NOT interchangeable:

1. **Registers (VGPR / AGPR / SGPR)** — *not memory and not a cache.*
   No addresses; the compiler assigns operands to numbered registers
   (`v0`, `v1`, …). Live inside each SIMD (4 SIMDs/CU). VGPR/AGPR are
   per-lane; SGPR is per-wave (uniform across the wave). Fastest,
   smallest, closest to the ALU. **VGPRs are not L1.**

2. **LDS (Local Data Share)** — real addressable memory with its own
   address space, but **software-managed**. Data only lives there
   because the kernel explicitly copied it in via `ds_write` / async
   DRAM→LDS, and reads it back via `ds_read`. Physically inside each
   CU; private to that CU (one threadblock maps to one CU), so it is
   the vehicle for intra-threadblock sharing. Banked SRAM (64 banks on
   gfx950, 32 on gfx942). **LDS is not L2** — it is a scratchpad, not a
   cache of DRAM. NVIDIA equivalent: shared memory.

3. **Caches (L1 / L2)** — real memory backing HBM, but
   **hardware-managed** and transparent. Never addressed directly; they
   intercept HBM accesses and fill/evict automatically. L1 (vL1) is
   per-CU; L2 is per-XCD, shared across all CUs on the chiplet.

The distinction that drives tuning:

| Category | Addressable? | Who manages it | Tuning lever |
|---|---|---|---|
| Registers | No (named operands) | Compiler | occupancy / spill |
| LDS | Yes (own space) | **Software** (explicit copy) | bank-conflict / swizzle |
| L1 / L2 | No (transparent) | **Hardware** (auto) | hit-rate / reuse, chiplet remap |

This is why GEMM/attention kernels do an explicit HBM→LDS staging
step: deliberately parking a tile in the CU scratchpad so the whole
wave reuses it without re-reading HBM. And why LDS is tuned for bank
conflicts while L2 is tuned for hit rate (e.g. chiplet/XCD block
remap reordering work so neighboring blocks reuse L2 lines). Per-arch
sizes (HBM, LDS/CU, register file): see
[target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md)
for the per-arch tables.

### 6.1 Global Memory Loads

- Coalesce loads: use one synchronous DRAM → VGPR → LDS staging
  pattern that keeps lanes contiguous.
- Use 16 B or larger vector loads when aligned.
  Buffer or global f16 vector loads of N ∈ {1, 2, 4} dwords.
- Use buffer loads on AMD for robust bounds behavior:
  a buffer-resource descriptor built from the base pointer and byte
  count, with `flags=0x00027000`; OOB lanes return zero.
- Use raw pointer loads only when alias / alignment is clear.
- Avoid scalar loads where vector load is possible.
- Hoist invariant loads out of loops.
- Reuse loaded values across multiple outputs.
- Use read-only cache hints if available
  (`AUX = CACHE_ALL / CACHE_GLOBAL / CACHE_STREAM / NON_TEMPORAL`).
- Avoid replay from misalignment.
- Handle tails with masked loads or padded descriptors
  (`transforms.pad`/`pad_dynamic`).

### 6.2 Global Memory Stores

- Store contiguous vectors.
- Avoid per-element scalar stores in epilogues.
- Convert accumulator vectors to packed output vectors
  (vector truncate f32 → f16).
- Use direct vector stores if already coalesced
  (a direct epilogue).
- Use LDS epilogue only when it meaningfully improves global
  coalescing (a CShuffle epilogue). See Section
  9 for the trade-off.
- Avoid store barriers unless required for cross-thread staging.
- Ensure store validity checks do not introduce excessive branches.
- Check whether stores are 8 B, 16 B, or scalar in ISA via
  the ISA-inspection probe (`buffer_store_dwordx2` / `_x4` vs
  `buffer_store_short`).

Vectorizing the epilogue is often the single largest optimization
for kernels that already have a good main loop. Replacing four scalar
`buffer_store` of `fp16` with one `fp16x8` `buffer_store` per lane
has been measured to roughly double the throughput on direct-conv
kernels — with no other change. Always inspect the epilogue ISA
(the ISA-inspection probe) before tuning the main loop; the `vmem_store`
bucket reveals scalar-store kernels immediately.

### 6.3 LDS / Shared Memory

- Use LDS to share input or weights across waves / threads.
- Avoid LDS if data is not reused enough.
- Use double buffering when global load latency matters
  (a double-buffered async pipeline).
- Use direct global-to-LDS async copies when available.
  Wrap the `raw_ptr_buffer_load_lds_addr` intrinsic with a per-wave
  LDS base.
- Avoid register-intermediate staging if direct DMA is possible.
- Swizzle LDS layout to reduce bank conflicts
  (an LDS layout object with a K-padding parameter).
- Add padding to avoid bank conflicts.
- Match LDS write layout to LDS read layout.
- Keep LDS footprint under occupancy thresholds (verify with
  the occupancy probe).
- Reuse LDS regions after phases when lifetimes do not overlap.
- Use one shared pool for multiple phases when safe.

Switching from `buffer_load → register → vector.store LDS` to direct
DRAM-to-LDS via the AMD `raw_ptr_buffer_load_lds` /
`amd_async_buffer_load` intrinsic is worth measuring even when
register staging looks fine. A direct-conv kernel was measured to
roughly double throughput from this change alone, before any layout
modification.

Async DRAM-to-LDS instructions on AMD generally write lane-contiguous
LDS addresses (lane `i` writes at `lds_ptr + i * size`). Arbitrary
per-lane swizzled destination pointers may compile but produce wrong
output. We tested this directly: passing a per-lane swizzled LDS
pointer to `raw_ptr_buffer_load_lds` corrupted the result, so swizzle
has to be expressed in the address arithmetic of a lane-contiguous
distribution, the way CK's tile descriptors do, not by handing the
intrinsic an arbitrary LDS pointer.

`AsyncTileLoader.choose_dwords` selects `{4, 3, 1}` only (the AMDGPU
`raw_ptr_buffer_load_lds` intrinsic on this target does not accept 2
dwords).

Staging weights through LDS is not always a win. CK does it because
its tile distribution and weight read pattern fully amortize the
cost. A naive port to an MFMA kernel that already keeps weights in
registers has been measured to *regress* throughput. Test the LDS
weight path with the actual read distribution before adopting it.

Diagnostic signature: a kernel that allocates LDS for an intermediate
tensor (P in attention, A or B in GEMM) which it immediately reads
back will show `ds_write_b<N> / outer-iter ≈ tile_bytes / threads` plus
an inner-loop `s_barrier`. If the reference implementation keeps the
same tensor in registers, that LDS allocation is structural overhead.
A unified-attention worked example shows the P-in-LDS round-trip case: 16 KiB of LDS for the
P tile plus 64 × `ds_write_b16` per K-iter, removed by switching to
register-PV + transposed-PV reads.

### 6.4 LDS Bank Conflicts

- Inspect LDS access patterns, not just total LDS bytes. Use an
  LDS-conflict analyzer to combine rocprof counters and
  ISA inspection.
- For AMD wave64, bank conflicts can appear with regular `(row,
  channel)` layouts.
- Try XOR swizzle on two-dimensional layouts
  (an LDS layout with an XOR swizzle mode).
- Try cyclic-shift swizzle when CK uses it.
- Try padding strides (K-pad defaults: `+8` sync, `0` async).
- Use transposed LDS loads where supported
  (a transpose LDS reader using `ds_read_tr16_b64` /
  `ds_read_tr_b8`).
- Count `ds_read` and `ds_write` instructions with
  the ISA-inspection probe.
- Compare swizzled vs contiguous variants with
  the intrinsic-count probe.

Cyclic-shift or XOR swizzle on a 2D tile layout only pays off in our
experiments when paired with the matching async load distribution.
With register-staged LDS writes, a manually-swizzled LDS layout has
been measured to regress slightly. Bank-conflict reduction alone is
not enough; the swizzle has to be aligned with how the load is
actually delivered to LDS.

#### 6.4a LDS Swizzle Strategy Selection: XOR vs Padding

Notation:

- **LDS_total**: Total LDS allocated by your kernel (per CU). From
  the occupancy probe or a static HSACO analysis.
- **LDS_avail**: Available LDS per CU on the target architecture.

| Architecture | GPU | LDS per CU | Preferred swizzle |
|---|---|---|---|
| gfx90a | MI210 / MI250X | 64 KB | XOR (capacity precious) |
| gfx942 | MI300 | 64 KB | XOR (capacity precious) |
| gfx950 | MI350X / MI355X | 160 KB | padding (capacity abundant) |

Two primary approaches exist for eliminating LDS bank conflicts, with
dramatically different performance characteristics depending on GPU
architecture:

**XOR Swizzle**: Mathematically proven bank-conflict-free addressing
using bitwise XOR operations.

- Cost: higher ALU overhead (more complex address computation; 5-7
  ALU vs 1-2 ALU per LDS access).
- LDS overhead: zero — no wasted LDS bytes.
- Best for: architectures with limited LDS capacity.
- Preferred when: LDS footprint is near capacity limits and occupancy
  depends on fitting in available LDS.
- Eliminates bank conflicts completely through address computation.

**Padding Swizzle**: Simple padding to avoid conflict patterns (e.g.,
`K=64 → K_PAD=72`).

- Cost: lower ALU overhead (simple offset calculation).
- LDS overhead: small percentage of total LDS (typically negligible
  on high-capacity architectures).
- Best for: architectures with abundant LDS capacity.
- Preferred when: LDS capacity is not the bottleneck and simpler
  addressing improves throughput.
- Eliminates bank conflicts through strategic padding.

Architecture-specific selection rule:

```python
if LDS_total < 96 * 1024:
    use_xor_swizzle()      # capacity-constrained
elif LDS_avail >= 128 * 1024:
    use_padding_swizzle()  # abundant LDS
else:
    benchmark_both()       # 96 KB <= LDS_b < 128 KB
```

**Scheduling-barrier caveat**: `sched_barrier` / `sched_group_barrier`
are compiler directives with no ISA encoding on any arch; they show up
only as asm comments. Check whether a hint hid the XOR overhead by
reading the instruction order, not by counting barriers in the ISA.

Critical asymmetry: `ds_write_b128` has 32-dword conflict period
while `ds_read_b128` has 64-dword period. Read and write conflict
behavior can differ — may need separate swizzle strategies. See
the empirical case studies and
[gfx950](../../hardware/amd-cdna/gfx950.md) for the full per-arch LDS table.

### 6.5 Registers

- Track VGPR and SGPR usage with the occupancy probe.
- High VGPR can reduce occupancy.
- Too-low VGPR cap can spill or reduce scheduling freedom.
- Hoist invariants but avoid keeping large temporary structures
  alive.
- Store only scalar state for tile windows / descriptors.
- Avoid persistent heavyweight objects in kernels.
- Prefer precomputed offsets over live descriptor state.
- Use compile-time constants to reduce scalar arithmetic
  (`IRBuilder.const_i32`).
- Watch register growth when adding double buffering or extra
  accumulators.
- Set `kernel.attrs["waves_per_eu"]` to override LLVM's
  occupancy hint when the compiler is wrong.

### 6.6 Caches

- Determine whether the benchmark uses warm cache or cold cache.
- Use rotating buffers if measuring bandwidth realistically.
- Use graph replay to separate cache behavior from launch overhead.
- Do not tune only for warm cache unless production has warm cache.
- Consider L2 prefetch only when the memory pattern is predictable.

## Lessons from worked examples

- **Direct global→LDS copy pays through the registers it frees.** Dropping the VGPR staging step removes the staging registers and instructions, but on a kernel that already waits on the load right away it can be neutral or slightly slower. The win comes when the freed registers let you deepen the tile (more K per iteration) that previously collapsed occupancy. Judge it together with the tile depth it unlocks (e.g. −32 VGPR let a 2× deeper K tile run at 2 waves/WG).
- **Async LDS copy is wave-level: give each wave its own LDS slice.** The instruction writes `wave_size × bytes_per_lane` contiguous bytes at a wave-uniform base. If several waves share one base they overwrite each other. Add `wave_id × wave_size × bytes_per_lane` to the base, for every operand (A and B). Signature: single-wave exact, multi-wave garbage, multi-wave without the async copy exact.
- **Default caching beats streaming/non-temporal hints when tiles are reused.** Consecutive K-tiles inside a CTA reuse L2, and a small operand shared by many CTAs reuses L1. Streaming or non-temporal hints throw that reuse away. Sweep hints per operand, but expect "cache all" to win unless the data is truly touched once.
- **Pre-permuted (preshuffled) weights turn strided tile loads into one burst.** If each per-K-tile load of a row-major operand touches many rows a full stride apart, a wave cannot coalesce it. Permute the operand once on the host so each tile is contiguous bytes; the in-tile order stays the same, so the LDS read pattern is unchanged. Keep the one-time permute out of the timed/captured region. The gain scales with the share of time spent in that load (≈2× where it dominates, ≈1.5% where addressing was already near-optimal).
- **Deep tiles hit an LDS-vs-occupancy frontier.** Per-WG footprint sets WGs/CU. Past the point where one more doubling drops WGs/CU (e.g. 2 → 1), the kernel turns latency-bound and gets slower, even if LDS still fits. Find the knee by sweeping depth, not by filling the LDS budget.

- **Direct-to-LDS copy can silently break an XOR-swizzled layout.** At some tile depths the async copy's destination pattern does not match the swizzle, and results collapse (cosine near 0.1) until a plain layout is used. The plain layout then brings back bank conflicts. An immediate `vmcnt(0)` wait after the copy also gives up its latency hiding. Validate at every tile depth, and pair the copy with deferred waits.
- **Load-to-LDS takes its LDS base from M0; advance it for every load.** The buffer/global load forms that write LDS use M0 as the wave's LDS base. If M0 is not updated between successive loads, later loads overwrite earlier slices. The bug can pass at small shapes and fail at larger ones (cosine ≈ 0.98), so test the largest tile count too.
- **Padding must keep 16-byte alignment.** A pad that removes bank conflicts but breaks 16-byte row alignment makes the compiler split `ds_read_b128` / `ds_write_b128` into narrower ops. Pad by a multiple of 16 bytes (e.g. +8 bf16 elements) and check the ISA for the wide forms.
- **Keep GEMM leading dimensions off multiples of 512 B.** Strides that are 512 B multiples map many rows to the same channel (the Tagram hotspot) and slow TN GEMM. Pad the leading dimension, e.g. `lda = ldb = K + 128` when `K % 256 == 0`. ([ROCm workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html))
- **Use DPP or permlane for fixed-pattern wave reductions.** A shuffle-XOR butterfly lowers to `ds_bpermute`, which crosses the LDS crossbar and waits on `lgkmcnt`. DPP row/bank modifiers ride on a VALU op at no memory cost. ([CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf), DPP)
- **Triage LDS stalls by where the stall sits.** Stall on the `ds_*` instruction itself: bank conflict, fix with swizzle or padding. Stall on `s_waitcnt lgkmcnt(0)` right after a `ds_write`: exposed latency, move the read later. A chain of `ds_bpermute` and barriers: cross-wave serialization, restructure the exchange. ([CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf), [FlyDSL tuning guide](https://github.com/ROCm/FlyDSL))

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#memory-hierarchy).

---
Epilogue store path: [epilogue](epilogue.md). Async copy + pipeline: [pipelining-scheduling](pipelining-scheduling.md). LDS swizzle knobs: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md); per-arch LDS table: [target-architecture-gfx950](../../hardware/amd-cdna/gfx950.md).
