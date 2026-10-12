---
name: chiplet-scheduling
description: "Algorithm 1: the MI355X chiplet hierarchy (XCD/L2/LLC), XCD grouping (Phase 1) + hierarchical windowed traversal (Phase 2), the L2/LLC tension and L2-greedy trap."
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# Topic B · The Chiplet Story (LLC, XCD Grouping, Hierarchical Windowed Traversal)

This is the meat of the walkthrough. The three concepts only make sense together, so I'll build them up in order:

1. The chiplet hardware (what an XCD is, where the LLC sits)
2. Why default scheduling is bad (the L2/LLC tension)
3. **XCD grouping** (Phase 1 of HK's Algorithm 1)
4. **Hierarchical windowed traversal** (Phase 2)
5. How they compose

## B.1 The MI355X memory hierarchy you need to picture

```
┌─────────────────────────────────────────────────────────────────┐
│                       HBM (288 GB)                              │
│                  ~few TB/s, ~500ns miss penalty                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │  LLC  (last-level)  │
                  │  shared by all XCDs │  ← single pool
                  │  ~1× bandwidth      │
                  │  ~300ns miss penalty│
                  └──────────┬──────────┘
                             │
        ┌─────┬─────┬─────┬──┴──┬─────┬─────┬─────┐
        ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
     ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐
     │XCD0││XCD1││XCD2││XCD3││XCD4││XCD5││XCD6││XCD7│
     │L2  ││L2  ││L2  ││L2  ││L2  ││L2  ││L2  ││L2  │   ← private L2 per XCD
     │4MB ││4MB ││4MB ││4MB ││4MB ││4MB ││4MB ││4MB │     ~3× LLC bandwidth
     │32CU││32CU││32CU││32CU││32CU││32CU││32CU││32CU│   ← 256 CUs total
     └────┘└────┘└────┘└────┘└────┘└────┘└────┘└────┘
```

Key facts to internalize:

| Thing | Number |
|---|---|
| XCDs (Accelerator Complex Dies) | **8** |
| CUs per XCD | **32** |
| Private L2 per XCD | **4 MB** |
| L2 to LLC bandwidth ratio | **~3:1** (L2 is faster) |
| L2 miss penalty | ~300 ns |
| LLC miss penalty (= go to HBM) | ~500 ns |

The critical structural point: **L2 is per-XCD and private. LLC is shared by all XCDs.** Data brought into XCD-3's L2 doesn't help XCD-5. Data in the LLC helps everyone but is slower.

This produces what the paper calls **orthogonal cache preferences**:

- **L2 wants:** thread blocks running on the same XCD to access the same input data (so the data lands in *their* L2 once and gets reused).
- **LLC wants:** thread blocks running on *different* XCDs to access overlapping rows/columns of the input (so the data lands in the LLC once and feeds all XCDs).

You can't naively optimize one without considering the other. That's the whole reason this algorithm exists.

## B.2 How the hardware schedules blocks by default

When you launch a kernel with `grid = (NX, NY)`, blocks get a flattened ID `xy = bx + NX * by`. The hardware dispatcher then sends them out **round-robin across XCDs**:

```
block 0 → XCD 0
block 1 → XCD 1
block 2 → XCD 2
block 3 → XCD 3
block 4 → XCD 4
block 5 → XCD 5
block 6 → XCD 6
block 7 → XCD 7
block 8 → XCD 0   ← wraps
block 9 → XCD 1
...
```

So **XCD k gets blocks { k, k+8, k+16, k+24, ... }** — every 8th block in launch order.

Combined with the standard row-major launch order `(bx, by) = (xy % NX, xy / NX)`, that means each XCD ends up running an **interleaved scatter** of output tiles — definitely not a rectangle.

## B.3 Why row-major + round-robin tanks the L2

Picture a GEMM `C = A @ B` with the output partitioned into a grid of tiles. Each output tile (bx, by) reads:

- One **row of A tiles** (always the same `by`-th row of A)
- One **column of B tiles** (always the same `bx`-th column of B)

So for L2 reuse, you want **blocks on the same XCD to share either `bx` or `by`** — they'll reuse the same column of B or the same row of A.

What you get under row-major + round-robin (with `NX = 16`, say):

```
output tile grid (rows = by, cols = bx):

      bx=0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
by=0  [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7][ 8][ 9][10][11][12][13][14][15]
by=1  [16][17][18][19][20][21][22][23][24][25][26][27][28][29][30][31]
by=2  [32][33]...
                                                              
XCD 0 runs:  block 0  block 8  block 16 block 24 ...
             (0,0)    (8,0)    (0,1)    (8,1)    ...
             
XCD 0's blocks land at (0,0), (8,0), (0,1), (8,1) — a sparse scatter.
They don't share rows of A (different by) consistently,
and don't share columns of B (different bx) consistently.
```

Measured: **L2 hit rate = 55 %**. LLC hit rate stays high (95 %) because the scattered pattern accidentally helps the shared cache, but effective bandwidth is bottlenecked by L2 misses → 15.1 TB/s on a 9216³ BF16 GEMM.

## B.4 The fix, in two phases

Algorithm 1 in the paper is a **block-ID remap done at the very top of the kernel** — it takes the hardware-assigned `xy` and computes a different `(row, col)` for the threadblock to work on. No kernel logic changes. It's purely a permutation of which threadblock does which tile.

The remap has two phases:

```
Phase 1: XCD Grouping              Phase 2: Hierarchical Windowed Traversal
─────────────────────              ──────────────────────────────────────────
"Make C consecutive logical        "Within an XCD's chunk of work,
 blocks land on the SAME XCD"       walk the output matrix in vertical
                                    windows of height W instead of row-major"

 Tuning knob: C                     Tuning knob: W
```

Both phases are needed. Phase 1 alone gives each XCD a contiguous chunk of blocks but they're still walked row-major (and a row-major chunk has poor 2D locality). Phase 2 alone schedules the matrix in tiles but doesn't undo the hardware round-robin, so each tile gets scattered across XCDs again.

---

## B.5 · Phase 1: XCD Grouping (deep dive)

**The pseudocode (lines 10–14 of Algorithm 1 in the paper):**

```python
xcd          = xy % nXCD                       # 1) which XCD did HW route me to?
local        = xy // nXCD                      # 2) position within that XCD's stream
chunk_idx    = local // C                      # 3) which "chunk" of size C am I in?
pos          = local % C                       # 4) position within the chunk
xy_new       = chunk_idx * (nXCD * C) + xcd*C + pos
```

That formula is just an inverse-shuffle. Walk through it intuitively:

The hardware deals you a card (`xy`); you can tell which dealer's pile it came from (`xcd = xy % nXCD`) and which position in that pile (`local = xy // nXCD`). Now you want to **re-glue the piles into runs of C cards**, where each run of C is fully contained in a single XCD. The last line `chunk_idx * (nXCD*C) + xcd*C + pos` is just "find your chunk; within that chunk, find your XCD's slot; within that slot, find your position".

**Concrete example: nXCD = 8, C = 3.**

Hardware will physically run block `xy_new` on XCD `xy_new % 8`. We want logical blocks 0,1,2 to all land on XCD 0; logical blocks 3,4,5 on XCD 1; etc. The remap accomplishes this by feeding the hardware a permuted list:

```
Logical block we want to run:   0  1  2 | 3  4  5 | 6  7  8 | 9  10 11 | ...
We want them on XCD:            0  0  0 | 1  1  1 | 2  2  2 | 3  3  3  | ...

Hardware-assigned xy (= the remapped xy_new it computes for):
   logical 0 → xy_new = 0    → HW runs on XCD 0  ✓
   logical 1 → xy_new = 8    → HW runs on XCD 0  ✓   (because 8 % 8 = 0)
   logical 2 → xy_new = 16   → HW runs on XCD 0  ✓
   logical 3 → xy_new = 1    → HW runs on XCD 1  ✓
   logical 4 → xy_new = 9    → HW runs on XCD 1  ✓
   ...
```

We use the hardware's round-robin against itself: by emitting block IDs in stride-8, we *force* consecutive logical blocks onto the same XCD.

**What C controls:**

- **Small C (e.g., 1)**: each XCD gets one block then we move on — that's just the default round-robin. No L2 grouping.
- **Large C (e.g., 256)**: each XCD gets a huge contiguous run of logical block IDs. Their accesses overlap heavily → great L2 hit rate. But: other XCDs are now working on *completely different* slices of the input, so the LLC sees no overlap → LLC hit rate crashes.

This is the L2/LLC tension showing up as a single knob: **C trades LLC for L2.**

You can see it in the paper's numbers:

| Config (9216³) | W | C | L2 hit | LLC hit | Bandwidth |
|---|---|---|---|---|---|
| Row-major baseline | — | — | 55 % | 95 % | 15.1 TB/s |
| L2-greedy | 7 | 216 | 79 % | **24 %** | 14.9 TB/s ← LLC crashed |
| Balanced | 5 | 25 | 75 % | 93 % | **18.3 TB/s** |

The L2-greedy config has the highest L2 hit rate of all three and is *worse than the baseline* on actual bandwidth.

---

## B.6 · Phase 2: Hierarchical Windowed Traversal (deep dive)

Even after Phase 1 gives each XCD a contiguous run of C logical block IDs, *how* those C blocks visit the output matrix still matters. Row-major would make an XCD process a stripe of width C in one row of output tiles — which reuses one row of A perfectly but reuses *nothing* of B (a different column each tile).

The fix: visit output tiles in **vertical windows of height W rows**, not row-by-row.

**The pseudocode (lines 15–23 of Algorithm 1):**

```python
num_rows       = M // BLOCK_M
num_cols       = N // BLOCK_N
tid_per_group  = W * num_cols       # blocks in one window of W rows
group_id       = xy // tid_per_group
first_row      = group_id * W
win_h          = min(num_rows - first_row, W)  # last window may be shorter
ℓ              = xy % tid_per_group
row            = first_row + (ℓ % win_h)
col            = ℓ // win_h
return (row, col)
```

Read it as: "Pretend the output is a stack of horizontal strips, each W rows tall. Within a strip, walk **column-by-column** going down each column before moving right."

**Visual: output tile grid for W = 4, 8 columns, row-major vs windowed:**

```
Row-major order (numbers = visit order):
                                                       
   col→ 0   1   2   3   4   5   6   7
row 0  [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]
row 1  [ 8][ 9][10][11][12][13][14][15]
row 2  [16][17][18][19][20][21][22][23]
row 3  [24][25][26][27][28][29][30][31]
row 4  [32][33]...                          ← next strip
        ^^^
        Consecutive blocks share a row (same row of A reused)
        but jump across all columns of B → B reuse terrible

Windowed (W=4) order:
        
   col→ 0   1   2   3   4   5   6   7
row 0  [ 0][ 4][ 8][12][16][20][24][28]
row 1  [ 1][ 5][ 9][13][17][21][25][29]
row 2  [ 2][ 6][10][14][18][22][26][30]
row 3  [ 3][ 7][11][15][19][23][27][31]
row 4  [32][36]...                          ← next strip
       ^^^^^^^^
       Consecutive blocks 0,1,2,3 share the same COLUMN of B
       (column of B reused) AND only span W rows of A 
       (row of A reused within W blocks)
```

The windowed order achieves **2D locality** — every consecutive group of W blocks reuses both a row tile of A and a column tile of B. That's exactly what an L2 cache wants.

**What W controls:**

- **W = 1**: degenerates to row-major (no improvement).
- **W = num_rows**: visits one full column of output tiles before moving to the next column. Maximizes B-column reuse but loses A-row reuse.
- **Moderate W (4–8 typical)**: balances row-of-A reuse vs column-of-B reuse.

The interaction with C from Phase 1: you'd typically want `C ≈ W × cols_per_window`, so an XCD's whole assigned chunk fits inside a couple of windows. The paper's empirical sweet spots: **8×4 or 4×8 L2 tiles** on MI355X for typical GEMMs.

---

## B.7 · How the two phases compose

Picture the end-to-end effect on one XCD's work assignment:

```
Default scheduling:
  XCD 0 gets blocks { 0, 8, 16, 24, ... }
  → scattered across the output, no useful locality
  → 55 % L2 hit rate

After Phase 1 only (C=4):
  XCD 0 gets logical blocks { 0,1,2,3, 32,33,34,35, 64,65,66,67, ... }
  → contiguous chunks but each chunk is a row-major run → poor 2D locality

After Phase 2 only (W=4):
  Output visited in 4-row windows
  → great 2D locality globally, but each window's blocks STILL get round-robined
    across 8 XCDs → each XCD sees only 1/8 of the window → cache fragmented

After Phase 1 + Phase 2 (C=25, W=5 say):
  XCD 0 gets a contiguous block-chunk; that chunk gets visited in 5-row windows
  → 75 % L2 hit, 93 % LLC hit, +19 % bandwidth
```

The trick is that **Phase 2 is applied to the remapped block ID from Phase 1**, so the windowed traversal happens *within* each XCD's chunk, not across all XCDs. That's why the algorithm is called *hierarchical* — there's an outer scheduling decision (which XCD, Phase 1) and an inner one (where in the matrix that XCD walks, Phase 2).

## B.8 · The numbers, in one table

From the paper's Table 4 on 9216³ and 14592³ BF16 GEMM on MI355X:

| Problem | Scheme | W | C | L2 | LLC | BW (TB/s) | TFLOPS |
|---|---|---|---|---|---|---|---|
| 9216³ | Row-major | — | — | 55 % | 95 % | 15.1 | 1113 |
| 9216³ | L2-greedy | 7 | 216 | 79 % | 24 % | 14.9 | 991 |
| 9216³ | **Balanced** | **5** | **25** | **75 %** | **93 %** | **18.3** | **1145** |
| 14592³ | L2-greedy | 8 | 542 | 79 % | 7 % | 13.9 | — |
| 14592³ | **Balanced** | **8** | **64** | **78 %** | **55 %** | **16.6** | — |

Two clean lessons:

1. **L2-greedy is a trap.** Maximizing L2 hit rate alone *loses* performance vs even the naive baseline because LLC starvation dominates.
2. **The right (W, C) depends on problem size.** Bigger problems can afford bigger chunks (C=64 vs C=25) because there's more reuse to amortize.

## B.9 · Where this lives in your launch path

This is all done at the **very top of the kernel**, before any compute:

```cpp
__global__ void gemm_kernel(...) {
    // Step 1: take the raw block ID the HW gave us
    uint32_t raw_xy = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Step 2: apply Algorithm 1 (XCD grouping + windowed traversal)
    auto [row, col] = xcd_windowed_remap(raw_xy, W, C, M, N, BLOCK_M, BLOCK_N);
    
    // Step 3: use (row, col) as the "real" output tile coordinates
    // ... rest of kernel computes A[row,:] @ B[:,col]
}
```

No kernel body changes. No extra memory. ~20 lines of integer math per launch. Up to +19–22 % bandwidth. This is one of the better cost/benefit ratios in the paper.

## B.10 · Side-by-side with NVIDIA

The paper doesn't go deep on this, but for context:

| | NVIDIA (Hopper/Blackwell) | AMD MI355X |
|---|---|---|
| Chiplet count | 1 (H100) / 2 (B200) | **8** (MI355X) |
| Per-chiplet private L2 | yes (split L2 on Blackwell) | **yes (per XCD, 4 MB)** |
| Block scheduling primitive | **Thread block clusters** (SMs grouped, shared SMEM) | **None at the API level** — you remap block IDs |
| Default scheduler | rasterized across SMs | **round-robin across XCDs** |
| Where the trick lives | hardware-level cluster placement | **kernel-level integer remap** |

NVIDIA gives you a first-class API (`__cluster_dims__`) to express "these blocks should be co-located on adjacent SMs." AMD gives you nothing — but the round-robin is predictable enough that you can *defeat* it with the right block-ID permutation. That's why HK's contribution is an algorithm, not a primitive.

## B.11 · Decision rubric

If you're writing a new CDNA kernel and wondering whether to use Algorithm 1:

```
Does your kernel read input data that could be reused across blocks?
   ├─ No (every block reads totally different data, e.g., elementwise op)
   │  └─ Skip Algorithm 1. There's nothing to cache.
   │
   └─ Yes (GEMM, attention, convolution, ...)
      │
      ├─ Is your problem small (a few hundred CUs of work)?
      │  └─ Algorithm 1 helps less; round-robin damage is bounded.
      │
      └─ Is your problem big (10k+ blocks)?
         └─ Apply Algorithm 1. Start with (W, C) ≈ (4, 16) or (8, 32),
            then sweep. Watch L2 *and* LLC hit rate — never sacrifice
            one for the other. Target the bandwidth column, not the
            hit-rate column.
```

---

## Glossary (all in one place)

| Term | Meaning |
|---|---|
| **HIPCC** | AMD's HIP compiler driver, a thin wrapper around Clang/LLVM that emits CDNA assembly. AMD's analogue of `nvcc`. |
| **VGPR** | Vector general-purpose register; per-thread, 256 per wave at 2 waves/SIMD. Can be MFMA input or output. |
| **AGPR** | Accumulator register; 256 per wave. Hardware allows MFMA input, but HIPCC won't emit it — costs 19 % until you pin registers manually. |
| **`v_accvgpr_read`** | The instruction HIPCC inserts to copy AGPR → VGPR before MFMA input. Pure overhead. |
| **`pinned_register_tile`** | HK primitive that names exact register numbers including AGPRs, bypassing HIPCC's allocator. |
| **XCD** (Accelerator Complex Die) | A chiplet inside an MI355X. 8 per GPU, 32 CUs each, with a **private 4 MB L2 cache**. |
| **L2** | Cache local to one XCD. Fast (~3× LLC bandwidth) but only helps blocks on the same XCD. |
| **LLC** (Last-Level Cache) | The single shared cache that sits between all XCDs' L2s and HBM. Slower but visible to everyone. |
| **HBM** | High-bandwidth memory — the off-chip DRAM. The thing you're trying to avoid hitting. |
| **Round-robin scheduling** | The default AMD policy: block ID `xy` runs on XCD `xy % nXCD`. Stripes work across all XCDs. |
| **Algorithm 1** | HK's chiplet-aware scheduler. A two-phase block-ID remap done at kernel entry. Two knobs: `W`, `C`. |
| **XCD grouping (Phase 1)** | First half of Algorithm 1. Inverts the hardware round-robin so C consecutive logical blocks run on the same XCD. Knob: **C** (chunk size). |
| **Hierarchical windowed traversal (Phase 2)** | Second half of Algorithm 1. Within an XCD's chunk, visits output tiles in vertical windows of height W instead of row-major, so each window reuses both a row tile of A and a column tile of B. Knob: **W** (window height). |
| **L2/LLC tension** | The orthogonal cache preferences: L2 wants intra-XCD locality, LLC wants inter-XCD overlap. Optimizing one alone breaks the other. |
| **L2-greedy trap** | Pushing W and C to maximize L2 hit rate (e.g., 79 %) starves the LLC (24 %) and ends up *slower* than the naive baseline. |
| **Sweet spot on MI355X** | "8×4 or 4×8 L2 tiles" per the paper — roughly W,C pairs that produce L2 ~75 %, LLC ~90 %. |
| **Thread block cluster** | NVIDIA's primitive (Hopper+) for co-locating blocks on adjacent SMs. AMD has no equivalent — Algorithm 1 is the workaround. |

---

## The one-sentence takeaways

> **HIPCC** is Clang/LLVM for HIP, and its single most expensive limitation is refusing to use AGPRs as MFMA inputs — HK gets around it with inline-asm-backed register pinning.

> **LLC** is the GPU-wide cache between the per-XCD L2s and HBM; **XCD grouping** (Phase 1, knob C) defeats the hardware's round-robin to give each XCD a contiguous chunk of blocks, and **hierarchical windowed traversal** (Phase 2, knob W) walks that chunk in 2D windows so each window reuses both a row of A and a column of B — together they convert AMD's chiplet hierarchy from a liability into a +19% bandwidth lever.

Sources:
- [HipKittens paper (arXiv 2511.08083, HTML)](https://arxiv.org/html/2511.08083v1)
- [HipKittens blog post](https://hazyresearch.stanford.edu/blog/2025-11-09-hk)


---
Repo implementation: [pid-preprocessing-vs-algo1](../../../kernel-optimization/10-levers/chiplet-xcd-remap.md). Related: [overview-thesis](overview-thesis.md)
