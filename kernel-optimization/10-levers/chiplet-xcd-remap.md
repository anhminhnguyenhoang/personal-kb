---
name: chiplet-xcd-remap
description: How aiters pid_preprocessing.py helpers map line-for-line onto HK Algorithm 1 (Phase 1/2), the conservative defaults, and the split-K chiplet-awareness gap.
source: HipKittens paper (arXiv 2511.08083), HazyResearch blog, and HipKittens GitHub repo
---

# `remap_xcd` & `pid_grid` — walkthrough

Source: `aiter/ops/triton/utils/_triton/pid_preprocessing.py`
Interactive viz: open `../assets/remap_xcd_pid_grid.html` in a browser.
Narrative companion: [chiplet-scheduling](../../papers/hipkittens/01-paper/chiplet-scheduling.md) (XCD grouping, hierarchical windowed traversal).

---

## 1 · High-level intent

These four helpers are PID-preprocessing utilities that callers stick at the **very top** of a Triton kernel — *before* any compute — to re-permute the launch-order `program_id(0)` into a tile coordinate with good spatial locality on chiplet GPUs (MI300X/MI355X).

| Helper | Purpose | HipKittens Algorithm 1 |
|---|---|---|
| `remap_xcd_chunked(pid, GRID_MN, NUM_XCDS, CHUNK_SIZE)` | Force `CHUNK_SIZE` consecutive logical PIDs onto the same XCD | Phase 1 (knob *C*) |
| `remap_xcd(pid, GRID_MN, NUM_XCDS)` | Same idea with `CHUNK_SIZE = ceil(GRID/NUM_XCDS)` — each XCD gets one big run | Phase 1 at *C = max* (the L2-greedy extreme) |
| `pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)` | 1-D → 2-D unflatten; walk in vertical windows of height `GROUP_SIZE_M` | Phase 2 (knob *W*) |
| `pid_grid_3d(pid, num_pid_m, num_pid_n, num_pid_k)` | 1-D → 3-D unflatten for split-K (no chiplet awareness) | — |

The two phases compose: kernels typically do `pid = remap_xcd_chunked(pid, ...); pid_m, pid_n = pid_grid(pid, ...)`.

## 2 · Hardware context (one paragraph)

MI355X = 8 XCDs × 32 CUs × 4 MB private L2 per XCD, sharing one LLC above HBM. The HW dispatcher round-robins block IDs across XCDs: block *k* → XCD *k % 8*. Combined with row-major launch order, this scatters each XCD's work across the whole output (measured **55 % L2 hit rate** on 9216³ BF16 GEMM). The remap defeats round-robin by feeding the HW a permutation that puts overlapping work on the same XCD; the windowed traversal then walks that work in 2-D tiles so each window reuses both a row of A and a column of B.

## 3 · The math, in one screen

### `remap_xcd_chunked` — Phase 1

```python
# HW dispatch rule is fixed: block with id `new_pid` runs on XCD `new_pid % NUM_XCDS`.
# Goal: pick a `new_pid` such that CHUNK_SIZE consecutive *logical* pids land
# on the same physical XCD, defeating the round-robin scatter.

xcd          = pid % NUM_XCDS              # 1) which XCD HW would have run THIS pid on
                                           #    (= dealer label of the card we were dealt)

if pid > (GRID_MN // (NUM_XCDS*CHUNK_SIZE)) * (NUM_XCDS*CHUNK_SIZE):
    return pid                             # 2) tail guard: pids past the last full
                                           #    "cycle" of NUM_XCDS*CHUNK_SIZE pass through
                                           #    unchanged — remapping them would produce
                                           #    out-of-range or colliding new_pids.

local_pid    = pid // NUM_XCDS             # 3) my index inside MY XCD's private pid list.
                                           #    HW gives each XCD every NUM_XCDS-th pid, so
                                           #    XCD 1 sees pids [1, 9, 17, 25, ...]. For pid=17,
                                           #    local_pid = 17//8 = 2 → I'm at position 2 in
                                           #    XCD 1's list (the 3rd pid it has seen).
                                           #    This dense 0,1,2,3,... counter is what lets
                                           #    the next two lines reason about chunks.

chunk_idx    = local_pid // CHUNK_SIZE     # 4) which group of CHUNK_SIZE consecutive
                                           #    logical pids do I belong to? (chunks
                                           #    are numbered 0, 1, 2, ... globally)

pos_in_chunk = local_pid %  CHUNK_SIZE     # 5) my offset inside that chunk (0..CHUNK_SIZE-1)
                                           #    — this becomes the low bits of new_pid

return chunk_idx * NUM_XCDS * CHUNK_SIZE   # 6a) jump to the start of my chunk's "cycle"
                                           #     (each cycle is NUM_XCDS * CHUNK_SIZE pids wide,
                                           #     one chunk for every XCD)
     + xcd       * CHUNK_SIZE              # 6b) inside the cycle, jump to MY XCD's slot.
                                           #     This is the key step: because new_pid % NUM_XCDS
                                           #     will equal `xcd`, HW routes me to the same XCD
                                           #     the original `pid` already implied.
     + pos_in_chunk                        # 6c) inside my XCD's slot, take my offset.
                                           #     Together: consecutive logical pids in the same
                                           #     chunk get consecutive new_pids that all land
                                           #     on the same physical XCD.
```

Intuition: HW deals you a card (`pid`); you can see which dealer's pile it came from (`xcd`) and where in the pile (`local_pid`). The new ID glues piles into runs of `CHUNK_SIZE` cards, each fully inside one XCD.

### `pid_grid` — Phase 2

```python
if GROUP_SIZE_M == 1:                       # row-major baseline
    pid_m = pid // num_pid_n
    pid_n = pid %  num_pid_n
else:
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id         = pid // num_pid_in_group
    first_pid_m      = group_id * GROUP_SIZE_M
    group_size_m     = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid %  group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

Intuition: stack the output into horizontal strips of `GROUP_SIZE_M` rows; walk each strip column-by-column. Consecutive PIDs share a column of B; the strip bounds reuse of A's rows.

## 4 · ASCII: what each helper buys you

### Default round-robin (no helpers)

```
output tile grid (rows = pid_m, cols = pid_n), N=8
colour = XCD (= pid % 8)
   0  1  2  3  4  5  6  7
0 [0][1][2][3][4][5][6][7]
1 [0][1][2][3][4][5][6][7]    ← every column of every row goes to all 8 XCDs
2 [0][1][2][3][4][5][6][7]    ← XCD 0's work is a vertical stripe of col 0
3 [0][1][2][3][4][5][6][7]    ← no 2-D locality anywhere
```

### After `remap_xcd_chunked(CHUNK=4)` + `pid_grid(W=1)`

```
   0  1  2  3  4  5  6  7
0 [0][0][0][0][1][1][1][1]    ← XCD 0 owns a 4-wide horizontal stripe
1 [2][2][2][2][3][3][3][3]    ← good A-row reuse within each XCD's run
2 [4][4][4][4][5][5][5][5]    ← but no B-column reuse (each tile a new col)
3 [6][6][6][6][7][7][7][7]
```

### After `remap_xcd_chunked(CHUNK=4)` + `pid_grid(W=4)` (both phases)

```
   0  1  2  3  4  5  6  7
0 [0][0][1][1][2][2][3][3]    ← XCD 0 owns a tight 4×2 rectangle of tiles
1 [0][0][1][1][2][2][3][3]    ← both A-row AND B-column reuse within XCD
2 [0][0][1][1][2][2][3][3]    ← this is what L2 wants
3 [0][0][1][1][2][2][3][3]
```

The HTML viz lets you toggle each phase and watch the colouring change in real time.

## 5 · Defaults vs sweet spots

| Knob | File default | HK paper sweet spot (9216³ BF16) |
|---|---|---|
| `CHUNK_SIZE` / *C* | **2** | **25** |
| `GROUP_SIZE_M` / *W* | **1** (= row-major) | **5** |

Defaults are "barely-on" so a caller who imports the helpers but forgets to tune doesn't silently regress. The L2-greedy trap (`remap_xcd` alone with W=1) is real: paper measured 14.9 TB/s vs 15.1 TB/s baseline vs 18.3 TB/s balanced.

## 6 · Where these get called (sample)

```
csrc → not relevant
aiter/ops/triton/_triton_kernels/
  gemm/basic/{gemm_a16w16,a8w8,a16w8_blockscale,...}.py
  gemm/feed_forward/ff_a16w16_fused_gated.py
  gemm/fused/fused_gemm_*.py
  moe/{moe_op,moe_op_mxfp4,moe_op_gemm_a4w4,...}.py
  attention/{lean_atten,mla_decode_rope}.py
```

Notable gap: the sparse-MLA 3-D split-K path uses `pid_grid_3d`, which **does not** apply Phase 1 or Phase 2. For low-batch decode this likely scatters the MN slice across XCDs — an unexplored tuning lever flagged in `MEMORY.md` (see [[sparse_mla_3d_split_k_lever]]).

## 7 · Decision rubric for new kernels

```
Does the kernel read input data that could be reused across blocks?
  ├─ No (elementwise op) ────────────────► skip both helpers
  └─ Yes (GEMM / attention / conv)
     │
     ├─ Small problem (a few hundred blocks)
     │  └─ Phase 1 + Phase 2 still help, but marginally
     │
     └─ Big problem (10k+ blocks)
        └─ remap_xcd_chunked + pid_grid, sweep (C, W).
           Start (C, W) ≈ (16, 4) or (32, 8). Watch L2 AND LLC hit
           rate — never sacrifice one for the other. Target the
           bandwidth column, not the hit-rate column.
```

## 8 · References

- HipKittens paper — arXiv [2511.08083](https://arxiv.org/abs/2511.08083), Algorithm 1 (lines 10–23).
- HK blog — [hazyresearch.stanford.edu/blog/2025-11-09-hk](https://hazyresearch.stanford.edu/blog/2025-11-09-hk).
- HipKittens GitHub repo — [HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens).

## Lessons from worked examples

- **The remap needs more tiles than one chunk, and a chunk that fits the reuse window.** If `tiles ÷ chunk ≈ 1` the remap is a no-op and the entry-time index math is pure overhead (small loss). When it does help on a small shared operand, the best chunks are small: the chunk must be short enough that the shared operand's reuse stays inside one XCD's residency. Wider chunks regressed. Expect a modest gain (a few %) on bandwidth-bound shapes, and confirm it with paired repeats.

- **Decouple the visiting index from the tile id in a persistent loop.** If the grid is a multiple of the XCD count, a block's XCD is fixed by its index, so group-major linear tile ids spread each group's tiles over all XCDs and the shared operand is fetched into every L2. Keep the strided loop variable (it maps to the XCD) separate from the tile id, then apply the chunked remap from visiting index to tile id so a run of consecutive tiles of one group lands on one XCD.

---
Theory: [chiplet-scheduling](../../papers/hipkittens/01-paper/chiplet-scheduling.md). Interactive viz: [assets/remap_xcd_pid_grid.html](assets/remap_xcd_pid_grid.html)

