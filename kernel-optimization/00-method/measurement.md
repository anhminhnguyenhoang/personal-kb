---
name: measurement
description: "Benchmarking discipline for comparing a GPU kernel against an autotuning baseline: the autotune-median trap (use min/p10, never median), cold-L2 vs hot-L2 timing, why wall-clock lies at small shapes, correctness via cosine plus mean signed error on edge shapes, and change-one-thing verification."
source: distilled from GPU kernel benchmarking vs autotuning baselines on AMD CDNA3/CDNA4
---

# Measurement discipline

Getting the *measurement* right matters as much as any kernel change. A false
"kernel beats the baseline by 1.3x" claim can survive for days as a pure
measurement artifact. These rules apply to any comparison of a hand-written kernel
(e.g. one written in FlyDSL) against an autotuning baseline (e.g. Triton).

## 1. The autotune-median trap: use min/p10, never median

An autotuner runs many trial configs inside one benchmarked call. The slow trials
inflate the *median* by 30-40% (e.g. hundreds of dispatches spread over a 3x range
of times). A kernel that does not autotune has a clean median. Comparing the clean
median to the polluted median manufactures a fake win.

The fair number for an autotuning kernel is its **min** (the best config, which is
what deploys) or **p10** (steady-state best). Read the same statistic for both
sides. Match the profiler's kernel filter on the *full* distinctive kernel name. An
empty or over-broad substring can silently match the wrong kernel and report a
bogus time.

## 2. Cold-L2 vs hot-L2: it changes the verdict

- A cache-flushing benchmark helper (e.g. `do_bench`) flushes the L2 each rep. It
  gives the **cold-start** number and includes host launch overhead.
- A device-time kernel trace of back-to-back dispatches runs **hot**: warm L2,
  device time only. This is the realistic repeated-call pattern where weights stay
  L2-resident across calls.

The two differ most when the lever under test is L2 reuse (for example a chiplet
remap): a cold flush erodes it. A tie when hot and a small loss when cold is not
"the kernel regressing". The two numbers **bracket reality**. Quote both, and say
which one the deployment resembles.

## 3. Wall-clock lies at small shapes

At small shapes a kernel is launch-bound. Event-based wall-clock is then mostly
fixed host launch/dispatch overhead and reads flat across problem size. It can hide
a real store-path problem or a 30% tile regression. **Optimize against per-kernel
device time from a profiler.** Two levers can look identical in wall-clock while
device time moves 30%.

Do not expect a pure-Python substitute on every stack. Graph capture of a custom
launch path may produce an empty graph, and batched event timing can still be
dispatch-starved.

## 4. Correctness: cosine AND mean signed error, on the edges

A cosine gate (for example > 0.999 against eager) is necessary but **cosine alone
can mask a 123% relative error**. Use cosine *and* mean signed error. Always
include edge shapes: a partial bottom tile (rows not a tile multiple), an **empty
group**, one long group among many short ones, and a max length far above the
mean. A bf16 path is not bit-exact against an fp32 reference, so do not expect
`allclose`.

## 5. Change one thing, re-verify interleaved, re-measure surprises

- Change **one lever in isolation** on a clone. Leave the production kernel
  untouched. Verify correctness first, then measure.
- GPU clock drift between separate runs is real. Re-verify winners in a **single
  interleaved sweep** before promotion.
- **Re-measure a surprising negative on the *current* kernel** before trusting it.
  A "lever X regresses" result from an older kernel can reverse once measured on the
  fixed kernel. The old result was a *different bug* perturbing timing, not the
  lever.

## Lessons from worked examples

- **Benchmark the reference in the same process and hygiene, and read it standalone.** A reference timed in the same loop as your kernel can look ~5% faster because your kernel pre-warmed the L2 with its weights. Report the standalone reference as the fair number, and quote the paired one separately.
- **Re-measure the hot-kernel breakdown after any structural change.** An old attribution ("the reduce is 11% of time") can be stale by a large factor once a kernel changes. Re-profile before choosing the next lever.
- **Use bitwise parity for layout-only transforms.** Preshuffles, remaps and gating change where data lives, not the arithmetic. The outputs must match the baseline exactly (max diff 0); a tolerance-based check can hide a layout mismatch.

---
Related: [establish-baselines](establish-baselines.md) · [bottleneck-classification](bottleneck-classification.md) · [failure-modes](../30-failure-modes/failure-modes.md)
