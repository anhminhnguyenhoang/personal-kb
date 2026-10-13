---
name: cdna3-cdna4-deltas
description: CDNA3 (MI300X/MI325X, gfx942) vs CDNA4 (MI350X/MI355X, gfx950) side by side — per-CU peak rates, clocks and power, sustained-clock reality, and the porting deltas that break kernels.
source: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ; AMD CDNA3 and CDNA4 ISA reference guides ; arXiv 2510.27583
---

# CDNA3 → CDNA4: specs, peaks, and porting deltas

## 1. The package, side by side

```
            CDNA3  MI300X / MI325X                 CDNA4  MI350X / MI355X
   ┌──────────────────────────────────┐   ┌──────────────────────────────────┐
   │ 4 IODs (N6)                      │   │ 2 IODs (N6), faster IOD link     │
   │ 8 XCDs (N5) × 38 active CUs      │   │ 8 XCDs (N3P) × 32 active CUs     │
   │       = 304 CUs, 1216 matrix cores│   │       = 256 CUs, 1024 matrix cores│
   │ L2 4 MB / XCD   IC 256 MB device │   │ L2 4 MB / XCD   IC 256 MB device │
   │ LDS 64 KB / CU                   │   │ LDS 160 KB / CU                  │
   │ HBM3 192 GB / HBM3E 256 GB       │   │ HBM3E 288 GB                     │
   └──────────────────────────────────┘   └──────────────────────────────────┘
   fewer CUs on CDNA4, but each CU does 2× the ≤16-bit matrix math
```

| Item | MI300X | MI325X | MI350X | MI355X |
|---|---|---|---|---|
| ISA | gfx942 | gfx942 | gfx950 | gfx950 |
| CUs per XCD (built / active) | 40 / 38 | 40 / 38 | 36 / 32 | 36 / 32 |
| Active CUs | 304 | 304 | 256 | 256 |
| Max engine clock | 2100 MHz | 2100 MHz | 2200 MHz | 2400 MHz |
| Max board power | 750 W | 1000 W | 1000 W | 1400 W |
| HBM | 192 GB, 5.3 TB/s | 256 GB, 6.0 TB/s | 288 GB, 8.0 TB/s | 288 GB, 8.0 TB/s |
| Transistors | 153 B | 153 B | 185 B | 185 B |

MI300X and MI325X share compute; only memory and power differ.
MI350X and MI355X share silicon; only clock and power differ.
Form factors and the PCIe MI350P: [mi350-family](mi350-family.md).

## 2. Peak = per-CU rate × CUs × clock

```
peak_FLOPs = FLOPs_per_clock_per_CU × active_CUs × clock_Hz
```

| Operation | CDNA3 FLOPs/clk/CU | CDNA4 FLOPs/clk/CU | MI300X peak | MI355X peak |
|---|---|---|---|---|
| FP64 vector | 128 | 128 | 81.7 TF | 78.6 TF |
| FP64 matrix | 256 | **128** | 163.4 TF | 78.6 TF |
| FP32 vector / matrix | 256 | 256 | 163.4 TF | 157.3 TF |
| TF32 matrix | 1024 | **none** | 653.7 TF | — (software via BF16) |
| FP16 / BF16 matrix | 2048 | **4096** | 1.3 PF | 2.5 PF |
| FP8 / INT8 matrix | 4096 | **8192** | 2.6 PF | 5.0 PF |
| FP6 / FP4 matrix (incl. MX) | — | **16384** | — | 10 PF |
| 2:4 sparse (SMFMAC) | 2× dense | 2× dense | — | — |

Check: `2048 × 304 × 2.1e9 = 1.307 PF` (MI300X FP16). `4096 × 256 × 2.4e9 = 2.52 PF` (MI355X FP16).
The 2× per CU only comes from the new double-K instructions:
[matrix-core-generations](matrix-core-generations.md).

## 3. Peak clock is not the clock you get

```
   boost 2100 MHz ──────────────────────────────  peak tables use this
                                                   (dense GEMM, MI300X)
   measured  ~1.1–1.2 GHz ━━━━━━━━━━━━━━━━         power-limited
             └─ ~45% of peak TFLOPs end to end
             └─ ~80–85% of the clock-adjusted peak = software efficiency
```

- A dense MFMA loop is power-bound, not clock-bound. Clock drops far below boost.
- Input values change power, so they change speed. All-zero inputs run >20% faster than
  random floats. Benchmark with realistic data, never zeros or `torch.ones`.
- XCD clocks differ by 3–10% (on MI300X XCD0 is fastest, XCD7 slowest). For efficiency math, use
  the slowest XCD. A device-wide sync runs at the slowest XCD's pace.
- HBM bandwidth is set by the memory clock. A bandwidth-bound kernel gains nothing from engine clock.
- Short kernels may finish before the clock ramps. Warm up; take the median of repeats.

## 4. Porting deltas that break CDNA3 kernels on CDNA4

| Delta | CDNA3 | CDNA4 | What breaks |
|---|---|---|---|
| FP8 encoding | FNUZ | OCP | Bit-copied FP8 data decodes to wrong values → [low-precision-formats](low-precision-formats.md) |
| TF32 | hardware | removed | TF32 code paths must use BF16 or FP32 |
| LDS | 64 KB, 32 banks | 160 KB, 64 banks | 32-bank swizzles may conflict → [lds-model-and-banks](lds-model-and-banks.md) |
| Direct global→LDS | ≤ 4 B/lane | adds 12 and 16 B/lane | Old 4 B staging leaves width unused |
| Matrix shapes | K=16 (fp16), K=32 (fp8) | adds K=32 (fp16), K=128 (f8f6f4) | Old shapes run at half the new peak |
| FP6 / FP4 / MX scale | none | yes | — (new capability) |
| LDS transpose reads | none | `ds_read_b64_tr_b{4,8,16}`, `ds_read_b96_tr_b6` | — (new capability) |
| CUs | 304 | 256 | Grid sizes tuned to 304 CUs leave tails |
| FP64 matrix | 163 TF | 79 TF | HPC kernels lose half the matrix FP64 rate |

## 5. Partition modes (compute × memory)

| Mode | MI300X | MI355X |
|---|---|---|
| SPX | 8 XCDs, 192 GB, NPS1 | 8 XCDs, 288 GB, NPS1 |
| DPX | 4 XCDs, 96 GB, NPS1 | 4 XCDs, 144 GB, NPS2 |
| QPX | 2 XCDs, 48 GB, NPS1/4 | 2 XCDs, 72 GB, NPS2 |
| CPX | 1 XCD, 24 GB, NPS1 | 1 XCD, 36 GB, NPS2 |
| AMD-listed most efficient | QPX + NPS4 | DPX + NPS2 |

A smaller partition makes L2 and memory XCD-local. It also shrinks the device the kernel sees
(CU count, HBM). Re-tune grid size per mode.

Verify specifics against the primary source.

Sources:
- [ROCm: MI300 Series / MI350 Series workload optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html) (spec, peak, partition, XCD clock, input-value tables)
- [AMD CDNA3 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA4 ISA reference](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMD CDNA 4 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [arXiv 2510.27583 — AMD MI300X GPU Performance Analysis](https://arxiv.org/abs/2510.27583) (sustained-of-peak, measured clock)

---
Related: [mi350-family](mi350-family.md) · [gfx950](gfx950.md) · [gfx942-registers-occupancy](gfx942-registers-occupancy.md) · [memory-ladder-xcd-hbm](memory-ladder-xcd-hbm.md) · [matrix-core-generations](matrix-core-generations.md)
