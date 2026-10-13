# Hardware

Hardware facts that kernel design depends on, for AMD CDNA and NVIDIA Blackwell.

## AMD CDNA (amd-cdna/)

Folder index: [amd-cdna/README.md](amd-cdna/README.md).

- [cu-simd-wave-vector-alu](amd-cdna/cu-simd-wave-vector-alu.md) — What a vector ALU is and why CDNA picks 16 lanes; CU, SIMD, wave, and quad-cycle foundations.
- [threadblocks-and-registers](amd-cdna/threadblocks-and-registers.md) — Threadblock equals workgroup on AMD, and the register-budget derivation of why 8-wave ping-pong holds a 256x256 tile.
- [gfx942-registers-occupancy](amd-cdna/gfx942-registers-occupancy.md) — gfx942 (CDNA3, MI300X) VGPR, AGPR, and SGPR files, the flexible 256+256 split, and wave occupancy arithmetic.
- [cdna3-cdna4-deltas](amd-cdna/cdna3-cdna4-deltas.md) — CDNA3 vs CDNA4 side by side: per-CU peak rates, clocks, power, sustained clock, partitions, and porting deltas.
- [memory-ladder-xcd-hbm](amd-cdna/memory-ladder-xcd-hbm.md) — Memory ladder, roofline ridge per GPU and dtype, XCD locality, the 512 B Tagram cliff, and Infinity Fabric.
- [lds-model-and-banks](amd-cdna/lds-model-and-banks.md) — 32 vs 64 LDS banks, padding and XOR math, LDS bandwidth, direct global-to-LDS contract, and LDS occupancy.
- [matrix-core-generations](amd-cdna/matrix-core-generations.md) — MFMA/SMFMAC/scaled-MFMA matrix CDNA1–4, cycle counts, peak formula, and why legacy shapes hit half CDNA4 peak.
- [low-precision-formats](amd-cdna/low-precision-formats.md) — FP8 FNUZ vs OCP at bit level, FP6/FP4 ranges, MX E8M0 block scales, scaled converts, MFMA rounding.
- [gfx950](amd-cdna/gfx950.md) — gfx950 (CDNA4, MI350X/MI355X) reference: MFMA atoms, LDS and bank conflicts, cross-lane ops, occupancy caps, chiplets, compiler caveats.
- [mi350-family](amd-cdna/mi350-family.md) — Specifications, form factors, and deployment guidance for AMD Instinct MI350P, MI350X, and MI355X.

## NVIDIA Blackwell (nvidia-blackwell/)

- [asymmetric-hardware-scaling](nvidia-blackwell/asymmetric-hardware-scaling.md) — Blackwell tensor cores scaled 2.25x but SFUs and SMEM bandwidth stayed flat, so softmax exp became the attention bottleneck.
- [blackwell-tmem-2cta](nvidia-blackwell/blackwell-tmem-2cta.md) — TCGEN05/UMMA async tensor cores, 256 KB Tensor Memory, and 2-CTA cluster MMA; why they allow deeper pipelines than Hopper.
