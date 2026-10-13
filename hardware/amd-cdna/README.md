# AMD CDNA hardware

Hardware facts for CDNA3 (gfx942, MI300X/MI325X) and CDNA4 (gfx950, MI350X/MI355X), in reading order.

- [cu-simd-wave-vector-alu](cu-simd-wave-vector-alu.md) — Why CDNA SIMDs are 16 lanes wide; CU, SIMD, wave64, and quad-cycle issue foundations.
- [threadblocks-and-registers](threadblocks-and-registers.md) — Threadblock equals workgroup on AMD; register-budget derivation of why 8-wave ping-pong holds a 256x256 tile.
- [gfx942-registers-occupancy](gfx942-registers-occupancy.md) — gfx942 VGPR/AGPR/SGPR files, the flexible 256+256 split, and wave occupancy arithmetic.
- [cdna3-cdna4-deltas](cdna3-cdna4-deltas.md) — CDNA3 vs CDNA4 side by side: per-CU peak rates, clocks, power, sustained clock, partitions, porting deltas.
- [memory-ladder-xcd-hbm](memory-ladder-xcd-hbm.md) — Registers to HBM ladder, roofline ridge per GPU and dtype, XCD locality, 512 B Tagram cliff, Infinity Fabric.
- [lds-model-and-banks](lds-model-and-banks.md) — 32 vs 64 LDS banks, conflict math for padding and XOR, LDS bandwidth, direct global-to-LDS contract, LDS occupancy.
- [matrix-core-generations](matrix-core-generations.md) — MFMA/SMFMAC/scaled-MFMA capability matrix CDNA1–4, cycle counts, peak formula, why legacy shapes hit half CDNA4 peak.
- [low-precision-formats](low-precision-formats.md) — FP8 FNUZ vs OCP at bit level, FP6/FP4 ranges, MX E8M0 block scales, scaled converts, MFMA rounding per generation.
- [gfx950](gfx950.md) — gfx950 reference: MFMA atom catalog, LDS conflict periods, cross-lane ops, occupancy caps, buffer descriptor, compiler caveats.
- [mi350-family](mi350-family.md) — MI350P, MI350X, and MI355X specifications, form factors, and deployment guidance.
