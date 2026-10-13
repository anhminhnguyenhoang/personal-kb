# AMDGCN assembly (hand-written gfx942 / gfx950 kernels)

Notes for writing, reading and fixing raw CDNA3/CDNA4 assembly: the layer under every DSL.
The DSL-neutral method lives in [`../../kernel-optimization/`](../../kernel-optimization/README.md),
chip facts in [`../../hardware/amd-cdna/`](../../hardware/README.md).

Reading order:

- [kernel-abi-and-prologue](kernel-abi-and-prologue.md) — Kernel descriptor, initial SGPR/VGPR state, kernarg load or preload, wave-id prologue, building and launching a `.co`.
- [registers-agpr-liveness](registers-agpr-liveness.md) — Unified VGPR/AGPR file, `accum_offset` aliasing, async-load write timing, clobber bug classes, liveness method to free registers.
- [waitcnt-and-hazards](waitcnt-and-hazards.md) — vmcnt/lgkmcnt ordering and traps, barriers and LDS visibility, required wait states for MFMA, DPP, M0, trans ops; inline-asm rules.
- [memory-instruction-choice](memory-instruction-choice.md) — Buffer vs global vs flat vs scalar vs direct-to-LDS: address forms, bounds behavior, counters, register cost, when to use each.
- [memory-coherence-bits](memory-coherence-bits.md) — sc0/sc1/nt scopes, atomics, release/acquire sequences, and a correct cross-workgroup spin barrier.
- [lds-patterns-asm](lds-patterns-asm.md) — DS offset fields, transpose reads, ping-pong toggles, barrier counts, per-wave regions, cross-wave reductions.
- [mfma-in-asm](mfma-in-asm.md) — MFMA operands, lane layout formulas, cbsz/abid/blgp per family, scaled MFMA, forced RNE, FNUZ vs OCP fp8.
- [debugging-playbook](debugging-playbook.md) — Symptom table, sentinel/all-ones/identity tests, shape pyramid, binary round-trip checks, in-kernel timers.
- [aiter-asm-kernel-patterns](aiter-asm-kernel-patterns.md) — What public aiter asm kernels (paged attention, MoE, flatmm, top-k, allreduce) do, and how aiter ships and selects them.
