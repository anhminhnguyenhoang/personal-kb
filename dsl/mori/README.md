# MoRI (expert-parallel communication)

MoRI is AMD's GPU-initiated communication library for Instinct GPUs. It is built on a
symmetric-memory heap. These notes cover only **MoRI-EP**, the expert-parallel MoE
dispatch/combine ops (`mori.ops.EpDispatchCombineOp`). The repo also has storage
transfer, collectives, and other parts, which are not covered here.

Upstream: [ROCm/mori](https://github.com/ROCm/mori), [MORI-EP guide](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md). Paths and symbols drift; verify before use.

## Notes

- [ep-dispatch-combine](ep-dispatch-combine.md) — What dispatch/combine do, kernel types by topology, config fields, combine buffer modes, wire codecs, layouts, and the aiter seam.
- [launch-tuning](launch-tuning.md) — MANUAL vs AUTO, the JSON rule tables and keys, the two-phase tuner, per-phase geometry rules, and comms measurement hygiene.

Related: the single-GPU MoE pipeline around these ops is in [aiter operator-catalog](../aiter/operator-catalog.md).
