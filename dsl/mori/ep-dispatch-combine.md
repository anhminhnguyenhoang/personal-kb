---
name: ep-dispatch-combine
description: "MoRI-EP expert-parallel dispatch/combine: what the two all-to-alls do, the symmetric heap, kernel types by topology, config fields, combine buffer modes, wire codecs, 2D vs DeepEP 3D layouts, and how aiter calls it."
source: https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md ; https://github.com/ROCm/mori/blob/main/python/mori/ops/dispatch_combine.py ; https://github.com/ROCm/aiter/blob/main/aiter/dist/device_communicators/all2all.py
---

# MoRI-EP dispatch and combine

MoRI is AMD's GPU-initiated communication library. Its EP part (`mori.ops`) provides
the two all-to-alls of expert-parallel MoE. GPU kernels write directly into peer memory
(xGMI inside a node, RDMA across nodes); the host does not drive each transfer, so the
ops can be captured in a HIP graph.

```
 rank r: tokens [T, H], topk_ids [T, k], topk_weights [T, k]
     │ dispatch: send each token (once per destination rank) to the ranks that own its experts
     ▼
 rank r': received tokens for its local experts ─► local sort + grouped GEMM (e.g. aiter fused_moe)
     │ combine: send each expert output back to its source rank, weight it, sum over k
     ▼
 rank r: output [T, H]
```

## Building blocks

- **Symmetric heap (`mori.shmem`).** Every rank allocates the same-sized buffers, so a
  peer address is a known offset. Init once per process group
  (`shmem_torch_process_group_init`). Set `MORI_SHMEM_HEAP_SIZE` before init.
- **`EpDispatchCombineConfig`** holds the shape caps and launch geometry;
  **`EpDispatchCombineOp`** owns the buffers and exposes `dispatch`, `combine`, `reset`.
- Call `reset()` between iterations, or pass `call_reset=True` to `combine()`.

## Kernel type by topology

```
EP inside one node (xGMI only)?
├─ yes → IntraNode
└─ no  ─┬─ throughput (large batch) → InterNodeV1
        ├─ latency (small batch)    → InterNodeV1LL  or  AsyncLL (split send/recv phases)
        └─ baseline / debug         → InterNode
```

## Config fields that matter

| Field | Role |
|---|---|
| `hidden_dim`, `num_experts_per_rank`, `num_experts_per_token` (top-k) | Problem shape |
| `max_num_inp_token_per_rank` | Sizes the buffers; a cap, not the live count |
| `scale_dim`, `scale_type_size` | Per-token quant scales carried with the tokens (0 if none) |
| `kernel_type`, `gpu_per_node` | Transport and rank mapping |
| `block_num`, `warp_num_per_block`, `rdma_block_num`, `num_qp_per_pe` | Launch geometry: the tuning knobs ([launch-tuning](launch-tuning.md)) |
| `use_external_inp_buf` | Combine buffer mode (below) |
| `quant_type` | Wire codec |

Launch geometry can also be passed per call (`block_num=`, `warp_per_block=`; `-1`
means "use the config"). In `AUTO` mode those per-call values are ignored.

## Combine buffer modes

| Mode | How | Cost |
|---|---|---|
| External (default) | Pass any tensor to `combine()` | MoRI first copies it into its registered peer-visible buffer |
| Zero-copy | Get `get_registered_combine_input_buffer(dtype)`, have the expert GEMM epilogue write there, call `combine(..., use_external_inp_buf=0)` | No staging copy |

The two modes are different kernels with different best launch geometry. In MoRI's own
tuned tables they are separate keys.

## Wire codecs

- Dispatch can move tokens in a narrow dtype (for example fp8 or fp4 tokens with scales;
  `fp8_direct_cast` casts on the wire).
- Combine can use block-wise codecs (`fp8_blockwise`; `fp4_blockwise` only on gfx950).
- Bytes per token round trip set the cost. With hidden 7168, bf16 both ways is
  28,672 B/token; an fp4 dispatch with fp8 block-wise combine was reported at about
  11,200 B/token. Verify specifics against the primary source.

## Layouts

- Native: flat 2D `[tokens, H]` with indices.
- DeepEP-compatible 3D `[local_experts, capacity, H]`: `dispatch_standard_moe` /
  `combine_standard_moe`. Needs the build flag `ENABLE_STANDARD_MOE_ADAPT=ON` (off by
  default); without it these calls raise.

## How aiter calls MoRI

aiter's `MoriAll2AllManager` (`aiter/dist/device_communicators/all2all.py`) builds the
config with **fixed** launch geometry chosen by topology (intra-node, inter-node, or
low-latency), not from MoRI's tuned tables. A FlyDSL-kernel variant (`FlyDSLAll2AllManager`)
reuses MoRI's symmetric heap but replaces the comm kernels; it is intra-node only.

## Contracts and traps

- Combine multiplies by the routing weight and must **mask padded tokens** when shapes
  are padded to static sizes for graph capture.
- Each expert stays whole on one rank under EP, so the GEMM math is the same as on one
  GPU. Tensor parallelism instead splits the down-projection reduction across ranks.
- Expert load is uneven. The busiest rank sets the step time, so benchmark the
  bottleneck rank, not the mean.
- JIT build happens on first use. Warm up (or precompile) before timing.

Sources: [MORI-EP guide](https://github.com/ROCm/mori/blob/main/docs/MORI-EP-GUIDE.md) ·
[`dispatch_combine.py`](https://github.com/ROCm/mori/blob/main/python/mori/ops/dispatch_combine.py) ·
[aiter `all2all.py`](https://github.com/ROCm/aiter/blob/main/aiter/dist/device_communicators/all2all.py) ·
[Wide-EP DeepSeek on MI300X](https://github.com/ROCm/rocm-blogs/blob/release/blogs/software-tools-optimization/wide-ep-deepseek/README.md) ·
[MoRI on MI355X (LMSYS)](https://www.lmsys.org/blog/2026-05-28-mori/)
Paths drift; verify before use.

---
Related: [launch-tuning](launch-tuning.md) · [aiter operator-catalog](../aiter/operator-catalog.md)
