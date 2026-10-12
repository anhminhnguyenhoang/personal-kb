---
name: epilogue
description: epilogue ops (bias/act/residual/scale/quant/cast), direct vs LDS/CShuffle epilogue trade-off, the 'in-between naive LDS epilogue loses both ways' caveat, and output validation. Vectorizing the epilogue is often the single largest win.
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 9. Epilogue Optimization

### 9.1 Common Epilogues

- Bias add.
- Activation (`ReLU`, `SiLU`, `GELU`, `tanh`, `sigmoid`), as a chain of
  fused element-wise ops.
- Residual add.
- Scaling.
- Quantization / dequantization.
- Type conversion (vector truncate f32→f16, `cvt_pk_f32_fp8`).
- Row / column scale (block scale for MX).
- Clamp.
- Dropout (paired f32 RNG mask).
- Store transpose (transposed store through LDS, with a matching LDS
  reader formula).

### 9.2 Direct Epilogue

Direct stores, with the vectorization either done in the accumulator
or after the accumulator is read out.

- Best when accumulator lane layout already matches the global store
  layout.
- Convert vector accumulator to vector output.
- Use packed vector stores (buffer or global, f16, N dwords).
- Avoid scalar stores.
- Avoid unnecessary LDS.

### 9.3 LDS / Shuffle Epilogue

Stage the tile through LDS so the global store is contiguous (a
"CShuffle" epilogue).

- Use when accumulator lane layout is bad for global stores.
- Write per-lane accumulators to LDS.
- Barrier.
- Have a subset of threads read contiguous vectors from LDS.
- Store wide vectors globally.
- Ensure LDS write and read mappings are exact.
- Ensure only active store threads write.
- Include OOB checks for Q/N/K tails.
- Measure whether coalescing gain beats barrier/LDS cost.

"Use LDS for the epilogue" is not the same as "match the library's
LDS epilogue distribution". A naive LDS-staged epilogue with a flat
`[q, group, k]` layout was correct for one shape but wrong for
another and never beat direct vector stores in our experiments. CK's
LDS epilogue uses the MFMA distribution for the LDS write, then a
separate wide-store distribution where only `STORE_VECS = BLOCK_Q ×
BLOCK_C8` threads issue 16-byte global stores. Either copy that exact
mapping or stay with direct vector stores; the in-between case loses
both ways.

A related caveat applies to any structural change that removes one
cost: **the cost may simply move**, not disappear. In a unified-attention
worked example, removing a per-iter LDS-store stripe replaced it
with a much larger cross-lane permute storm — a material regression
on the same shape. Always confirm via the per-iter ISA histogram
(the ISA-inspection probe or the intrinsic-count probe) that the
savings outnumber the introduced overhead before declaring the
change a win.

### 9.4 Output Validation

- Epilogue bugs often affect only certain groups / channels /
  columns.
- Validate all channels, all groups, and both boundary and interior
  Q.
- Test `groups` values that change block count.
- Test both K tails and Q tails.

ck_dsl specifics: [primitive-map](../../dsl/ck-dsl/primitive-map.md#epilogue).

---
Epilogue knobs: [knob-catalog-and-sweep](../../dsl/ck-dsl/knob-catalog-and-sweep.md). ISA store-width check: [isa-resource-inspection](isa-resource-inspection.md).
