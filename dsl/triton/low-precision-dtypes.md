---
name: low-precision-dtypes
description: "fp8 and MXFP4 in AMD Triton: FNUZ vs OCP fp8 per arch (tl.float8e4b8 vs tl.float8e4nv), scale handling, and tl.dot_scaled on gfx950 with E8M0 group-32 scales and scale preshuffling."
source: https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py ; https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py ; https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html
---

# Triton on AMD: fp8 and MXFP4

## fp8: two encodings, one per arch

```
            sign exp mant  bias   max    NaN/inf            Triton type      torch dtype
 OCP E4M3     1   4   3     7     448    NaN only           tl.float8e4nv    float8_e4m3fn
 FNUZ E4M3    1   4   3     8     240    one NaN (0x80)     tl.float8e4b8    float8_e4m3fnuz
 OCP E5M2     1   5   2    15   57344    inf + NaN          tl.float8e5      float8_e5m2
 FNUZ E5M2    1   5   2    16   57344    one NaN (0x80)     tl.float8e5b16   float8_e5m2fnuz
```

| Arch | Native MFMA fp8 | The other one |
|---|---|---|
| gfx942 (CDNA3) | FNUZ (`e4b8`, `e5b16`) | OCP is software-emulated or rejected by `tl.dot` |
| gfx950 (CDNA4) | OCP (`e4nv`, `e5`) | FNUZ emulated |

The backend comment in `HIPOptions.supported_fp8_dtypes` states this split.

### Traps

- **Same bits, 2× off.** The same byte decodes to values a factor of 2 apart (bias differs by 1).
  Read an OCP checkpoint as FNUZ and every value is half. No error, just bad output.
- **Fix at load time.** Serving stacks convert OCP E4M3 weights to FNUZ once on gfx942 and double
  the scale (see SGLang/vLLM `normalize_e4m3fn_to_e4m3fnuz`). The bit `0x80` is −0 in OCP and NaN in
  FNUZ; set it to 0 during the conversion.
- **The output dtype picks the clamp.** A quant kernel that takes `FP8_MAX = torch.finfo(out.dtype).max`
  gets 240 or 448 from the buffer you pass. Allocate the right dtype for the arch.
- **Saturate before cast.** Clamp to ±max before converting. Overflow can become NaN.

### Scale placement (per-tensor / per-token / per-block)

Fold scales outside the MFMA: `acc = tl.dot(a_fp8, b_fp8)` in fp32, then multiply by
`a_scale[:, None] * b_scale[None, :]`. For block scales along K, scale each K-block partial
product before you add it to the accumulator.

## MXFP4 / MXFP8 with `tl.dot_scaled` (gfx950)

`tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc)` lowers to
`v_mfma_scale_f32_16x16x128_f8f6f4` or `..._32x32x64_f8f6f4` on CDNA4.

- OCP MX only: one **E8M0** scale byte (power of 2, bias 127) per **32** K-elements.
  The group size is fixed by the format. Do not tune it.
- FP4 E2M1 packs 2 values per byte. K in elements = 2 × K in bytes.
- `tl.dot_scaled` takes scales in the logical 2D layout `(M, K/32)` and `(N, K/32)`.

### Why scale loads hurt, and the preshuffle fix

In a 16x16x128 scaled MFMA, 4 lanes share the K axis and 16 lanes the M/N axis.
Each lane needs one scale per MFMA, strided in memory. Loads become narrow byte loads
(`ds_read_u8`, `global_load_ubyte`) with extra waits.

```
 plain layout  : lane L needs scale[m, k0], scale[m, k0+4], ...   -> strided bytes
 preshuffled   : lane L's 4 scales (for 4 MFMAs) sit next to each other -> one 32-bit load
```

The upstream tutorial stores scales preshuffled in global memory (minimum tile 32x32x256),
then reshapes/permutes them inside the kernel back to the 2D logical layout `tl.dot_scaled` expects.
The compiler sees contiguous per-lane data and emits wide loads.

Rule: match the scale layout to what *this* kernel unshuffles. A scale tensor shuffled for a different
kernel (or for the raw hardware layout) gives wrong results with no error.

### Check the cost

Compare the ISA of the scaled kernel with a plain fp8 `tl.dot` of the same tile. Count
`ds_read_u8`, `s_waitcnt lgkmcnt`, and scratch. If scale traffic dominates and preshuffling does not
fix it, a hand-written kernel may be needed. See [scope-and-ceiling](scope-and-ceiling.md).

## Accuracy gates

- fp8 E4M3 has 3 mantissa bits: about 2⁻⁴ relative step. Compare against an fp32 reference with a
  tolerance sized to that, not to bf16.
- Gate quantized paths on error ratio or task accuracy, not bit-equality.

Verify specifics against the primary source.

Sources: [compiler.py supported_fp8_dtypes](https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py) ·
[Triton tutorial 10: block-scaled matmul (CDNA4 scale preshuffle)](https://github.com/triton-lang/triton/blob/main/python/tutorials/10-block-scaled-matmul.py) ·
[HIP low-precision float types (FNUZ vs OCP)](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html) ·
[OCP Microscaling Formats spec v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ·
[SGLang PR #2601 (FNUZ normalization)](https://github.com/sgl-project/sglang/pull/2601) ·
[AMD CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)

---
Related: [codegen-pipeline](codegen-pipeline.md) · [operator-recipes](operator-recipes.md) · [gfx950](../../hardware/amd-cdna/gfx950.md)
