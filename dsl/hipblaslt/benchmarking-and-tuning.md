---
name: benchmarking-and-tuning
description: The hipBLASLt workflow from capturing an app's GEMMs to replaying tuned kernels in production (LOG_MASK=32, hipblaslt-bench flags, tuning file, override and cache modes, Stream-K env vars).
source: https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt (docs/reference/env-variables.rst, docs/how-to/how-to-use-hipblaslt-offline-tuning.rst, docs/how-to/how-to-use-streamk.rst)
---

# hipBLASLt benchmarking and tuning workflow

## The loop

```
1 capture      HIPBLASLT_LOG_MASK=32  ./app        → one hipblaslt-bench command line per GEMM call
2 dedupe       sort -u the commands (an LLM has only a few distinct shapes)
3 tune         HIPBLASLT_TUNING_FILE=t.txt  hipblaslt-bench <cmd>   → best solution index per shape
4 replay       HIPBLASLT_TUNING_OVERRIDE_FILE=t.txt ./app            (or TUNING_MODE=cache, below)
5 verify       hipblaslt-bench <cmd> --print_kernel_info  → shows the tuned solution index
```

- With mask 32 (bench), each API call logs a ready-to-run `hipblaslt-bench` command. This is the standard way to reproduce a framework's GEMMs outside the framework. `HIPBLASLT_LOG_FILE=path_%i` writes one file per PID.
- With `HIPBLASLT_TUNING_FILE` set, the bench switches to tuning defaults: about 1000 iterations and 1000 cold iterations, a 512 MB rotating buffer, and every solution. `HIPBLASLT_TUNING_USER_MAX_WORKSPACE` limits the workspace.
- **Override file.** The tuned solution goes to position 0 of the heuristic results. Matching is **exact** on trans, M/N/K/batch, and the main dtypes. The file is read once per process.
- **Runtime cache** (newer releases): `HIPBLASLT_TUNING_MODE=cache` with `HIPBLASLT_TUNING_CACHE_PATH=t.txt`. It is read-only, so many ranks can share one file. Unlike the override file, it also applies to `hipblasLtMatmul(algo=nullptr)` calls. It drops entries that no longer validate after a rebuild, and those shapes fall back to the heuristic.
- **Tuning results do not move between library builds or archs.** Solution indices change. Tune again after every hipBLASLt upgrade.

## hipblaslt-bench essentials

| Need | Flags |
| --- | --- |
| shape / layout | `-m -n -k --transA T --transB N --batch_count --lda ...` |
| types | `-r bf16_r` or `--a_type f8_r --b_type f8_r --c_type bf16_r --d_type bf16_r --compute_type f32_r` |
| scales | `--scaleA/--scaleB` (1 scalar, 2 vector, 3 MX block-32 UE8M0, …), `--amaxD` |
| epilogue | `--activation_type gelu --bias_vector --bias_type bf16_r --use_e --gradient` |
| algos | `--algo_method` (`heuristic`, `all`, `index`), `--requested_solution N` (-1 = all), `--solution_index I`, `--print_kernel_info` |
| per-call tuning | `--splitk S --wgm W` (needs `--api_method cpp` or `mix`) |
| grouped GEMM | repeat `-m/-n/-k` once per group + `--grouped_gemm --api_method cpp` |
| timing | `-i`, `-j` (cold iters), `--rotating MB`, `--use_gpu_timer`, `--flush` |
| correctness | `-v` (CPU reference; prints norm error) |

Measurement rules (the general method is in [measurement](../../kernel-optimization/method/measurement.md)):
- Use a rotating buffer larger than L2 and the MALL/LLC (512 MB is safe) to get HBM-realistic numbers. Without it, small and medium shapes report L2-hot bandwidth.
- Use `--use_gpu_timer` for tiny kernels. The host timer adds launch overhead.
- On gfx942, the bench quietly converts OCP `f8_r` to `f8_fnuz_r`. Library calls from your own code do not do that conversion.

## Extra diagnostics

| Env | Adds |
| --- | --- |
| `HIPBLASLT_BENCH_FREQ=1` / `_FREQ_ALL=1` | the lowest / average SCLK per XCD, and MCLK, during the run. Use it to catch clock throttling. |
| `HIPBLASLT_BENCH_PERF=1` | CU count, tiles per CU, tile/CU/wave granularity, bytes moved, efficiency |
| `HIPBLASLT_ENABLE_MARKER=1` | ROCProfiler (roctx) ranges around calls |
| `TENSILE_DB` (bits 0x2 and 0x4) | log the solution tag at selection |

## Stream-K / persistent controls

- `TENSILE_SOLUTION_SELECTION_METHOD`: 0 = standard tuned libraries, 2 = Origami + Stream-K on MI300. On MI350, Origami + Stream-K is the only library, and the variable does nothing.
- `TENSILE_PERSISTENT_DYNAMIC_GRID` (default 6 = auto), `_FIXED_GRID`, `_MAX_CUS`, `_GRID_MULTIPLIER`, `_DYNAMIC_WGM` control the grid size of persistent kernels. Older releases named these `TENSILE_STREAMK_*`, and those names still work.
- `--splitk` and `--wgm` let you try GSU and mapping changes without rebuilding.

## Other tools

`hipblaslt-sequence` runs a YAML list of GEMM layers with fixed algo indices and flushes in between. Use it to imitate a transformer layer. `rocblas-bench -f gemm` gives a rocBLAS baseline with similar flags. The `install.sh` flag `-n` builds the clients only, and `--logic-yaml-filter` limits the build to some logic files. Both speed up rebuilds a lot.

Sources: [hipBLASLt env variables](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html), [offline tuning how-to](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html), [Stream-K how-to](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-streamk.html), [hipBLASLt clients](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipblaslt/clients).
Paths drift; verify before use.

---
Related: [api-concepts](api-concepts.md) · [tensilelite-kernels-and-selection](tensilelite-kernels-and-selection.md) · [pitfalls](pitfalls.md)
