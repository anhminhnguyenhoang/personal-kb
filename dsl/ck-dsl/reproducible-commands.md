---
name: reproducible-commands
description: "Exact venv / PYTHONPATH / harness invocations: the single validation block, DSL probe quick-shots, rocprof attach, and LDS/occupancy follow-up tools (runbook S19)."
source: CK DSL Optimization Runbook (ROCm/rocm-libraries@1efeadd, projects/composablekernel/python/ck_dsl/dsl_docs/optimization/optimization_runbook.md)
---

## 19. Reproducible Commands

All commands assume `python` is the interpreter from the team's
canonical environment (CPython 3.12 with `torch + rocm`, `aiter`
importable, GPU visible). `PYTHONPATH` is set so `ck_dsl` is
importable — the conventional layout is to set it to the
`composablekernel/python` directory.

### 19.1 PYTHONPATH bootstrap

From the `composablekernel` checkout root:

```bash
export PYTHONPATH=python
```

### 19.2 The single validation block

```bash
cd <composablekernel-checkout>
export PYTHONPATH=python

PYTHONDONTWRITEBYTECODE=1 python python/test/test_ck_dsl.py
PYTHONDONTWRITEBYTECODE=1 python python/test/test_ck_dsl_examples.py

OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
python -m ck_dsl.examples.bake_off_implicit_gemm --output-dir "$OUT_DIR"
python -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify

python python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096
python python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128
python python/ck_dsl/examples/ck_tile_parity.py --op all

export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" python \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --scenario decode_d128_b16 --attempts 1 --warmup 0 --paths auto,2d,3d
```

### 19.3 DSL probe quick-shots

```bash
cd <composablekernel-checkout>/python
PROBES=ck_dsl/dsl_docs/optimization/utilities/tools/dsl_probes

python "$PROBES/probe_occupancy.py"        --demo attention_tiled_2d --arch gfx950
python "$PROBES/probe_intrinsic_counts.py" --demo attention_tiled_2d
python "$PROBES/probe_isa_inspect.py"      --demo attention_tiled_2d --mcpu gfx950
python "$PROBES/probe_lowering_compare.py" --demo attention_tiled_2d --arch gfx950
python "$PROBES/probe_config_sweep.py"     --demo attention_tiled_2d
python "$PROBES/probe_targeted_bench.py"   --dry-run
```

### 19.4 rocprof attach (matched to a single kernel)

```bash
sudo rocm-smi --setperflevel high
sudo rocm-smi --setsclk 7

cat > metrics.txt <<'EOF'
pmc: SQ_WAVES
pmc: SQ_BUSY_CU_CYCLES
pmc: SQ_INSTS_MFMA
pmc: SQ_ACTIVE_INST_MFMA
pmc: TCC_EA_RDREQ_64B
pmc: TCC_EA_WRREQ_64B
pmc: TCC_HIT
pmc: TCC_MISS
pmc: SQ_WAIT_INST_LDS
pmc: TCP_PENDING_STALL_CYCLES
pmc: LDS_BANK_CONFLICT
EOF

rocprofv3 -i metrics.txt -o run.csv --stats --kernel-trace -- \
  python ck_dsl/dsl_docs/optimization/utilities/tools/dsl_probes/probe_rocprof_single.py \
  --builder mypkg.mymod:make_runner \
  --problem-json /tmp/problem.json \
  --iters 50 --warmup 10
```

### 19.5 LDS / occupancy follow-up tools

```bash
python ck_dsl/dsl_docs/optimization/utilities/tools/stage4_analyze/analyze_lds_conflicts.py \
  --rocprof run_kernel_stats.csv --isa kernel.s --arch gfx950

python ck_dsl/dsl_docs/optimization/utilities/tools/stage5_compare/compare_rocprof_stats.py
```

---
Diagnostic how-to: [dsl-probe-workflow](dsl-probe-workflow.md) · benchmark hygiene in [establish-baselines](../../kernel-optimization/00-method/establish-baselines.md). Paths drift — verify against current source before relying on them.
