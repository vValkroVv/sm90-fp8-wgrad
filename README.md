# SM90 FP8 Wgrad

This repo is a small, compute-only benchmark for one MoE backward hotspot:
per-expert weight gradients on NVIDIA Hopper.

It compares three Wgrad paths:

- SonicMoE BF16
- DeepGEMM public FP8 Wgrad
- a custom CuTe/QuACK-style SM90 FP8 Wgrad kernel with BF16 output

The scope is deliberately narrow. Routing, FP8 quantization, scale generation,
padding, metadata construction, and tensor allocation are outside the timed
region. The timed work is only the two grouped Wgrad GEMMs used by a sparse MoE
MLP:

```text
down Wgrad:    grad_y.T       @ hidden
gate/up Wgrad: grad_gate_up.T @ x
```

That makes the numbers easy to audit. It also keeps the repo small enough that
someone can reproduce the kernel result without bringing up a full training
stack.

## why this exists

Long MoE sequences make grouped Wgrad expensive. The normal BF16 path is solid,
but it leaves performance on the table once the inputs are already available in
FP8 form. We wrote this benchmark to answer a specific question:

> If the backward pass already has grouped FP8 operands and per-block scales,
> how fast can Wgrad run on H100/H200, and does it pass the same DeepSeek
> `calc_diff` gate used by DeepGEMM?

The original Qwen3-30B-A3B 32k run was the first useful data point:

| tokens | Sonic BF16 ms | Custom CuTe FP8 BF16-out ms | speed vs Sonic | latency reduction | throughput improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32768 | 9.562768 | 5.861056 | 1.6316x | 38.7% | +63.2% |

The full matrix below is the publishable run. It covers Qwen3-30B-A3B,
DeepSeek-V3, MiniMax-M3, and Qwen3.5-397B-A17B shapes across balanced and
skewed routes, with EP proxies for `1`, `8`, and `16`.

## results

This section is meant to be replaced after a full run:

```bash
python scripts/render_matrix_results.py \
  --input "$OUT" \
  --output "$OUT/results_for_readme.md" \
  --readme README.md
```

The renderer replaces only the block between the markers below.

<!-- SM90_WGRAD_RESULTS_BEGIN -->

Full public matrix results are pending.

Paste or auto-render the final tables here after running:

```bash
OUT="$STORAGE_ROOT/artifacts/sm90_fp8_wgrad_matrix_full_$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_model_shape_matrix.sh \
  --models all \
  --all-eps \
  --route both \
  --tokens-list 4096,8192,16384,32768 \
  --warmup-iters 500 \
  --active-iters 5000 \
  --repeat 3 \
  --devices 1,2,4,5 \
  --output-dir "$OUT"
```

Expected result format:

| model | route | EP | 4k | 8k | 16k | 32k | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-30B-A3B | skewed | 1 | pending | pending | pending | pending | custom speed vs Sonic |
| DeepSeek-V3 | skewed | 1 | pending | pending | pending | pending | custom speed vs Sonic |

Conclusion slots:

- Custom CuTe FP8 correctness:
- Best speedup:
- Long-sequence behavior:
- DeepGEMM public baseline behavior:

<!-- SM90_WGRAD_RESULTS_END -->

## install

Use a CUDA 13 PyTorch environment with H100/H200 access. The benchmark expects
PyTorch FP8 support, SonicMoE, QuACK, DeepGEMM, and NVIDIA CuTe/CUTLASS DSL.

```bash
git clone https://github.com/<org>/sm90-fp8-wgrad.git
cd sm90-fp8-wgrad

source /path/to/cuda13-venv/bin/activate
python -m pip install -e . --no-deps
python -m pip install -r requirements-cu13.txt --no-deps

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export QUACK_CACHE_ENABLED=0
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/check_env.py
```

## correctness

The correctness gate is DeepGEMM's own numeric check:
`deep_gemm.testing.calc_diff`. The default gate is `1e-3`.

```bash
python scripts/check_correctness.py \
  --tokens 4096 \
  --route skewed \
  --seed 1234 \
  --deepseek-diff-gate 1e-3 \
  --output-dtype bf16 \
  --output-dir artifacts/correctness_4k

cat artifacts/correctness_4k/summary.md
```

For a quick import and shape smoke test:

```bash
python scripts/check_correctness.py \
  --tokens 256 \
  --experts 8 \
  --hidden 256 \
  --intermediate 128 \
  --route skewed \
  --output-dir artifacts/correctness_smoke
```

## single-shape benchmark

```bash
python scripts/benchmark_wgrad.py \
  --tokens 32768 \
  --route skewed \
  --seed 1234 \
  --warmup-iters 100 \
  --active-iters 1000 \
  --repeat 3 \
  --include-sonic \
  --include-deepgemm \
  --include-custom \
  --deepseek-diff-gate 1e-3 \
  --output-dir artifacts/wgrad_32k

cat artifacts/wgrad_32k/summary.md
```

## sequence sweep

```bash
python scripts/sweep_seq_lens.py \
  --tokens-list 4096,8192,16384,32768 \
  --route skewed \
  --seed 1234 \
  --warmup-iters 100 \
  --active-iters 1000 \
  --repeat 3 \
  --deepseek-diff-gate 1e-3 \
  --output-dir artifacts/seq_sweep

cat artifacts/seq_sweep/summary.md
```

## full model-shape matrix

This is the command used for the public result block. It runs Sonic/custom
together and DeepGEMM separately, so one broken baseline row does not hide the
custom kernel result.

```bash
export SM90_WGRAD_CALC_DIFF_CHUNK_ELEMENTS=1048576
export SM90_WGRAD_FINITE_CHECK_CHUNK_ELEMENTS=1048576

OUT="$STORAGE_ROOT/artifacts/sm90_fp8_wgrad_matrix_full_$(date -u +%Y%m%dT%H%M%SZ)"

bash scripts/run_model_shape_matrix.sh \
  --models all \
  --all-eps \
  --route both \
  --tokens-list 4096,8192,16384,32768 \
  --warmup-iters 500 \
  --active-iters 5000 \
  --repeat 3 \
  --devices 1,2,4,5 \
  --output-dir "$OUT"

python scripts/render_matrix_results.py \
  --input "$OUT" \
  --output "$OUT/results_for_readme.md" \
  --csv "$OUT/results_flat.csv"
```

If `run_model_shape_matrix.sh` returns nonzero because DeepGEMM failed a row,
the completed rows are still in `$OUT`. Run the renderer anyway.

To embed the generated result block into this README:

```bash
python scripts/render_matrix_results.py \
  --input "$OUT" \
  --output "$OUT/results_for_readme.md" \
  --csv "$OUT/results_flat.csv" \
  --readme README.md
```

To archive everything:

```bash
cd "$(dirname "$OUT")"
zip -r "$(basename "$OUT").zip" "$(basename "$OUT")"
```

## model shapes

| model key | public label | experts | top-k | hidden | intermediate |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_30b` | Qwen3-30B-A3B | 128 | 8 | 2048 | 768 |
| `deepseek_v3` | DeepSeek-V3 | 256 | 8 | 7168 | 2048 |
| `minimax_m3` | MiniMax-M3 | 128 | 4 | 6144 | 3072 |
| `qwen3_5_397b_a17b` | Qwen3.5-397B-A17B | 512 | 10 | 4096 | 1024 |

The EP proxy keeps the global token count fixed and reduces the local expert
count:

```text
local_experts = global_experts / EP
local_tokens  = global_tokens
```

With fewer local experts, each local expert receives more rows. That is the
case we care about for expert parallel training.

## how to read the numbers

- A speedup is publishable only when the row status is `ok` and the max
  DeepSeek `calc_diff` is at or below the configured gate.
- Custom speed is always measured against Sonic BF16 for the same model, route,
  EP setting, and token count.
- DeepGEMM failed rows are kept in the JSON and Markdown output. They are useful
  diagnostics, but they are not valid speedup rows.
- DeepGEMM is capped to local token counts `<=8192` by default because the
  public Wgrad API has known failures on larger shapes in this harness.
- The benchmark reports valid TFLOP/s from useful rows, not padded rows.

## outputs

Single-shape benchmark output:

```text
<output-dir>/results.json        # full machine-readable result
<output-dir>/summary.json        # compatibility alias
<output-dir>/summary.md          # human-readable table
<output-dir>/results.csv         # compact table
<output-dir>/env.json            # environment snapshot
<output-dir>/logs/*.error.log    # full logs for failed cases
```

Sequence and matrix output:

```text
<output-dir>/matrix_runs.tsv
<output-dir>/results_for_readme.md
<output-dir>/results_flat.csv
<output-dir>/<model>/ep<EP>/<route>/sonic_custom/results.json
<output-dir>/<model>/ep<EP>/<route>/deepgemm/results.json
<output-dir>/<model>/ep<EP>/<route>/*/logs/*.error.log
```

Each JSON row includes implementation provenance, source hash, timing
statistics, valid TFLOP/s, per-expert count statistics, DeepSeek `calc_diff`,
peak GPU memory, and failure details when a baseline cannot run.

## license

The repo is intended to be small enough to audit before reuse. See `LICENSE`
and `NOTICE.md`.
