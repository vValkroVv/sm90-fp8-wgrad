# benchmark contract

This benchmark measures one thing: grouped MoE Wgrad on SM90.

It is not a full MoE backward benchmark. The following work is outside the timed
region:

- top-k routing
- FP8 quantization
- scale generation
- padding and metadata construction
- dgrad, dSwiGLU, scatter, and autograd wrapper overhead
- tensor allocation

The timed region contains only the two grouped Wgrad GEMMs:

```text
down Wgrad:    grad_y.T       @ hidden
gate/up Wgrad: grad_gate_up.T @ x
```

## default shape

The original public shape is Qwen3-30B-A3B:

```text
experts        = 128
top_k          = 8
hidden         = 2048
intermediate   = 768
down Wgrad     = [experts, hidden, intermediate]
gate/up Wgrad  = [experts, 2 * intermediate, hidden]
scale block K  = 128 rows
input dtype    = BF16 source tensors, FP8 E4M3 grouped operands
output dtype   = BF16 for the custom kernel
```

For a token count `S`:

```text
valid_rows = S * top_k
```

Rows are grouped by expert. Each expert has a variable number of valid rows and
scale blocks of 128 rows.

## model matrix

The full public run uses four MoE shapes:

| model key | label | experts | top-k | hidden | intermediate |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_30b` | Qwen3-30B-A3B | 128 | 8 | 2048 | 768 |
| `deepseek_v3` | DeepSeek-V3 | 256 | 8 | 7168 | 2048 |
| `minimax_m3` | MiniMax-M3 | 128 | 4 | 6144 | 3072 |
| `qwen3_5_397b_a17b` | Qwen3.5-397B-A17B | 512 | 10 | 4096 | 1024 |

Use the matrix runner for this:

```bash
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

## EP proxy

The benchmark is single-process and single-GPU. EP is represented by reducing
the number of local experts while keeping the global token count fixed:

```text
local_experts = global_experts / EP
local_tokens  = global_tokens
```

That is the useful training proxy here. With fewer local experts, each local
expert receives more rows. Do not multiply tokens by EP for the normal matrix
run; that is a stress mode and changes the problem.

## routes

Two synthetic routes are used:

- `balanced`: every expert gets roughly the same number of rows.
- `skewed`: hot experts receive many more rows, closer to the failure mode we
  care about for long MoE sequences.

Both routes must be reported. Balanced results are easier to compare across
backends. Skewed results show whether the grouped scheduler still behaves when
row counts are uneven.

## math operation

The custom kernel computes the scaled FP8 Wgrad:

```text
out[e, m, n] = sum_k lhs_q[k, m] * lhs_scale[block(k), m]
                    * rhs_q[k, n] * rhs_scale[block(k), n]
```

The public benchmark compares the final BF16 Wgrad output against a BF16 source
reference using DeepGEMM's `deep_gemm.testing.calc_diff`.

Default correctness gate:

```text
max DeepSeek calc_diff <= 1e-3
```

Rows that fail this gate are kept in the output, but they are not valid speedup
claims.

## baselines

The matrix compares:

- SonicMoE BF16 Wgrad
- custom CuTe FP8 Wgrad with BF16 output
- DeepGEMM public FP8 Wgrad

Sonic and custom run together by default. DeepGEMM runs separately, so a
DeepGEMM failure does not hide completed Sonic/custom rows.

DeepGEMM is capped to local token counts `<=8192` by default. The public
DeepGEMM Wgrad API has known failures on larger and uneven grouped shapes in
this harness.

## outputs

Each run writes machine-readable JSON and compact publication files:

```text
<output-dir>/matrix_runs.tsv
<output-dir>/results_for_readme.md
<output-dir>/results_flat.csv
<output-dir>/<model>/ep<EP>/<route>/sonic_custom/results.json
<output-dir>/<model>/ep<EP>/<route>/deepgemm/results.json
<output-dir>/<model>/ep<EP>/<route>/*/logs/*.error.log
```

Every JSON row includes timing, valid TFLOP/s, per-expert row counts, padding
stats, DeepSeek `calc_diff`, peak GPU memory, implementation provenance, source
hash, and failure details.
