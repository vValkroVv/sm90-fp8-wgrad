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

## dense matmul calibration

As a hardware sanity check, the same H100 80GB class environment was also run
through a plain BF16 square GEMM benchmark. This is not a Wgrad baseline; it is
only a quick read on how many TFLOP/s a single dense matmul can reach on the
measured card. Full files are in
`artifacts/sm90_fp8_dense_matmul_20260629T201037Z`, with a short table in
`docs/dense_matmul.md`.

| dtype | M=N=K | mean ms | best TFLOP/s | mean TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 8192 | 3.435214 | 321.402 | 320.071 |
| BF16 | 16384 | 28.595499 | 316.584 | 307.604 |
| BF16 | 32768 | 236.528854 | 297.875 | 297.506 |

The short version: this card/environment sustains roughly `300-321` dense BF16
TFLOP/s on large square GEMMs. Grouped MoE Wgrad is harder to schedule, so the
Wgrad TFLOP/s below should be read against that dense upper-context, not as a
direct apples-to-apples GEMM ceiling.

## results

The block below is generated from the full H100 matrix artifact
`artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z`. The complete table is
also kept in `docs/results.md`, with a flat CSV in `docs/results_flat.csv`.

To refresh the block after a new run:

```bash
python scripts/render_matrix_results.py \
  --input artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z \
  --csv docs/results_flat.csv \
  --output docs/results.md \
  --standalone-output \
  --readme README.md
```

The renderer replaces only the block between the markers below.

<!-- SM90_WGRAD_RESULTS_BEGIN -->

Generated at `2026-06-29T20:34:32.183272+00:00` from matrix artifact `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z`.

### run status

| implementation | ok | failed | skipped | total |
| --- | ---: | ---: | ---: | ---: |
| Sonic BF16 | 96 | 0 | 0 | 96 |
| Custom CuTe FP8 | 96 | 0 | 0 | 96 |
| DeepGEMM FP8 | 44 | 4 | 0 | 48 |

### headline

- Custom CuTe FP8 average speed vs Sonic BF16 across valid rows: `1.4604x`.
- Best custom row: `DeepSeek-V3` route `skewed`, EP `8`, 32k tokens, `1.8310x`.
- Slowest valid custom row: `Qwen3.5-397B-A17B` route `skewed`, EP `1`, 4k tokens, `0.9160x`.
- Worst custom DeepSeek `calc_diff`: `2.712e-07`.
- DeepGEMM failed or skipped rows are reported below and are not used for custom speedup claims.

### custom speedup matrix

| model | route | EP | 4k | 8k | 16k | 32k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V3 | balanced | 1 | 1.3031x | 1.4396x | 1.5571x | 1.6086x |
| DeepSeek-V3 | skewed | 1 | 0.9891x | 1.0745x | 1.6750x | 1.4625x |
| DeepSeek-V3 | balanced | 8 | 1.5973x | 1.6376x | 1.5514x | 1.5378x |
| DeepSeek-V3 | skewed | 8 | 1.6688x | 1.7150x | 1.6069x | 1.8310x |
| DeepSeek-V3 | balanced | 16 | 1.6854x | 1.6364x | 1.5654x | 1.5742x |
| DeepSeek-V3 | skewed | 16 | 1.5677x | 1.6526x | 1.6337x | 1.6038x |
| MiniMax-M3 | balanced | 1 | 1.1144x | 1.4387x | 1.1299x | 1.2244x |
| MiniMax-M3 | skewed | 1 | 0.9981x | 1.3599x | 1.4478x | 1.6152x |
| MiniMax-M3 | balanced | 8 | 1.5705x | 1.6140x | 1.6519x | 1.4507x |
| MiniMax-M3 | skewed | 8 | 1.5143x | 1.5665x | 1.7527x | 1.6500x |
| MiniMax-M3 | balanced | 16 | 1.6314x | 1.5363x | 1.4641x | 1.5447x |
| MiniMax-M3 | skewed | 16 | 1.6417x | 1.6763x | 1.7112x | 1.7378x |
| Qwen3-30B-A3B | balanced | 1 | 1.3427x | 1.4316x | 1.5710x | 1.6354x |
| Qwen3-30B-A3B | skewed | 1 | 1.0249x | 1.1394x | 1.2593x | 1.5301x |
| Qwen3-30B-A3B | balanced | 8 | 1.3651x | 1.5446x | 1.5529x | 1.6385x |
| Qwen3-30B-A3B | skewed | 8 | 1.1624x | 1.1800x | 1.2583x | 1.6296x |
| Qwen3-30B-A3B | balanced | 16 | 1.4195x | 1.0597x | 1.5212x | 1.5887x |
| Qwen3-30B-A3B | skewed | 16 | 1.2125x | 1.2670x | 1.3679x | 1.2900x |
| Qwen3.5-397B-A17B | balanced | 1 | 1.0191x | 1.2159x | 1.2628x | 1.5855x |
| Qwen3.5-397B-A17B | skewed | 1 | 0.9160x | 1.0202x | 1.0866x | 1.6327x |
| Qwen3.5-397B-A17B | balanced | 8 | 1.1640x | 1.4846x | 1.6075x | 1.5569x |
| Qwen3.5-397B-A17B | skewed | 8 | 1.2526x | 1.6637x | 1.6542x | 1.6411x |
| Qwen3.5-397B-A17B | balanced | 16 | 1.2569x | 1.6851x | 1.6753x | 1.6042x |
| Qwen3.5-397B-A17B | skewed | 16 | 1.2126x | 1.5852x | 1.5581x | 1.6490x |

### Sonic BF16 vs custom CuTe FP8

| model | route | EP | tokens | Sonic ms | Custom ms | speed vs Sonic | custom TFLOP/s | max calc_diff | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek-V3 | balanced | 1 | 4k | 16.869 | 12.945 | 1.3031x | 223.0 | 2.54e-07 | ok |
| DeepSeek-V3 | balanced | 1 | 8k | 26.466 | 18.385 | 1.4396x | 314.0 | 2.47e-07 | ok |
| DeepSeek-V3 | balanced | 1 | 16k | 45.511 | 29.228 | 1.5571x | 395.0 | 2.41e-07 | ok |
| DeepSeek-V3 | balanced | 1 | 32k | 83.262 | 51.762 | 1.6086x | 446.1 | 2.40e-07 | ok |
| DeepSeek-V3 | skewed | 1 | 4k | 16.160 | 16.338 | 0.9891x | 176.7 | 2.17e-07 | ok |
| DeepSeek-V3 | skewed | 1 | 8k | 23.337 | 21.720 | 1.0745x | 265.8 | 2.26e-07 | ok |
| DeepSeek-V3 | skewed | 1 | 16k | 55.984 | 33.422 | 1.6750x | 345.4 | 2.36e-07 | ok |
| DeepSeek-V3 | skewed | 1 | 32k | 86.055 | 58.842 | 1.4625x | 392.4 | 2.39e-07 | ok |
| DeepSeek-V3 | balanced | 8 | 4k | 10.341 | 6.474 | 1.5973x | 445.8 | 2.39e-07 | ok |
| DeepSeek-V3 | balanced | 8 | 8k | 19.591 | 11.963 | 1.6376x | 482.5 | 2.40e-07 | ok |
| DeepSeek-V3 | balanced | 8 | 16k | 35.947 | 23.171 | 1.5514x | 498.3 | 2.41e-07 | ok |
| DeepSeek-V3 | balanced | 8 | 32k | 73.492 | 47.790 | 1.5378x | 483.1 | 2.39e-07 | ok |
| DeepSeek-V3 | skewed | 8 | 4k | 11.746 | 7.038 | 1.6688x | 410.1 | 2.41e-07 | ok |
| DeepSeek-V3 | skewed | 8 | 8k | 22.116 | 12.896 | 1.7150x | 447.6 | 2.48e-07 | ok |
| DeepSeek-V3 | skewed | 8 | 16k | 40.607 | 25.270 | 1.6069x | 456.9 | 2.52e-07 | ok |
| DeepSeek-V3 | skewed | 8 | 32k | 93.983 | 51.330 | 1.8310x | 449.8 | 2.35e-07 | ok |
| DeepSeek-V3 | balanced | 16 | 4k | 10.246 | 6.079 | 1.6854x | 474.8 | 2.41e-07 | ok |
| DeepSeek-V3 | balanced | 16 | 8k | 19.198 | 11.732 | 1.6364x | 492.0 | 2.39e-07 | ok |
| DeepSeek-V3 | balanced | 16 | 16k | 37.528 | 23.974 | 1.5654x | 481.6 | 2.39e-07 | ok |
| DeepSeek-V3 | balanced | 16 | 32k | 73.770 | 46.863 | 1.5742x | 492.7 | 2.43e-07 | ok |
| DeepSeek-V3 | skewed | 16 | 4k | 9.986 | 6.370 | 1.5677x | 453.1 | 2.42e-07 | ok |
| DeepSeek-V3 | skewed | 16 | 8k | 20.000 | 12.102 | 1.6526x | 477.0 | 2.38e-07 | ok |
| DeepSeek-V3 | skewed | 16 | 16k | 39.983 | 24.473 | 1.6337x | 471.7 | 2.38e-07 | ok |
| DeepSeek-V3 | skewed | 16 | 32k | 79.899 | 49.819 | 1.6038x | 463.5 | 2.42e-07 | ok |
| MiniMax-M3 | balanced | 1 | 4k | 9.328 | 8.371 | 1.1144x | 221.7 | 2.54e-07 | ok |
| MiniMax-M3 | balanced | 1 | 8k | 16.940 | 11.775 | 1.4387x | 315.2 | 2.47e-07 | ok |
| MiniMax-M3 | balanced | 1 | 16k | 21.296 | 18.848 | 1.1299x | 393.8 | 2.41e-07 | ok |
| MiniMax-M3 | balanced | 1 | 32k | 40.204 | 32.835 | 1.2244x | 452.1 | 2.41e-07 | ok |
| MiniMax-M3 | skewed | 1 | 4k | 10.427 | 10.447 | 0.9981x | 177.6 | 2.17e-07 | ok |
| MiniMax-M3 | skewed | 1 | 8k | 18.862 | 13.870 | 1.3599x | 267.5 | 2.25e-07 | ok |
| MiniMax-M3 | skewed | 1 | 16k | 30.744 | 21.235 | 1.4478x | 349.5 | 2.35e-07 | ok |
| MiniMax-M3 | skewed | 1 | 32k | 63.064 | 39.043 | 1.6152x | 380.2 | 2.42e-07 | ok |
| MiniMax-M3 | balanced | 8 | 4k | 6.480 | 4.126 | 1.5705x | 449.7 | 2.42e-07 | ok |
| MiniMax-M3 | balanced | 8 | 8k | 12.388 | 7.675 | 1.6140x | 483.5 | 2.37e-07 | ok |
| MiniMax-M3 | balanced | 8 | 16k | 24.509 | 14.836 | 1.6519x | 500.2 | 2.41e-07 | ok |
| MiniMax-M3 | balanced | 8 | 32k | 46.112 | 31.786 | 1.4507x | 467.0 | 2.42e-07 | ok |
| MiniMax-M3 | skewed | 8 | 4k | 6.684 | 4.414 | 1.5143x | 420.3 | 2.42e-07 | ok |
| MiniMax-M3 | skewed | 8 | 8k | 12.856 | 8.206 | 1.5665x | 452.2 | 2.38e-07 | ok |
| MiniMax-M3 | skewed | 8 | 16k | 27.520 | 15.702 | 1.7527x | 472.7 | 2.42e-07 | ok |
| MiniMax-M3 | skewed | 8 | 32k | 53.855 | 32.640 | 1.6500x | 454.8 | 2.41e-07 | ok |
| MiniMax-M3 | balanced | 16 | 4k | 6.261 | 3.838 | 1.6314x | 483.5 | 2.43e-07 | ok |
| MiniMax-M3 | balanced | 16 | 8k | 11.356 | 7.392 | 1.5363x | 502.0 | 2.41e-07 | ok |
| MiniMax-M3 | balanced | 16 | 16k | 22.728 | 15.524 | 1.4641x | 478.1 | 2.41e-07 | ok |
| MiniMax-M3 | balanced | 16 | 32k | 45.724 | 29.600 | 1.5447x | 501.5 | 2.41e-07 | ok |
| MiniMax-M3 | skewed | 16 | 4k | 6.651 | 4.051 | 1.6417x | 458.0 | 2.47e-07 | ok |
| MiniMax-M3 | skewed | 16 | 8k | 13.014 | 7.763 | 1.6763x | 478.0 | 2.35e-07 | ok |
| MiniMax-M3 | skewed | 16 | 16k | 26.555 | 15.518 | 1.7112x | 478.3 | 2.40e-07 | ok |
| MiniMax-M3 | skewed | 16 | 32k | 54.736 | 31.498 | 1.7378x | 471.2 | 2.38e-07 | ok |
| Qwen3-30B-A3B | balanced | 1 | 4k | 1.444 | 1.075 | 1.3427x | 287.6 | 2.46e-07 | ok |
| Qwen3-30B-A3B | balanced | 1 | 8k | 2.439 | 1.703 | 1.4316x | 363.1 | 2.44e-07 | ok |
| Qwen3-30B-A3B | balanced | 1 | 16k | 4.654 | 2.962 | 1.5710x | 417.6 | 2.40e-07 | ok |
| Qwen3-30B-A3B | balanced | 1 | 32k | 8.890 | 5.436 | 1.6354x | 455.1 | 2.41e-07 | ok |
| Qwen3-30B-A3B | skewed | 1 | 4k | 1.367 | 1.334 | 1.0249x | 231.9 | 2.25e-07 | ok |
| Qwen3-30B-A3B | skewed | 1 | 8k | 2.317 | 2.034 | 1.1394x | 304.1 | 2.33e-07 | ok |
| Qwen3-30B-A3B | skewed | 1 | 16k | 4.430 | 3.518 | 1.2593x | 351.6 | 2.44e-07 | ok |
| Qwen3-30B-A3B | skewed | 1 | 32k | 9.993 | 6.531 | 1.5301x | 378.8 | 2.50e-07 | ok |
| Qwen3-30B-A3B | balanced | 8 | 4k | 1.013 | 0.742 | 1.3651x | 416.9 | 2.36e-07 | ok |
| Qwen3-30B-A3B | balanced | 8 | 8k | 2.103 | 1.362 | 1.5446x | 454.2 | 2.50e-07 | ok |
| Qwen3-30B-A3B | balanced | 8 | 16k | 4.155 | 2.675 | 1.5529x | 462.3 | 2.45e-07 | ok |
| Qwen3-30B-A3B | balanced | 8 | 32k | 8.569 | 5.230 | 1.6385x | 473.0 | 2.42e-07 | ok |
| Qwen3-30B-A3B | skewed | 8 | 4k | 0.966 | 0.831 | 1.1624x | 372.0 | 2.52e-07 | ok |
| Qwen3-30B-A3B | skewed | 8 | 8k | 1.795 | 1.521 | 1.1800x | 406.6 | 2.41e-07 | ok |
| Qwen3-30B-A3B | skewed | 8 | 16k | 3.731 | 2.965 | 1.2583x | 417.1 | 2.22e-07 | ok |
| Qwen3-30B-A3B | skewed | 8 | 32k | 9.592 | 5.886 | 1.6296x | 420.3 | 2.65e-07 | ok |
| Qwen3-30B-A3B | balanced | 16 | 4k | 1.008 | 0.710 | 1.4195x | 435.6 | 2.56e-07 | ok |
| Qwen3-30B-A3B | balanced | 16 | 8k | 1.453 | 1.372 | 1.0597x | 450.9 | 2.41e-07 | ok |
| Qwen3-30B-A3B | balanced | 16 | 16k | 4.038 | 2.654 | 1.5212x | 466.0 | 2.36e-07 | ok |
| Qwen3-30B-A3B | balanced | 16 | 32k | 8.296 | 5.222 | 1.5887x | 473.8 | 2.39e-07 | ok |
| Qwen3-30B-A3B | skewed | 16 | 4k | 0.954 | 0.787 | 1.2125x | 393.2 | 2.35e-07 | ok |
| Qwen3-30B-A3B | skewed | 16 | 8k | 1.841 | 1.453 | 1.2670x | 425.6 | 2.45e-07 | ok |
| Qwen3-30B-A3B | skewed | 16 | 16k | 3.883 | 2.839 | 1.3679x | 435.8 | 2.44e-07 | ok |
| Qwen3-30B-A3B | skewed | 16 | 32k | 7.614 | 5.902 | 1.2900x | 419.1 | 2.71e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 1 | 4k | 7.652 | 7.509 | 1.0191x | 137.3 | 1.91e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 1 | 8k | 12.737 | 10.475 | 1.2159x | 196.8 | 2.23e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 1 | 16k | 17.748 | 14.055 | 1.2628x | 293.4 | 2.27e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 1 | 32k | 32.287 | 20.364 | 1.5855x | 405.0 | 2.40e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 1 | 4k | 7.959 | 8.689 | 0.9160x | 118.6 | 2.09e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 1 | 8k | 10.747 | 10.534 | 1.0202x | 195.7 | 2.18e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 1 | 16k | 15.601 | 14.357 | 1.0866x | 287.2 | 2.33e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 1 | 32k | 38.394 | 23.516 | 1.6327x | 350.7 | 2.40e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 8 | 4k | 2.989 | 2.568 | 1.1640x | 401.4 | 2.40e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 8 | 8k | 6.798 | 4.579 | 1.4846x | 450.2 | 2.41e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 8 | 16k | 13.742 | 8.549 | 1.6075x | 482.3 | 2.41e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 8 | 32k | 26.077 | 16.749 | 1.5569x | 492.3 | 2.35e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 8 | 4k | 3.562 | 2.844 | 1.2526x | 362.5 | 2.36e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 8 | 8k | 8.268 | 4.970 | 1.6637x | 414.8 | 2.38e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 8 | 16k | 15.439 | 9.333 | 1.6542x | 441.8 | 2.34e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 8 | 32k | 30.001 | 18.280 | 1.6411x | 451.1 | 2.33e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 16 | 4k | 2.951 | 2.348 | 1.2569x | 439.0 | 2.46e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 16 | 8k | 7.310 | 4.338 | 1.6851x | 475.2 | 2.49e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 16 | 16k | 14.121 | 8.429 | 1.6753x | 489.2 | 2.44e-07 | ok |
| Qwen3.5-397B-A17B | balanced | 16 | 32k | 27.277 | 17.004 | 1.6042x | 485.0 | 2.40e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 16 | 4k | 3.005 | 2.478 | 1.2126x | 416.0 | 2.48e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 16 | 8k | 7.222 | 4.556 | 1.5852x | 452.5 | 2.52e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 16 | 16k | 13.809 | 8.863 | 1.5581x | 465.2 | 2.39e-07 | ok |
| Qwen3.5-397B-A17B | skewed | 16 | 32k | 29.253 | 17.741 | 1.6490x | 464.8 | 2.47e-07 | ok |

### DeepGEMM public baseline

| model | route | EP | tokens | status | total ms | max calc_diff | note |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| DeepSeek-V3 | balanced | 1 | 4k | ok | 77.472 | 8.76e-09 |  |
| DeepSeek-V3 | balanced | 1 | 8k | ok | 80.011 | 8.48e-09 |  |
| DeepSeek-V3 | skewed | 1 | 4k | ok | 79.994 | 8.49e-09 |  |
| DeepSeek-V3 | skewed | 1 | 8k | ok | 83.449 | 7.89e-09 |  |
| DeepSeek-V3 | balanced | 8 | 4k | ok | 11.394 | 9.06e-09 |  |
| DeepSeek-V3 | balanced | 8 | 8k | ok | 15.567 | 9.06e-09 |  |
| DeepSeek-V3 | skewed | 8 | 4k | ok | 12.302 | 1.07e-08 |  |
| DeepSeek-V3 | skewed | 8 | 8k | ok | 17.633 | 9.86e-09 |  |
| DeepSeek-V3 | balanced | 16 | 4k | ok | 7.560 | 9.35e-09 |  |
| DeepSeek-V3 | balanced | 16 | 8k | ok | 12.410 | 7.99e-09 |  |
| DeepSeek-V3 | skewed | 16 | 4k | ok | 8.360 | 9.33e-09 |  |
| DeepSeek-V3 | skewed | 16 | 8k | ok | 13.715 | 7.78e-09 |  |
| MiniMax-M3 | balanced | 1 | 4k | ok | 49.452 | 8.12e-09 |  |
| MiniMax-M3 | balanced | 1 | 8k | ok | 50.606 | 7.72e-09 |  |
| MiniMax-M3 | skewed | 1 | 4k | ok | 50.537 | 6.93e-09 |  |
| MiniMax-M3 | skewed | 1 | 8k | ok | 52.327 | 8.42e-09 |  |
| MiniMax-M3 | balanced | 8 | 4k | ok | 7.166 | 9.64e-09 |  |
| MiniMax-M3 | balanced | 8 | 8k | ok | 9.789 | 8.30e-09 |  |
| MiniMax-M3 | skewed | 8 | 4k | ok | 7.525 | 9.82e-09 |  |
| MiniMax-M3 | skewed | 8 | 8k | ok | 10.624 | 6.30e-09 |  |
| MiniMax-M3 | balanced | 16 | 4k | ok | 5.021 | 9.95e-09 |  |
| MiniMax-M3 | balanced | 16 | 8k | ok | 8.292 | 1.04e-08 |  |
| MiniMax-M3 | skewed | 16 | 4k | ok | 5.277 | 9.12e-09 |  |
| MiniMax-M3 | skewed | 16 | 8k | ok | 10.359 | 8.71e-09 |  |
| Qwen3-30B-A3B | balanced | 1 | 4k | ok | 4.266 | 7.46e-09 |  |
| Qwen3-30B-A3B | balanced | 1 | 8k | ok | 4.520 | 1.14e-08 |  |
| Qwen3-30B-A3B | skewed | 1 | 4k | failed | 4.367 | 2.23e-03 | DeepSeek calc_diff max=2.234249e-03 exceeds gate 1.000000e-03; down=2.234249e-03; gate_up=1.282205e-03 |
| Qwen3-30B-A3B | skewed | 1 | 8k | ok | 4.742 | 9.79e-04 |  |
| Qwen3-30B-A3B | balanced | 8 | 4k | ok | 0.913 | 7.46e-09 |  |
| Qwen3-30B-A3B | balanced | 8 | 8k | ok | 1.483 | 1.62e-08 |  |
| Qwen3-30B-A3B | skewed | 8 | 4k | ok | 1.016 | 1.12e-08 |  |
| Qwen3-30B-A3B | skewed | 8 | 8k | ok | 1.671 | 1.24e-08 |  |
| Qwen3-30B-A3B | balanced | 16 | 4k | ok | 0.789 | 1.24e-08 |  |
| Qwen3-30B-A3B | balanced | 16 | 8k | ok | 1.409 | 1.49e-08 |  |
| Qwen3-30B-A3B | skewed | 16 | 4k | ok | 0.832 | 4.98e-09 |  |
| Qwen3-30B-A3B | skewed | 16 | 8k | ok | 1.474 | 8.71e-09 |  |
| Qwen3.5-397B-A17B | balanced | 1 | 4k | failed | 44.480 | 1.93e-03 | DeepSeek calc_diff max=1.927168e-03 exceeds gate 1.000000e-03; down=1.927168e-03; gate_up=4.641518e-09 |
| Qwen3.5-397B-A17B | balanced | 1 | 8k | failed |  |  | RuntimeError: CUDA driver error (csrc/apis/../jit_kernels/impls/../../jit/handle.hpp:178): 1 (CUDA_ERROR_INVALID_VALUE, invalid argument) |
| Qwen3.5-397B-A17B | skewed | 1 | 4k | ok | 45.325 | 5.37e-04 |  |
| Qwen3.5-397B-A17B | skewed | 1 | 8k | failed |  |  | RuntimeError: CUDA driver error (csrc/apis/../jit_kernels/impls/../../jit/handle.hpp:178): 1 (CUDA_ERROR_INVALID_VALUE, invalid argument) |
| Qwen3.5-397B-A17B | balanced | 8 | 4k | ok | 5.955 | 1.23e-08 |  |
| Qwen3.5-397B-A17B | balanced | 8 | 8k | ok | 7.014 | 8.58e-09 |  |
| Qwen3.5-397B-A17B | skewed | 8 | 4k | ok | 6.510 | 1.05e-04 |  |
| Qwen3.5-397B-A17B | skewed | 8 | 8k | ok | 8.260 | 1.12e-08 |  |
| Qwen3.5-397B-A17B | balanced | 16 | 4k | ok | 3.432 | 7.46e-09 |  |
| Qwen3.5-397B-A17B | balanced | 16 | 8k | ok | 5.128 | 1.34e-08 |  |
| Qwen3.5-397B-A17B | skewed | 16 | 4k | ok | 3.882 | 6.91e-09 |  |
| Qwen3.5-397B-A17B | skewed | 16 | 8k | ok | 5.890 | 3.50e-09 |  |

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

## dense matmul sanity benchmark

This is a plain cuBLAS/PyTorch `C=A@B` check. It is not grouped Wgrad. Use it
to see the dense GEMM TFLOP/s available on the same GPU and environment.

```bash
python scripts/benchmark_dense_matmul.py \
  --sizes 8192,16384,32768 \
  --dtype bf16 \
  --warmup-iters 20 \
  --active-iters 100 \
  --repeat 3 \
  --output-dir artifacts/dense_matmul_bf16

cat artifacts/dense_matmul_bf16/summary.md
```

The script counts FLOPs as `2 * M * N * K` and reports lower-bound HBM bytes
as `A read + B read + C write`.

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
<output-dir>/results.json                  # dense matmul and single-shape runs
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



