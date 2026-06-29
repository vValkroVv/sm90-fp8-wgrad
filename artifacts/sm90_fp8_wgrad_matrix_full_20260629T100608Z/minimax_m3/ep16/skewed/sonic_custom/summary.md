# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:52:15.735030+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep16/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 6.651284 | 1.0000x | 278.958 | 1.472954e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 4.051343 | 1.6417x | 457.978 | 2.473591e-07 |
| 8192 | Sonic BF16 | ok | 13.013546 | 1.0000x | 285.153 | 2.116535e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 7.763177 | 1.6763x | 478.007 | 2.346710e-07 |
| 16384 | Sonic BF16 | ok | 26.554825 | 1.0000x | 279.486 | 4.226128e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 15.518363 | 1.7112x | 478.253 | 2.399069e-07 |
| 32768 | Sonic BF16 | ok | 54.736421 | 1.0000x | 271.180 | 8.982475e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 31.498188 | 1.7378x | 471.246 | 2.381862e-07 |
