# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T16:59:43.992626+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep16/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 6.261136 | 1.0000x | 296.340 | 7.034395e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 3.837815 | 1.6314x | 483.459 | 2.425695e-07 |
| 8192 | Sonic BF16 | ok | 11.356238 | 1.0000x | 326.768 | 1.287670e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 7.391908 | 1.5363x | 502.015 | 2.408287e-07 |
| 16384 | Sonic BF16 | ok | 22.727733 | 1.0000x | 326.548 | 2.855193e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 15.523640 | 1.4641x | 478.090 | 2.409688e-07 |
| 32768 | Sonic BF16 | ok | 45.724004 | 1.0000x | 324.631 | 5.298851e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 29.599846 | 1.5447x | 501.469 | 2.406872e-07 |
