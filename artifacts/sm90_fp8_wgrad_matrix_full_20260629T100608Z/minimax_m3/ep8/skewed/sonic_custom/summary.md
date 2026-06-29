# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:04:24.063202+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep8/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 6.684421 | 1.0000x | 277.575 | 9.009095e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 4.414130 | 1.5143x | 420.338 | 2.422193e-07 |
| 8192 | Sonic BF16 | ok | 12.855522 | 1.0000x | 288.658 | 2.023284e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 8.206380 | 1.5665x | 452.191 | 2.381209e-07 |
| 16384 | Sonic BF16 | ok | 27.520391 | 1.0000x | 269.680 | 3.232884e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 15.701926 | 1.7527x | 472.662 | 2.415874e-07 |
| 32768 | Sonic BF16 | ok | 53.855312 | 1.0000x | 275.616 | 7.061707e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 32.640415 | 1.6500x | 454.755 | 2.407557e-07 |
