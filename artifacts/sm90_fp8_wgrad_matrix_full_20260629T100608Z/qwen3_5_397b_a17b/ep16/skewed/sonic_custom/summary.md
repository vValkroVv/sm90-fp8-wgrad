# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T19:00:36.422787+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep16/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 3.004570 | 1.0000x | 343.075 | 1.154774e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 2.477871 | 1.2126x | 415.999 | 2.476595e-07 |
| 8192 | Sonic BF16 | ok | 7.222286 | 1.0000x | 285.448 | 3.158043e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 4.556148 | 1.5852x | 452.484 | 2.521294e-07 |
| 16384 | Sonic BF16 | ok | 13.808635 | 1.0000x | 298.593 | 6.613049e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 8.862629 | 1.5581x | 465.231 | 2.390155e-07 |
| 32768 | Sonic BF16 | ok | 29.253464 | 1.0000x | 281.893 | 1.276354e-07 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 17.740513 | 1.6490x | 464.831 | 2.470014e-07 |
