# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T18:16:15.697762+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep1/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 7.959047 | 1.0000x | 129.512 | 1.034023e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 8.689320 | 0.9160x | 118.627 | 2.089290e-07 |
| 8192 | Sonic BF16 | ok | 10.747175 | 1.0000x | 191.826 | 2.101281e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 10.534474 | 1.0202x | 195.699 | 2.183499e-07 |
| 16384 | Sonic BF16 | ok | 15.600683 | 1.0000x | 264.294 | 3.417271e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 14.357369 | 1.0866x | 287.181 | 2.327823e-07 |
| 32768 | Sonic BF16 | ok | 38.394257 | 1.0000x | 214.780 | 6.664254e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 23.516303 | 1.6327x | 350.665 | 2.400663e-07 |
