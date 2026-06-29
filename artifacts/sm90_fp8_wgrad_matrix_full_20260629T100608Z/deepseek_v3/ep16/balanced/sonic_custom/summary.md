# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:17:27.613968+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep16/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 10.245901 | 1.0000x | 281.695 | 8.781470e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 6.079332 | 1.6854x | 474.759 | 2.407649e-07 |
| 8192 | Sonic BF16 | ok | 19.197508 | 1.0000x | 300.687 | 1.170563e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 11.731578 | 1.6364x | 492.043 | 2.393265e-07 |
| 16384 | Sonic BF16 | ok | 37.528390 | 1.0000x | 307.630 | 2.314884e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 23.973721 | 1.5654x | 481.564 | 2.390694e-07 |
| 32768 | Sonic BF16 | ok | 73.769796 | 1.0000x | 312.997 | 4.533372e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 46.862976 | 1.5742x | 492.708 | 2.432873e-07 |
