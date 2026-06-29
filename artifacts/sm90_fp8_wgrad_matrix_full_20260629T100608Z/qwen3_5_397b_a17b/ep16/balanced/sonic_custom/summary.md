# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T18:46:23.806550+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep16/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 2.951364 | 1.0000x | 349.260 | 3.352888e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 2.348055 | 1.2569x | 438.998 | 2.455214e-07 |
| 8192 | Sonic BF16 | ok | 7.310396 | 1.0000x | 282.007 | 5.961235e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 4.338218 | 1.6851x | 475.215 | 2.485589e-07 |
| 16384 | Sonic BF16 | ok | 14.121107 | 1.0000x | 291.986 | 1.639460e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 8.429144 | 1.6753x | 489.156 | 2.444564e-07 |
| 32768 | Sonic BF16 | ok | 27.277217 | 1.0000x | 302.316 | 3.390116e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 17.003628 | 1.6042x | 484.975 | 2.395401e-07 |
