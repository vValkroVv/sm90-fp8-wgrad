# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:37:23.760020+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep16/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 1.007675 | 1.0000x | 306.882 | 7.449476e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 0.709900 | 1.4195x | 435.607 | 2.561966e-07 |
| 8192 | Sonic BF16 | ok | 1.453327 | 1.0000x | 425.558 | 3.974286e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 1.371511 | 1.0597x | 450.945 | 2.413392e-07 |
| 16384 | Sonic BF16 | ok | 4.037694 | 1.0000x | 306.351 | 6.204019e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 2.654270 | 1.5212x | 466.023 | 2.361397e-07 |
| 32768 | Sonic BF16 | ok | 8.296251 | 1.0000x | 298.195 | 8.940897e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 5.221927 | 1.5887x | 473.753 | 2.387956e-07 |
