# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:58:11.785737+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep8/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 11.745910 | 1.0000x | 245.721 | 1.144604e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 7.038461 | 1.6688x | 410.064 | 2.412291e-07 |
| 8192 | Sonic BF16 | ok | 22.116386 | 1.0000x | 261.003 | 2.826910e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 12.896214 | 1.7150x | 447.607 | 2.476105e-07 |
| 16384 | Sonic BF16 | ok | 40.606828 | 1.0000x | 284.309 | 5.158041e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 25.270247 | 1.6069x | 456.856 | 2.523211e-07 |
| 32768 | Sonic BF16 | ok | 93.983281 | 1.0000x | 245.679 | 1.049916e-07 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 51.329789 | 1.8310x | 449.831 | 2.354726e-07 |
