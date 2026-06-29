# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T16:20:26.274039+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep1/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 10.427232 | 1.0000x | 177.940 | 2.036334e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 10.447004 | 0.9981x | 177.604 | 2.173563e-07 |
| 8192 | Sonic BF16 | ok | 18.862130 | 1.0000x | 196.736 | 7.271315e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 13.870197 | 1.3599x | 267.541 | 2.249614e-07 |
| 16384 | Sonic BF16 | ok | 30.744018 | 1.0000x | 241.403 | 1.443799e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 21.234940 | 1.4478x | 349.504 | 2.346033e-07 |
| 32768 | Sonic BF16 | ok | 63.064073 | 1.0000x | 235.370 | 3.377439e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 39.043453 | 1.6152x | 380.177 | 2.423090e-07 |
