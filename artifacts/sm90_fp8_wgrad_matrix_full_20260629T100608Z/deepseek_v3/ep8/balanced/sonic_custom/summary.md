# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:14:24.831648+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep8/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 10.340885 | 1.0000x | 279.107 | 1.866879e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 6.473869 | 1.5973x | 445.826 | 2.390905e-07 |
| 8192 | Sonic BF16 | ok | 19.590820 | 1.0000x | 294.650 | 6.785924e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 11.963453 | 1.6376x | 482.506 | 2.395447e-07 |
| 16384 | Sonic BF16 | ok | 35.946868 | 1.0000x | 321.165 | 1.258683e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 23.170634 | 1.5514x | 498.254 | 2.411174e-07 |
| 32768 | Sonic BF16 | ok | 73.492213 | 1.0000x | 314.179 | 2.398026e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 47.790110 | 1.5378x | 483.149 | 2.394115e-07 |
