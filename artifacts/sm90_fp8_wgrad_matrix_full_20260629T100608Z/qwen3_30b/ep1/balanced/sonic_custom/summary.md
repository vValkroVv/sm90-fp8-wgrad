# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:22:23.826026+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep1/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 1.443680 | 1.0000x | 214.201 | 6.210555e-10 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 1.075213 | 1.3427x | 287.606 | 2.461869e-07 |
| 8192 | Sonic BF16 | ok | 2.438512 | 1.0000x | 253.628 | 1.397083e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 1.703338 | 1.4316x | 363.096 | 2.436099e-07 |
| 16384 | Sonic BF16 | ok | 4.653739 | 1.0000x | 265.797 | 2.018607e-09 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 2.962222 | 1.5710x | 417.575 | 2.397501e-07 |
| 32768 | Sonic BF16 | ok | 8.890301 | 1.0000x | 278.270 | 6.207447e-09 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 5.436136 | 1.6354x | 455.084 | 2.409421e-07 |
