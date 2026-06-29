# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:23:26.963408+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep8/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 1.012639 | 1.0000x | 305.378 | 8.694189e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 0.741825 | 1.3651x | 416.861 | 2.363818e-07 |
| 8192 | Sonic BF16 | ok | 2.103238 | 1.0000x | 294.059 | 6.209582e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 1.361647 | 1.5446x | 454.211 | 2.500373e-07 |
| 16384 | Sonic BF16 | ok | 4.154583 | 1.0000x | 297.732 | 2.855470e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 2.675368 | 1.5529x | 462.348 | 2.450891e-07 |
| 32768 | Sonic BF16 | ok | 8.569004 | 1.0000x | 288.703 | 4.965575e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 5.229808 | 1.6385x | 473.039 | 2.424753e-07 |
