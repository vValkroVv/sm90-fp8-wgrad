# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:02:17.108454+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep1/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 16.869061 | 1.0000x | 171.095 | 2.927562e-10 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 12.945158 | 1.3031x | 222.957 | 2.539448e-07 |
| 8192 | Sonic BF16 | ok | 26.466123 | 1.0000x | 218.107 | 5.909091e-10 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 18.384566 | 1.4396x | 313.983 | 2.469737e-07 |
| 16384 | Sonic BF16 | ok | 45.510745 | 1.0000x | 253.674 | 1.263751e-09 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 29.228052 | 1.5571x | 394.993 | 2.407098e-07 |
| 32768 | Sonic BF16 | ok | 83.261556 | 1.0000x | 277.316 | 2.461056e-09 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 51.761806 | 1.6086x | 446.077 | 2.398619e-07 |
