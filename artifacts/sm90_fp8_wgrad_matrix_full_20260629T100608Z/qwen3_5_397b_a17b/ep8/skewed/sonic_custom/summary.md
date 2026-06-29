# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T18:43:10.102452+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep8/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 3.561886 | 1.0000x | 289.395 | 1.196501e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 2.843596 | 1.2526x | 362.496 | 2.362489e-07 |
| 8192 | Sonic BF16 | ok | 8.268024 | 1.0000x | 249.344 | 3.586145e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 4.969544 | 1.6637x | 414.844 | 2.381189e-07 |
| 16384 | Sonic BF16 | ok | 15.438660 | 1.0000x | 267.068 | 4.260861e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 9.333094 | 1.6542x | 441.779 | 2.344070e-07 |
| 32768 | Sonic BF16 | ok | 30.000736 | 1.0000x | 274.871 | 9.923462e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 18.280379 | 1.6411x | 451.103 | 2.325602e-07 |
