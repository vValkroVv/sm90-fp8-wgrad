# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:10:40.196308+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep8/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 12.302073 |  | 234.612 | 1.072861e-08 |
| 8192 | DeepGEMM FP8 | ok | 17.632735 |  | 327.370 | 9.860416e-09 |
