# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T11:24:23.170255+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep1/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 79.993834 |  | 36.081 | 8.486347e-09 |
| 8192 | DeepGEMM FP8 | ok | 83.449420 |  | 69.173 | 7.889550e-09 |
