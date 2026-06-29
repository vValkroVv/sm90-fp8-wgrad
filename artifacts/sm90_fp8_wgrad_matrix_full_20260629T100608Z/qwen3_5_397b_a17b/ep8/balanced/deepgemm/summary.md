# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:55:45.221799+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep8/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 5.955119 |  | 173.093 | 1.231557e-08 |
| 8192 | DeepGEMM FP8 | ok | 7.014172 |  | 293.917 | 8.582929e-09 |
