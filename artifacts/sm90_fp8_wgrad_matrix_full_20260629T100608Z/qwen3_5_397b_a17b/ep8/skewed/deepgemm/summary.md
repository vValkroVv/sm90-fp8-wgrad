# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:59:57.820908+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep8/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 6.510324 |  | 158.332 | 1.047361e-04 |
| 8192 | DeepGEMM FP8 | ok | 8.259523 |  | 249.601 | 1.123904e-08 |
