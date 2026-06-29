# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:23:16.405212+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep8/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 1.016016 |  | 304.363 | 1.119369e-08 |
| 8192 | DeepGEMM FP8 | ok | 1.671392 |  | 370.036 | 1.243389e-08 |
