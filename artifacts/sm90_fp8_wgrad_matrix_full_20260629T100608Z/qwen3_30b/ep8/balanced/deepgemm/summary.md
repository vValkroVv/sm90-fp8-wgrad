# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:09:38.420443+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep8/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 0.913022 |  | 338.697 | 7.464666e-09 |
| 8192 | DeepGEMM FP8 | ok | 1.483495 |  | 416.904 | 1.617187e-08 |
