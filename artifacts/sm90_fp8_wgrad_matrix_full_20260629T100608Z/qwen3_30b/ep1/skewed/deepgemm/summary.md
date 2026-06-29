# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:08:50.522442+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep1/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | failed |  |  |  | 2.234249e-03 |
| 8192 | DeepGEMM FP8 | ok | 4.742124 |  | 130.422 | 9.794324e-04 |
