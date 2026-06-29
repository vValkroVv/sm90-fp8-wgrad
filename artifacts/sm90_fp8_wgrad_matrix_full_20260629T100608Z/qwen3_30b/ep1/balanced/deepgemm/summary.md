# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:08:44.429604+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep1/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 4.265514 |  | 72.497 | 7.464897e-09 |
| 8192 | DeepGEMM FP8 | ok | 4.520232 |  | 136.824 | 1.135041e-08 |
