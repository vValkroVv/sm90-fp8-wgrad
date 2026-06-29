# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:24:57.487642+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep16/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 0.832007 |  | 371.677 | 4.979257e-09 |
| 8192 | DeepGEMM FP8 | ok | 1.474257 |  | 419.517 | 8.708727e-09 |
