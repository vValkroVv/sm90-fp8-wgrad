# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T18:19:05.568523+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep16/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 3.882317 |  | 265.510 | 6.906812e-09 |
| 8192 | DeepGEMM FP8 | ok | 5.890270 |  | 349.998 | 3.496607e-09 |
