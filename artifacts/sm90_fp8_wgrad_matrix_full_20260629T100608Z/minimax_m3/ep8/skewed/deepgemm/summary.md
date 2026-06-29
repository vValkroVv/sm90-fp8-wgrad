# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:42:53.237587+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep8/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 7.524649 |  | 246.580 | 9.820566e-09 |
| 8192 | DeepGEMM FP8 | ok | 10.623582 |  | 349.303 | 6.302203e-09 |
