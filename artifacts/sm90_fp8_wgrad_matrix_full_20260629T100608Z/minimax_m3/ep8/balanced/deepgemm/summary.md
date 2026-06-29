# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:37:44.983534+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep8/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 7.165905 |  | 258.924 | 9.637606e-09 |
| 8192 | DeepGEMM FP8 | ok | 9.788682 |  | 379.096 | 8.300021e-09 |
