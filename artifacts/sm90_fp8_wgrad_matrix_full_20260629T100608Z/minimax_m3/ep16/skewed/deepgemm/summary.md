# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T16:43:54.743085+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep16/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 5.276982 |  | 351.607 | 9.119256e-09 |
| 8192 | DeepGEMM FP8 | ok | 10.358785 |  | 358.232 | 8.708290e-09 |
