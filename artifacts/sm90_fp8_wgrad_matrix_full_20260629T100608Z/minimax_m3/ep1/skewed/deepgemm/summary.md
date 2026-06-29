# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:35:19.114950+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep1/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 50.537255 |  | 36.714 | 6.933812e-09 |
| 8192 | DeepGEMM FP8 | ok | 52.327059 |  | 70.916 | 8.415405e-09 |
