# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:38:16.316400+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep16/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 8.359854 |  | 345.247 | 9.328877e-09 |
| 8192 | DeepGEMM FP8 | ok | 13.715353 |  | 420.874 | 7.778818e-09 |
