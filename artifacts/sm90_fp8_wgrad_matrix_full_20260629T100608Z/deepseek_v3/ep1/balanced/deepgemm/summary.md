# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T11:08:30.032238+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep1/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 77.471977 |  | 37.255 | 8.760602e-09 |
| 8192 | DeepGEMM FP8 | ok | 80.010515 |  | 72.146 | 8.478974e-09 |
