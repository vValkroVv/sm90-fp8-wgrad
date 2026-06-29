# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:20:03.484083+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep16/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 7.559503 |  | 381.800 | 9.354680e-09 |
| 8192 | DeepGEMM FP8 | ok | 12.409909 |  | 465.147 | 7.994401e-09 |
