# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T18:12:38.168891+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep16/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 3.431985 |  | 300.349 | 7.463264e-09 |
| 8192 | DeepGEMM FP8 | ok | 5.127645 |  | 402.053 | 1.343557e-08 |
