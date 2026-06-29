# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:24:45.446456+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep1/skewed/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 45.324785 |  | 22.742 | 5.365064e-04 |
| 8192 | DeepGEMM FP8 | failed |  |  |  |  |
