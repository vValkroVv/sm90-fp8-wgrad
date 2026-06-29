# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:24:11.440421+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep16/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 0.788655 |  | 392.107 | 1.244606e-08 |
| 8192 | DeepGEMM FP8 | ok | 1.408785 |  | 439.013 | 1.493839e-08 |
