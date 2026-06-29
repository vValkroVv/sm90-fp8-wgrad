# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T11:31:57.372694+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep8/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 11.393739 |  | 253.316 | 9.060447e-09 |
| 8192 | DeepGEMM FP8 | ok | 15.567137 |  | 370.809 | 9.063021e-09 |
