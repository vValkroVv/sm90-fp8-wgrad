# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T16:24:14.334005+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep16/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 5.021357 |  | 369.507 | 9.951283e-09 |
| 8192 | DeepGEMM FP8 | ok | 8.291998 |  | 447.522 | 1.042175e-08 |
