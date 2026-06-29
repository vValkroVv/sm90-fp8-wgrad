# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T14:25:53.685399+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep1/balanced/deepgemm/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | DeepGEMM FP8 | ok | 49.451543 |  | 37.520 | 8.124207e-09 |
| 8192 | DeepGEMM FP8 | ok | 50.606319 |  | 73.328 | 7.717711e-09 |
