# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T17:52:02.390666+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_5_397b_a17b/ep1/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 7.652414 | 1.0000x | 134.702 | 2.328755e-10 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 7.508999 | 1.0191x | 137.274 | 1.912409e-07 |
| 8192 | Sonic BF16 | ok | 12.736922 | 1.0000x | 161.859 | 4.657248e-10 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 10.475285 | 1.2159x | 196.805 | 2.226997e-07 |
| 16384 | Sonic BF16 | ok | 17.748193 | 1.0000x | 232.315 | 7.914861e-10 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 14.054525 | 1.2628x | 293.369 | 2.269263e-07 |
| 32768 | Sonic BF16 | ok | 32.286524 | 1.0000x | 255.411 | 1.629881e-09 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 20.363520 | 1.5855x | 404.956 | 2.402205e-07 |
