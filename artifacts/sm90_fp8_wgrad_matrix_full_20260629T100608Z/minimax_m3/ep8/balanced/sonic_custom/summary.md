# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T16:39:28.648318+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep8/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 6.480321 | 1.0000x | 286.317 | 3.517641e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 4.126326 | 1.5705x | 449.656 | 2.424106e-07 |
| 8192 | Sonic BF16 | ok | 12.388114 | 1.0000x | 299.549 | 5.795992e-09 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 7.675225 | 1.6140x | 483.484 | 2.371753e-07 |
| 16384 | Sonic BF16 | ok | 24.508789 | 1.0000x | 302.818 | 1.283459e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 14.836447 | 1.6519x | 500.235 | 2.410554e-07 |
| 32768 | Sonic BF16 | ok | 46.112185 | 1.0000x | 321.898 | 2.693137e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 31.785756 | 1.4507x | 466.983 | 2.421623e-07 |
