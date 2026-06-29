# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:32:56.239741+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep16/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 9.985608 | 1.0000x | 289.038 | 1.836218e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 6.369695 | 1.5677x | 453.117 | 2.418478e-07 |
| 8192 | Sonic BF16 | ok | 20.000124 | 1.0000x | 288.620 | 3.795802e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 12.102166 | 1.6526x | 476.975 | 2.376862e-07 |
| 16384 | Sonic BF16 | ok | 39.982704 | 1.0000x | 288.747 | 6.878563e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 24.473258 | 1.6337x | 471.734 | 2.381855e-07 |
| 32768 | Sonic BF16 | ok | 79.899325 | 1.0000x | 288.985 | 1.374124e-07 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 49.818974 | 1.6038x | 463.473 | 2.416242e-07 |
