# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:23:57.828047+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep1/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 1.366708 | 1.0000x | 226.265 | 3.920869e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 1.333550 | 1.0249x | 231.891 | 2.253006e-07 |
| 8192 | Sonic BF16 | ok | 2.317434 | 1.0000x | 266.879 | 1.520341e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 2.033837 | 1.1394x | 304.093 | 2.333746e-07 |
| 16384 | Sonic BF16 | ok | 4.430348 | 1.0000x | 279.199 | 4.613472e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 3.518085 | 1.2593x | 351.598 | 2.443418e-07 |
| 32768 | Sonic BF16 | ok | 9.992668 | 1.0000x | 247.572 | 6.836113e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 6.530552 | 1.5301x | 378.820 | 2.498421e-07 |
