# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T13:32:03.207723+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/deepseek_v3/ep1/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 16.159779 | 1.0000x | 178.605 | 9.115000e-09 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 16.337888 | 0.9891x | 176.658 | 2.165019e-07 |
| 8192 | Sonic BF16 | ok | 23.337084 | 1.0000x | 247.350 | 1.598415e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 21.719816 | 1.0745x | 265.768 | 2.255816e-07 |
| 16384 | Sonic BF16 | ok | 55.983737 | 1.0000x | 206.218 | 3.251172e-08 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 33.422480 | 1.6750x | 345.422 | 2.359840e-07 |
| 32768 | Sonic BF16 | ok | 86.054948 | 1.0000x | 268.314 | 6.312904e-08 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 58.842442 | 1.4625x | 392.399 | 2.389141e-07 |
