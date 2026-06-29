# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T10:39:12.926153+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/qwen3_30b/ep16/skewed/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 0.953680 | 1.0000x | 324.257 | 2.979046e-08 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 0.786563 | 1.2125x | 393.151 | 2.349875e-07 |
| 8192 | Sonic BF16 | ok | 1.841155 | 1.0000x | 335.917 | 4.595672e-08 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 1.453118 | 1.2670x | 425.619 | 2.450636e-07 |
| 16384 | Sonic BF16 | ok | 3.882940 | 1.0000x | 318.560 | 1.030890e-07 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 2.838624 | 1.3679x | 435.757 | 2.438149e-07 |
| 32768 | Sonic BF16 | ok | 7.614138 | 1.0000x | 324.909 | 1.925312e-07 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 5.902438 | 1.2900x | 419.132 | 2.712117e-07 |
