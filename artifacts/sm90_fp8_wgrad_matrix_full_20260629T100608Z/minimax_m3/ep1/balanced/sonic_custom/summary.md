# Wgrad Sequence-Length Sweep

created_at: `2026-06-29T15:06:50.704276+00:00`
deepseek_diff_gate: `0.001`
results_json: `artifacts/sm90_fp8_wgrad_matrix_full_20260629T100608Z/minimax_m3/ep1/balanced/sonic_custom/results.json`

| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4096 | Sonic BF16 | ok | 9.328284 | 1.0000x | 198.903 | 2.586168e-10 |
| 4096 | Custom CuTe FP8 BF16-out | ok | 8.370751 | 1.1144x | 221.656 | 2.539655e-07 |
| 8192 | Sonic BF16 | ok | 16.939549 | 1.0000x | 219.064 | 5.819801e-10 |
| 8192 | Custom CuTe FP8 BF16-out | ok | 11.774563 | 1.4387x | 315.158 | 2.468411e-07 |
| 16384 | Sonic BF16 | ok | 21.296044 | 1.0000x | 348.502 | 1.212904e-09 |
| 16384 | Custom CuTe FP8 BF16-out | ok | 18.847903 | 1.1299x | 393.768 | 2.414058e-07 |
| 32768 | Sonic BF16 | ok | 40.203654 | 1.0000x | 369.205 | 2.593456e-09 |
| 32768 | Custom CuTe FP8 BF16-out | ok | 32.834870 | 1.2244x | 452.062 | 2.409898e-07 |
