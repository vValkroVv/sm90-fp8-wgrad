# Dense Matmul Benchmark

created_at: `2026-06-29T20:40:09.712667+00:00`
results_json: `artifacts/sm90_fp8_dense_matmul_20260629T201037Z/results.json`

FLOPs are counted as `2 * M * N * K`. HBM bytes are the lower bound: read A, read B, write C.

| dtype | M | N | K | status | mean ms | best TFLOP/s | mean TFLOP/s | peak GB |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| BF16 | 8192 | 8192 | 8192 | ok | 3.435214 | 321.402 | 320.071 | 0.436 |
| BF16 | 16384 | 16384 | 16384 | ok | 28.595499 | 316.584 | 307.604 | 1.644 |
| BF16 | 32768 | 32768 | 32768 | ok | 236.528854 | 297.875 | 297.506 | 6.476 |
