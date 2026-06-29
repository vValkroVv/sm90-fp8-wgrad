# dense matmul calibration

Source artifact: `artifacts/sm90_fp8_dense_matmul_20260629T201037Z`.

This is a simple BF16 square GEMM sanity check on the same H100 80GB class
environment used for the Wgrad matrix. It is not a Wgrad benchmark. It gives a
rough compute ceiling for a large dense matmul on the measured card.

FLOPs are counted as `2 * M * N * K`.

| dtype | M | N | K | mean ms | best TFLOP/s | mean TFLOP/s | peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8192 | 8192 | 8192 | 3.435214 | 321.402 | 320.071 | 0.436 |
| BF16 | 16384 | 16384 | 16384 | 28.595499 | 316.584 | 307.604 | 1.644 |
| BF16 | 32768 | 32768 | 32768 | 236.528854 | 297.875 | 297.506 | 6.476 |

Takeaway: this run sees about `300-321` dense BF16 TFLOP/s from a single large
matmul. The Wgrad TFLOP/s in `docs/results.md` should be read against that
hardware calibration, while remembering that grouped MoE Wgrad has routing,
padding, per-expert size variance, and grouped-kernel scheduling overheads that
a single dense GEMM does not have.
