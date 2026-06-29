# Notices And Provenance

This repository contains a standalone extraction of an experimental SM90 FP8
grouped Wgrad kernel and benchmark harness.

External projects used as runtime dependencies:

- SonicMoE, `sonic-moe==0.1.2.post1`, Apache-2.0.
- QuACK, `quack-kernels==0.4.0`, Apache-2.0.
- DeepGEMM, commit `88965b078186ee7510ab9fc4f1d5ebc19adfa8d1`, MIT.
- NVIDIA CUTLASS/CuTe DSL packages, version `4.4.2`.

The custom kernel is a CuTe/QuACK-style SM90 WGMMA implementation. Keep this
notice and the Apache-2.0 license text when redistributing source that is
derived from QuACK/SonicMoE kernel structure.

The headline result documented in `README.md` is a compute-only Wgrad
comparison. It does not include router, top-k, quantization, packing, dgrad,
dSwiGLU, scatter, autograd wrapper overhead, or full MoE backward time.
