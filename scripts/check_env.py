#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata as md
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def module_status(name: str) -> dict[str, str | bool]:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": True,
        "file": str(getattr(mod, "__file__", "<builtin>")),
        "version": str(getattr(mod, "__version__", "<unknown>")),
    }


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "<missing>"


def custom_fp8_dtype_status() -> dict[str, str | bool]:
    try:
        import torch
        from sm90_fp8_wgrad.gemm_sm90 import _torch_to_cute_dtype

        if not hasattr(torch, "float8_e4m3fn"):
            return {"ok": False, "error": "torch.float8_e4m3fn is missing"}
        return {"ok": True, "cute_dtype": str(_torch_to_cute_dtype(torch.float8_e4m3fn))}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def deepseek_calc_diff_status() -> dict[str, str | bool]:
    try:
        from deep_gemm.testing import calc_diff

        return {"ok": True, "callable": str(calc_diff)}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> int:
    import torch

    print("python", sys.version.replace("\n", " "))
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    print("torch_has_float8_e4m3fn", hasattr(torch, "float8_e4m3fn"))
    if torch.cuda.is_available():
        print("gpu0", torch.cuda.get_device_name(0))
        print("capability0", torch.cuda.get_device_capability(0))

    for pkg in [
        "triton",
        "cuda-python",
        "nvidia-cutlass-dsl",
        "quack-kernels",
        "sonic-moe",
        "deep_gemm",
    ]:
        print(f"pkg {pkg}: {package_version(pkg)}")

    for mod in [
        "triton",
        "cuda.bindings.driver",
        "cutlass",
        "cutlass.cute",
        "quack",
        "sonicmoe",
        "deep_gemm",
        "deep_gemm.testing",
    ]:
        print(f"module {mod}: {module_status(mod)}")

    from sm90_fp8_wgrad import fp8_grouped_wgrad_sm90

    print("custom_kernel_entry", fp8_grouped_wgrad_sm90)
    print("custom_fp8_dtype_map", custom_fp8_dtype_status())
    print("deepseek_calc_diff", deepseek_calc_diff_status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
