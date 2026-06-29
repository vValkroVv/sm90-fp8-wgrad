from __future__ import annotations

import os
from typing import Literal

import torch

from .block_meta import WgradScaleBlockMeta
from .layout_mode import (
    PACKED_ADAPTIVE64_SCALED_TMA_KERNEL_DEBUG_MODES,
    PACKED_ADAPTIVE64_TMA_KERNEL_DEBUG_MODES,
    KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN,
    LAYOUT_MODE_PACKED,
    PACKED_TAIL_TMA_ADAPTIVE32_64_128,
    PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
    PACKED_TAIL_TMA_ADAPTIVE64,
    PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES,
    resolve_wgrad_layout_mode,
)
from .quant import (
    Fp8GroupedWgradOperand,
    packed_kernel_data_base_rows,
    packed_kernel_data_cu_seqlens_k,
    packed_kernel_data_guard_rows,
)


def _check_fp8_operand(name: str, operand: Fp8GroupedWgradOperand) -> None:
    if operand.data.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"{name}.data must be torch.float8_e4m3fn, got {operand.data.dtype}")
    if operand.scales.dtype != torch.float32:
        raise RuntimeError(f"{name}.scales must be torch.float32, got {operand.scales.dtype}")
    if operand.data.ndim != 2 or operand.scales.ndim != 2:
        raise RuntimeError(f"{name} must have data [K,N] and scales [B,N]")
    if int(operand.data.shape[1]) != int(operand.scales.shape[1]):
        raise RuntimeError(f"{name} data/scales column mismatch")
    if int(operand.data.stride(0)) != 1:
        raise RuntimeError(
            f"{name}.data must be SM90 FP8 K-major logical [K,N] with "
            f"stride(0) == 1, got stride={tuple(operand.data.stride())}"
        )
    if int(operand.data.stride(1)) % 16 != 0:
        raise RuntimeError(
            f"{name}.data leading dimension must be divisible by 16 for SM90 FP8 TMA, "
            f"got stride={tuple(operand.data.stride())}"
        )
    if not operand.scales.is_contiguous():
        raise RuntimeError(f"{name}.scales must be contiguous")
    if operand.kernel_data is not None:
        if operand.kernel_data.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"{name}.kernel_data must be torch.float8_e4m3fn, got {operand.kernel_data.dtype}")
        if operand.kernel_data.ndim != 2:
            raise RuntimeError(f"{name}.kernel_data must be 2D, got {tuple(operand.kernel_data.shape)}")
        if int(operand.kernel_data.shape[0]) != int(operand.data.shape[1]):
            raise RuntimeError(
                f"{name}.kernel_data leading dim must match operand cols: "
                f"kernel_data={tuple(operand.kernel_data.shape)} data={tuple(operand.data.shape)}"
            )
        if int(operand.kernel_data.stride(1)) != 1:
            raise RuntimeError(
                f"{name}.kernel_data must be packed varlen-K [cols,K] with "
                f"K-contiguous stride(1) == 1 for SM90 FP8 WGMMA, "
                f"got stride={tuple(operand.kernel_data.stride())}"
            )
        if int(operand.kernel_data.stride(0)) % 16 != 0:
            raise RuntimeError(
                f"{name}.kernel_data row pitch must be divisible by 16 for SM90 FP8 TMA, "
                f"got stride={tuple(operand.kernel_data.stride())}"
            )


def _auto_tuned_config(m: int, n: int) -> str:
    """Resolve the stage-specific Wgrad config for known layer-24 shapes.

    The current H100 sweep shows `default` wins for both down and gate/up after
    the first staged-scale change. Keep auto conservative while still allowing
    fast server-side overrides through env vars.
    """
    fallback = os.environ.get("KERNEL_LAB_WGRAD_AUTO_FALLBACK", "default").strip() or "default"
    if int(m) == 2048 and int(n) == 768:
        return os.environ.get("KERNEL_LAB_WGRAD_AUTO_DOWN", fallback).strip() or fallback
    if int(m) == 1536 and int(n) == 2048:
        return os.environ.get("KERNEL_LAB_WGRAD_AUTO_GATE_UP", fallback).strip() or fallback
    return fallback


_PACKED_DONOR_TMA_KERNEL_MODES = {"tma_only"}


def _packed_uses_donor_tma_layout(kernel_debug_mode: str) -> bool:
    return str(kernel_debug_mode).strip().lower() in _PACKED_DONOR_TMA_KERNEL_MODES


def _packed_uses_kernel_data_cu_seqlens(
    kernel_debug_mode: str,
    packed_tail_tma_mode: str,
) -> bool:
    normalized = str(kernel_debug_mode).strip().lower()
    if normalized in PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES:
        return True
    if normalized in PACKED_ADAPTIVE64_SCALED_TMA_KERNEL_DEBUG_MODES:
        return str(packed_tail_tma_mode) == PACKED_TAIL_TMA_ADAPTIVE64
    return (
        normalized in PACKED_ADAPTIVE64_TMA_KERNEL_DEBUG_MODES
        and str(packed_tail_tma_mode)
        in (
            PACKED_TAIL_TMA_ADAPTIVE64,
            PACKED_TAIL_TMA_ADAPTIVE32_64_128,
            PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
        )
    )


def _packed_donor_tma_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return the proven QuACK varlen-K TMA layout for packed launch tests.

    The low-level SM90 donor smoke that passes uses `[M_or_N, K]` with
    `stride(0) == 1`. The normal packed FP8 Wgrad launch view is K-contiguous
    (`stride(1) == 1`) for later WGMMA experiments. Patch 4 proved this view
    for `tma_only`; Patch 5 uses the same proven TMA address contract while
    enabling the WGMMA consumer in `unscaled`.
    """
    if int(tensor.stride(0)) == 1:
        return tensor
    return tensor.mT.contiguous().mT


def fp8_grouped_wgrad_sm90(
    lhs: Fp8GroupedWgradOperand,
    rhs: Fp8GroupedWgradOperand,
    out: torch.Tensor,
    meta: WgradScaleBlockMeta,
    *,
    output_layout: Literal["emn"] = "emn",
    tuned_config: str = "default",
    accumulate: bool = False,
) -> torch.Tensor:
    """Compute grouped scaled FP8 Wgrad on SM90.

    Contract:
      lhs.data:   logical [K_rows, M], stride(0) == 1, stride(1) % 16 == 0
      rhs.data:   logical [K_rows, N], stride(0) == 1, stride(1) % 16 == 0
      packed lhs/rhs.kernel_data: launch-only [M_or_N, wgmma_K],
                                  stride(1) == 1
      lhs.scales: [total_scale_blocks, M]
      rhs.scales: [total_scale_blocks, N]
      out:        [E, M, N]

    The mathematical operation is:

      out[e, m, n] = sum_b sum_k lhs_q[k,m] * lhs_s[b,m]
                              * rhs_q[k,n] * rhs_s[b,n]

    where `b` ranges over the 128-row scale blocks belonging to expert `e`.
    """
    layout_mode = resolve_wgrad_layout_mode()
    _check_fp8_operand("lhs", lhs)
    _check_fp8_operand("rhs", rhs)
    kernel_debug_mode = os.environ.get("KERNEL_LAB_WGRAD_KERNEL_MODE", "normal").strip().lower()
    use_kernel_data_cu = layout_mode == LAYOUT_MODE_PACKED and _packed_uses_kernel_data_cu_seqlens(
        kernel_debug_mode,
        str(meta.packed_tail_tma_mode),
    )
    if (
        use_kernel_data_cu
        and kernel_debug_mode == KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN
        and str(meta.packed_tail_tma_mode) != PACKED_TAIL_TMA_ADAPTIVE64
    ):
        raise RuntimeError(
            "tma_kmajor_adaptive64_drain requires meta.packed_tail_tma_mode='adaptive64'; "
            f"got {meta.packed_tail_tma_mode!r}"
        )
    expected_k_rows = int(meta.valid_rows) if layout_mode == LAYOUT_MODE_PACKED else int(meta.padded_rows)
    expected_kernel_k_rows = (
        int(packed_kernel_data_base_rows(meta))
        if use_kernel_data_cu
        else int(meta.wgmma_rows) if layout_mode == LAYOUT_MODE_PACKED else expected_k_rows
    )
    if lhs.data.shape[0] != rhs.data.shape[0]:
        raise RuntimeError(
            "lhs/rhs K_rows mismatch: "
            f"lhs={tuple(lhs.data.shape)} stride={tuple(lhs.data.stride())} "
            f"rhs={tuple(rhs.data.shape)} stride={tuple(rhs.data.stride())} "
            f"layout_mode={layout_mode!r}"
        )
    if int(lhs.data.shape[0]) != expected_k_rows:
        raise RuntimeError(
            f"operand K_rows does not match metadata for layout_mode={layout_mode!r}: "
            f"got {int(lhs.data.shape[0])}, expected {expected_k_rows}"
        )
    if layout_mode == LAYOUT_MODE_PACKED:
        for name, operand in (("lhs", lhs), ("rhs", rhs)):
            if operand.kernel_data is None:
                raise RuntimeError(f"packed {name}.kernel_data is required")
            if int(operand.kernel_data.shape[1]) < expected_kernel_k_rows:
                raise RuntimeError(
                    f"packed {name}.kernel_data K rows must cover meta.wgmma_rows: "
                    f"got {tuple(operand.kernel_data.shape)}, expected at least K={expected_kernel_k_rows}"
                )
            guard_rows = (
                int(packed_kernel_data_guard_rows(meta))
                if use_kernel_data_cu
                else int(meta.block_k) - 1
            )
            min_guarded_rows = expected_kernel_k_rows + guard_rows if expected_kernel_k_rows > 0 else 0
            if int(operand.kernel_data.shape[1]) < min_guarded_rows:
                raise RuntimeError(
                    f"packed {name}.kernel_data must include a {guard_rows}-row TMA guard: "
                    f"got K={int(operand.kernel_data.shape[1])}, expected at least {min_guarded_rows}"
                )
    if lhs.scales.shape[0] != rhs.scales.shape[0]:
        raise RuntimeError("lhs/rhs scale block count mismatch")
    if int(lhs.scales.shape[0]) != int(meta.total_blocks):
        raise RuntimeError("scale block count does not match metadata")
    if out.ndim != 3:
        raise RuntimeError(f"expected out [E,M,N], got {tuple(out.shape)}")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"out dtype must be bf16 or fp32, got {out.dtype}")
    if output_layout != "emn":
        raise RuntimeError("only contiguous [E,M,N] output is implemented for the benchmark")
    if not out.is_contiguous():
        raise RuntimeError("custom FP8 Wgrad output must be contiguous [E,M,N]")
    resolved_config = _auto_tuned_config(int(out.shape[1]), int(out.shape[2])) if tuned_config == "auto" else tuned_config

    from .gemm_sm90 import fp8_grouped_wgrad_sm90_impl

    if layout_mode == LAYOUT_MODE_PACKED and lhs.kernel_data is not None and rhs.kernel_data is not None:
        if _packed_uses_donor_tma_layout(kernel_debug_mode):
            lhs_kernel = _packed_donor_tma_view(lhs.kernel_data)
            rhs_kernel = _packed_donor_tma_view(rhs.kernel_data)
        else:
            lhs_kernel = lhs.kernel_data
            rhs_kernel = rhs.kernel_data
    else:
        lhs_kernel = lhs.data
        rhs_kernel = rhs.data
    launch_cu_seqlens_k = (
        packed_kernel_data_cu_seqlens_k(meta)
        if use_kernel_data_cu
        else meta.wgmma_cu_seqlens_k if layout_mode == LAYOUT_MODE_PACKED else meta.padded_cu_seqlens_k
    )

    fp8_grouped_wgrad_sm90_impl(
        lhs_kernel,
        lhs.scales,
        rhs_kernel,
        rhs.scales,
        out,
        launch_cu_seqlens_k,
        meta.block_offsets,
        meta.block_row_start,
        meta.block_row_count,
        tuned_config=resolved_config,
        accumulate=bool(accumulate),
    )
    return out
