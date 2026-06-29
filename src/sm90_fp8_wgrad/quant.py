from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except Exception as exc:  # pragma: no cover - import checked at runtime
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

from .block_meta import WgradScaleBlockMeta
from .layout_mode import (
    LAYOUT_MODE_PACKED,
    LAYOUT_MODE_PADDED,
    PACKED_TAIL_TMA_ADAPTIVE32_64_128,
    PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
    PACKED_TAIL_TMA_ADAPTIVE64,
    PACKED_TAIL_TMA_MASKED128,
    resolve_packed_tail_tma_mode,
    resolve_wgrad_layout_mode,
)

FP8_MAX_E4M3 = 448.0
FP8_SCALE_EPS = 1.0e-12


@dataclass(frozen=True)
class Fp8GroupedWgradOperand:
    # Logical shape is [k_rows, N], where k_rows is either the physically padded
    # legacy K extent or the valid packed K extent. Packed ragged-TMA mode keeps
    # `data` in that logical form for references and debugging, but launches the
    # SM90 kernel with a separate [cols, physical_K] `kernel_data` tensor.
    # The launch view is [cols, physical_K] with K-contiguous storage. That keeps
    # global, shared, and WGMMA operand major modes consistent with the working
    # padded FP8 path while still reducing per-expert K padding.
    data: torch.Tensor
    scales: torch.Tensor
    kernel_data: torch.Tensor | None = None


def packed_kernel_data_uses_tma_slots(meta: WgradScaleBlockMeta) -> bool:
    """Return whether packed launch storage uses A2 physical TMA slots.

    `masked128` deliberately keeps the pre-A3 launch contract: physical row
    starts are WGMMA-rounded starts plus a global guard. Adaptive policies use
    the new per-expert `tma_block_row_start` storage so a future kernel can
    issue shorter tail TMA copies without crossing expert boundaries.
    """

    return resolve_packed_tail_tma_mode(meta.packed_tail_tma_mode) != PACKED_TAIL_TMA_MASKED128


def packed_kernel_data_block_row_start(meta: WgradScaleBlockMeta) -> torch.Tensor:
    if packed_kernel_data_uses_tma_slots(meta):
        return meta.tma_block_row_start
    return meta.wgmma_block_row_start


def packed_kernel_data_cu_seqlens_k(meta: WgradScaleBlockMeta) -> torch.Tensor:
    if packed_kernel_data_uses_tma_slots(meta):
        return meta.tma_cu_seqlens_k
    return meta.wgmma_cu_seqlens_k


def packed_kernel_data_base_rows(meta: WgradScaleBlockMeta) -> int:
    if packed_kernel_data_uses_tma_slots(meta):
        return int(meta.tma_rows)
    return int(meta.wgmma_rows)


def packed_kernel_data_guard_rows(meta: WgradScaleBlockMeta) -> int:
    if int(packed_kernel_data_base_rows(meta)) <= 0:
        return 0
    mode = resolve_packed_tail_tma_mode(meta.packed_tail_tma_mode)
    if mode == PACKED_TAIL_TMA_MASKED128:
        return int(meta.block_k) - 1
    if mode == PACKED_TAIL_TMA_ADAPTIVE64:
        return 63
    if mode == PACKED_TAIL_TMA_ADAPTIVE32_64_128:
        # The A2 policy still maps 65..127-row tails to a 128-row physical
        # slot, so keep the conservative final guard for this mixed policy.
        return int(meta.block_k) - 1
    if mode == PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE:
        return 31
    raise RuntimeError(f"unsupported packed tail TMA mode: {mode!r}")


def packed_kernel_data_storage_rows(meta: WgradScaleBlockMeta) -> int:
    return int(packed_kernel_data_base_rows(meta)) + int(packed_kernel_data_guard_rows(meta))


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError(f"Triton is required: {_TRITON_IMPORT_ERROR!r}")


if triton is not None and tl is not None:

    @triton.jit
    def _quantize_grouped_wgrad_128x1_kernel(
        x_ptr,
        q_ptr,
        kernel_q_ptr,
        s_ptr,
        block_row_start_ptr,
        kernel_block_row_start_ptr,
        block_row_count_ptr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        FP8_MAX: tl.constexpr,
        SCALE_EPS: tl.constexpr,
        Q_STRIDE_K: tl.constexpr,
        Q_STRIDE_N: tl.constexpr,
        KERNEL_Q_STRIDE_COL: tl.constexpr,
        KERNEL_Q_STRIDE_K: tl.constexpr,
        PACKED_LAYOUT: tl.constexpr,
        STORE_KERNEL_DATA: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        k_offsets = tl.arange(0, BLOCK_K)
        n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_start = tl.load(block_row_start_ptr + pid_b).to(tl.int64)
        row_count = tl.load(block_row_count_ptr + pid_b).to(tl.int64)

        rows = row_start + k_offsets
        if PACKED_LAYOUT:
            dst_rows = row_start + k_offsets
            kernel_dst_rows = tl.load(kernel_block_row_start_ptr + pid_b).to(tl.int64) + k_offsets
        else:
            dst_rows = pid_b * BLOCK_K + k_offsets
            kernel_dst_rows = dst_rows
        k_mask = k_offsets < row_count
        n_mask = n_offsets < N
        mask = k_mask[:, None] & n_mask[None, :]

        vals = tl.load(
            x_ptr + rows[:, None] * N + n_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        amax = tl.max(tl.abs(vals), axis=0)
        scale = tl.maximum(amax / FP8_MAX, SCALE_EPS)
        tl.store(s_ptr + pid_b * N + n_offsets, scale, mask=n_mask)

        q_vals = vals / scale[None, :]
        tl.store(
            q_ptr + dst_rows[:, None] * Q_STRIDE_K + n_offsets[None, :] * Q_STRIDE_N,
            q_vals,
            mask=mask,
        )
        if STORE_KERNEL_DATA:
            tl.store(
                kernel_q_ptr
                + n_offsets[None, :] * KERNEL_Q_STRIDE_COL
                + kernel_dst_rows[:, None] * KERNEL_Q_STRIDE_K,
                q_vals,
                mask=mask,
            )

else:

    def _quantize_grouped_wgrad_128x1_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"Triton is required: {_TRITON_IMPORT_ERROR!r}")


def quantize_grouped_wgrad_operand_128x1(
    x_grouped: torch.Tensor,
    meta: WgradScaleBlockMeta,
    *,
    block_n: int = 32,
    layout_mode: str | None = None,
) -> Fp8GroupedWgradOperand:
    """Quantize grouped rows for custom scaled FP8 Wgrad.

    Valid rows are read from Sonic's expert-major order.

    `layout_mode="padded"` preserves the legacy physical K padding contract:
    operand data are stored in per-expert 128-row slots and zero-filled beyond
    each expert's valid tail.

    `layout_mode="packed"` keeps operand data packed on valid K rows while
    preserving the same per-128-row scale-block contract. The scale buffers and
    block metadata remain padded per expert; only the FP8 operand rows stop
    materializing padded zeros.
    """
    _require_triton()
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")
    if x_grouped.ndim != 2:
        raise RuntimeError(f"expected x_grouped [valid_TK,N], got {tuple(x_grouped.shape)}")
    if not x_grouped.is_cuda:
        raise RuntimeError("x_grouped must be CUDA")
    if not x_grouped.is_contiguous():
        x_grouped = x_grouped.contiguous()

    valid_tk, n = map(int, x_grouped.shape)
    if int(meta.valid_rows) != valid_tk:
        raise RuntimeError("meta.cu_seqlens_k[-1] must match x_grouped rows")
    if int(meta.padded_rows) != int(meta.total_blocks) * int(meta.block_k):
        raise RuntimeError("metadata padded_rows must equal total_blocks * block_k")

    resolved_layout_mode = resolve_wgrad_layout_mode(layout_mode)
    if resolved_layout_mode == LAYOUT_MODE_PACKED:
        q_rows = int(meta.valid_rows)
        kernel_rows = packed_kernel_data_storage_rows(meta)
        kernel_block_row_start = packed_kernel_data_block_row_start(meta)
    elif resolved_layout_mode == LAYOUT_MODE_PADDED:
        q_rows = int(meta.padded_rows)
        kernel_rows = q_rows
        kernel_block_row_start = meta.wgmma_block_row_start
    else:  # pragma: no cover - resolve_wgrad_layout_mode already checks this
        raise RuntimeError(f"unsupported Wgrad layout mode {resolved_layout_mode!r}")

    # Keep the logical operand compact in shape, but use a 128-byte K pitch.
    # The packed shape is still [valid_K, cols]; this only changes the column
    # stride. The earlier 16-byte-only pitch was legal by our host checks, but
    # the component test showed corrupted logical `data` values while the
    # launch-only kernel_data copy was correct. Matching the 128-byte pitch
    # used by the padded path avoids that FP8 strided-store corner.
    aligned_tk = max(128, ((q_rows + 127) // 128) * 128)
    q = torch.empty_strided(
        (q_rows, n),
        (1, aligned_tk),
        device=x_grouped.device,
        dtype=torch.float8_e4m3fn,
    )
    if resolved_layout_mode == LAYOUT_MODE_PADDED:
        q.zero_()
    kernel_q = None
    if resolved_layout_mode == LAYOUT_MODE_PACKED:
        aligned_k = max(16, ((int(kernel_rows) + 15) // 16) * 16)
        kernel_q = torch.empty(
            (n, aligned_k),
            device=x_grouped.device,
            dtype=torch.float8_e4m3fn,
        )
        kernel_q.zero_()
    scales = torch.empty((int(meta.total_blocks), n), device=x_grouped.device, dtype=torch.float32)
    grid = (int(meta.total_blocks), triton.cdiv(n, int(block_n)))
    _quantize_grouped_wgrad_128x1_kernel[grid](
        x_grouped,
        q,
        q if kernel_q is None else kernel_q,
        scales,
        meta.block_row_start,
        kernel_block_row_start,
        meta.block_row_count,
        n,
        int(meta.block_k),
        int(block_n),
        FP8_MAX_E4M3,
        FP8_SCALE_EPS,
        int(q.stride(0)),
        int(q.stride(1)),
        0 if kernel_q is None else int(kernel_q.stride(0)),
        0 if kernel_q is None else int(kernel_q.stride(1)),
        resolved_layout_mode == LAYOUT_MODE_PACKED,
        kernel_q is not None,
        num_warps=8,
    )
    return Fp8GroupedWgradOperand(q, scales, kernel_data=kernel_q)


def quantize_grouped_wgrad_operand_tail_k(
    x_grouped: torch.Tensor,
    tail_indices: torch.Tensor,
    tail_meta: WgradScaleBlockMeta,
    *,
    block_n: int = 32,
) -> Fp8GroupedWgradOperand:
    """Quantize compact per-expert tail rows for tail64/tail32 Wgrad.

    This is input preparation for the custom FP8 Wgrad tail kernel. It gathers
    only the valid tail rows into compact expert-major order, then reuses the
    normal grouped Wgrad operand layout with `tail_meta.block_k` physical rows.
    """
    if tail_indices.ndim != 1:
        raise RuntimeError(f"expected tail_indices [tail_rows], got {tuple(tail_indices.shape)}")
    if not tail_indices.is_cuda:
        raise RuntimeError("tail_indices must be CUDA")
    if int(tail_meta.block_k) not in (32, 64):
        raise RuntimeError(f"tail_meta.block_k must be 32 or 64, got {tail_meta.block_k}")
    if x_grouped.ndim != 2:
        raise RuntimeError(f"expected x_grouped [valid_TK,N], got {tuple(x_grouped.shape)}")
    if not x_grouped.is_cuda:
        raise RuntimeError("x_grouped must be CUDA")
    tail_source = x_grouped.index_select(
        0,
        tail_indices.to(device=x_grouped.device, dtype=torch.long),
    ).contiguous()
    if int(tail_meta.valid_rows) != int(tail_source.shape[0]):
        raise RuntimeError("tail_meta.valid_rows must match compact tail source rows")
    return quantize_grouped_wgrad_operand_128x1(
        tail_source,
        tail_meta,
        block_n=block_n,
        layout_mode=LAYOUT_MODE_PADDED,
    )
