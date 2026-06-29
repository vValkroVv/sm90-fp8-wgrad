from __future__ import annotations

from dataclasses import dataclass

import torch

from .layout_mode import (
    PACKED_TAIL_TMA_ADAPTIVE32_64_128,
    PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
    PACKED_TAIL_TMA_ADAPTIVE64,
    PACKED_TAIL_TMA_MASKED128,
    resolve_packed_tail_tma_mode,
)


@dataclass(frozen=True)
class WgradScaleBlockMeta:
    """Scale-block metadata for grouped FP8 Wgrad.

    The metadata is built outside the timed Wgrad section. It uses Sonic's
    expert-major `cu_seqlens_k` route order and maps every expert's real rows
    into 128-row scale blocks.

    `cu_seqlens_k` is always the valid packed expert-major row contract.
    `padded_cu_seqlens_k` preserves the per-expert 128-row scale-block layout
    used by the legacy padded operand path and by scale-buffer addressing.
    `wgmma_cu_seqlens_k` is the packed kernel operand layout: it keeps experts
    compact, but rounds each expert's K extent to 32 rows because SM90 FP8
    WGMMA cannot legally issue a K atom smaller than 32 rows.

    The same metadata therefore supports both:

    - padded operand rows: data materializes per-expert 128-row K slots
    - packed operand rows: data stays compact on valid K rows while scales keep
      the 128-row block contract
    """

    cu_seqlens_k: torch.Tensor
    padded_cu_seqlens_k: torch.Tensor
    wgmma_cu_seqlens_k: torch.Tensor
    tma_cu_seqlens_k: torch.Tensor
    blocks_per_expert: torch.Tensor
    block_offsets: torch.Tensor
    block_row_start: torch.Tensor
    wgmma_block_row_start: torch.Tensor
    tma_block_row_start: torch.Tensor
    block_row_count: torch.Tensor
    tail_valid_rows: torch.Tensor
    tail_wgmma_atoms: torch.Tensor
    tail_copy_k: torch.Tensor
    has_tail: torch.Tensor
    block_k: int
    valid_rows: int
    padded_rows: int
    wgmma_rows: int
    tma_rows: int
    wgmma_k_alignment: int
    total_blocks: int
    packed_tail_tma_mode: str


@dataclass(frozen=True)
class WgradFullBlockSplit:
    """Metadata for a split full-block FP8 path plus compact ragged tails.

    `full_meta` contains only complete 128-row blocks. `tail_indices` indexes
    the original Sonic expert-major valid-row order for rows excluded from
    `full_meta`, and `tail_cu_seqlens_k` groups those tail rows by expert for a
    second varlen-K GEMM.
    """

    full_meta: WgradScaleBlockMeta
    tail_indices: torch.Tensor
    tail_cu_seqlens_k: torch.Tensor
    full_rows: int
    tail_rows: int


@dataclass(frozen=True)
class WgradTailBlockMeta:
    """Compact metadata for each expert's final partial 128-row tail.

    The `meta` fields are expressed in compact tail-row order. `tail_indices`
    maps those compact rows back to Sonic's original expert-major valid rows.
    """

    meta: WgradScaleBlockMeta
    tail_indices: torch.Tensor
    tail_rows: int
    tail_block_k: int


def _ceil_to(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _select_tail_copy_k(tail_rows: torch.Tensor, *, mode: str) -> torch.Tensor:
    mode = resolve_packed_tail_tma_mode(mode)
    zeros = torch.zeros_like(tail_rows)
    has_tail = tail_rows != 0
    if mode == PACKED_TAIL_TMA_MASKED128:
        return torch.where(has_tail, torch.full_like(tail_rows, 128), zeros)
    if mode == PACKED_TAIL_TMA_ADAPTIVE64:
        return torch.where(
            has_tail,
            torch.where(tail_rows <= 64, torch.full_like(tail_rows, 64), torch.full_like(tail_rows, 128)),
            zeros,
        )
    if mode == PACKED_TAIL_TMA_ADAPTIVE32_64_128:
        return torch.where(
            has_tail,
            torch.where(
                tail_rows <= 32,
                torch.full_like(tail_rows, 32),
                torch.where(tail_rows <= 64, torch.full_like(tail_rows, 64), torch.full_like(tail_rows, 128)),
            ),
            zeros,
        )
    if mode == PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE:
        return torch.where(has_tail, torch.div(tail_rows + 31, 32, rounding_mode="floor") * 32, zeros)
    raise RuntimeError(f"unsupported packed tail TMA mode: {mode!r}")


def _build_tma_layout_cpu(
    counts: torch.Tensor,
    block_offsets_cpu: torch.Tensor,
    *,
    block_k: int,
    packed_tail_tma_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build physical TMA offsets for future adaptive packed-K producer slots."""

    block_k_i = int(block_k)
    if block_k_i != 128:
        raise RuntimeError("adaptive packed-tail TMA metadata is defined for 128-row scale blocks")
    mode = resolve_packed_tail_tma_mode(packed_tail_tma_mode)
    full_blocks = torch.div(counts, block_k_i, rounding_mode="floor")
    tail_valid_rows = counts % block_k_i
    tail_copy_k = _select_tail_copy_k(tail_valid_rows, mode=mode)
    tail_wgmma_atoms = torch.div(tail_valid_rows + 31, 32, rounding_mode="floor")
    has_tail = tail_valid_rows != 0
    tma_counts = full_blocks * block_k_i + tail_copy_k

    tma_cu_cpu = torch.empty((int(counts.numel()) + 1,), dtype=torch.int64)
    tma_cu_cpu[0] = 0
    tma_cu_cpu[1:] = torch.cumsum(tma_counts, dim=0)

    total_blocks = int(block_offsets_cpu[-1].item())
    tma_starts = torch.empty((total_blocks,), dtype=torch.int64)
    cursor = 0
    for e, _ in enumerate(counts.tolist()):
        expert_tma_start = int(tma_cu_cpu[e].item())
        b0 = int(block_offsets_cpu[e].item())
        b1 = int(block_offsets_cpu[e + 1].item())
        for block_in_expert in range(b1 - b0):
            tma_starts[cursor] = expert_tma_start + block_in_expert * block_k_i
            cursor += 1

    return (
        tma_cu_cpu,
        tma_starts,
        tail_valid_rows,
        tail_wgmma_atoms,
        tail_copy_k,
        has_tail,
        int(tma_cu_cpu[-1].item()),
    )


def _assert_main_tma_layout(
    *,
    mode: str,
    tma_rows: int,
    padded_rows: int,
    wgmma_rows: int,
    tma_cu_cpu: torch.Tensor,
) -> None:
    if bool((tma_cu_cpu[1:] < tma_cu_cpu[:-1]).any()):
        raise RuntimeError("tma_cu_seqlens_k must be monotonic")
    if int(tma_rows) > int(padded_rows):
        raise RuntimeError("adaptive TMA rows must not exceed padded128 rows")
    if mode == PACKED_TAIL_TMA_MASKED128 and int(tma_rows) != int(padded_rows):
        raise RuntimeError("masked128 TMA rows must preserve padded128 physical rows")
    if mode in (PACKED_TAIL_TMA_ADAPTIVE64, PACKED_TAIL_TMA_ADAPTIVE32_64_128) and int(tma_rows) < int(wgmma_rows):
        raise RuntimeError("adaptive64/adaptive32_64_128 TMA rows must cover WGMMA rows")
    if mode == PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE and int(tma_rows) != int(wgmma_rows):
        raise RuntimeError("adaptive32_64_twophase TMA rows must equal WGMMA rows")


def _identity_tma_fields_cpu(
    counts: torch.Tensor,
    padded_cu_cpu: torch.Tensor,
    block_row_start: torch.Tensor,
    *,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Populate the new TMA fields for legacy diagnostic metadata."""

    block_k_i = int(block_k)
    tail_valid_rows = counts % block_k_i
    tail_wgmma_atoms = torch.div(tail_valid_rows + 31, 32, rounding_mode="floor")
    has_tail = tail_valid_rows != 0
    tail_copy_k = torch.where(
        has_tail,
        torch.div(tail_valid_rows + block_k_i - 1, block_k_i, rounding_mode="floor") * block_k_i,
        torch.zeros_like(tail_valid_rows),
    )
    return (
        padded_cu_cpu,
        block_row_start,
        tail_valid_rows,
        tail_wgmma_atoms,
        tail_copy_k,
        has_tail,
        int(padded_cu_cpu[-1].item()),
    )


def _build_wgmma_layout_cpu(
    cu_cpu: torch.Tensor,
    counts: torch.Tensor,
    block_offsets_cpu: torch.Tensor,
    *,
    block_k: int,
    align_k: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build packed kernel K offsets rounded to SM90 FP8 WGMMA atom granularity."""
    align_k_i = int(align_k)
    if align_k_i != 32:
        raise RuntimeError("SM90 FP8 packed Wgrad currently requires 32-row WGMMA K alignment")
    wgmma_counts = torch.tensor(
        [_ceil_to(int(count.item()), align_k_i) for count in counts],
        dtype=torch.int64,
    )
    wgmma_cu_cpu = torch.empty((int(wgmma_counts.numel()) + 1,), dtype=torch.int64)
    wgmma_cu_cpu[0] = 0
    wgmma_cu_cpu[1:] = torch.cumsum(wgmma_counts, dim=0)

    total_blocks = int(block_offsets_cpu[-1].item())
    wgmma_starts = torch.empty((total_blocks,), dtype=torch.int64)
    cursor = 0
    for e, _ in enumerate(counts.tolist()):
        expert_wgmma_start = int(wgmma_cu_cpu[e].item())
        b0 = int(block_offsets_cpu[e].item())
        b1 = int(block_offsets_cpu[e + 1].item())
        for block_in_expert in range(b1 - b0):
            wgmma_starts[cursor] = expert_wgmma_start + block_in_expert * int(block_k)
            cursor += 1
    return wgmma_cu_cpu, wgmma_starts, int(wgmma_cu_cpu[-1].item())


def build_wgrad_scale_block_meta(
    cu_seqlens_k: torch.Tensor,
    *,
    block_k: int = 128,
    packed_tail_tma_mode: str | None = None,
) -> WgradScaleBlockMeta:
    """Build 128-row scale metadata for grouped valid-row Wgrad operands."""
    if cu_seqlens_k.ndim != 1:
        raise RuntimeError(f"expected cu_seqlens_k [E+1], got {tuple(cu_seqlens_k.shape)}")
    if not cu_seqlens_k.is_cuda:
        raise RuntimeError("cu_seqlens_k must be a CUDA tensor")
    if int(block_k) != 128:
        raise RuntimeError("SM90 FP8 Wgrad scale block must be 128 for this project")
    packed_tail_tma_mode = resolve_packed_tail_tma_mode(packed_tail_tma_mode)

    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous()

    # Input preparation is allowed to synchronize; the timed Wgrad calls do not
    # invoke this helper.
    cu_cpu = cu_seqlens_k.detach().cpu().to(torch.int64)
    counts = cu_cpu[1:] - cu_cpu[:-1]
    if bool((counts < 0).any()):
        raise RuntimeError("cu_seqlens_k must be monotonic")

    blocks = torch.div(counts + int(block_k) - 1, int(block_k), rounding_mode="floor")
    block_offsets_cpu = torch.empty((int(blocks.numel()) + 1,), dtype=torch.int64)
    block_offsets_cpu[0] = 0
    block_offsets_cpu[1:] = torch.cumsum(blocks, dim=0)
    total_blocks = int(block_offsets_cpu[-1].item())
    padded_cu_cpu = block_offsets_cpu * int(block_k)

    starts = torch.empty((total_blocks,), dtype=torch.int64)
    lens = torch.empty((total_blocks,), dtype=torch.int64)
    cursor = 0
    for e, k_e in enumerate(counts.tolist()):
        expert_start = int(cu_cpu[e].item())
        for b in range(int(blocks[e].item())):
            row0 = expert_start + b * int(block_k)
            starts[cursor] = row0
            lens[cursor] = min(int(block_k), expert_start + int(k_e) - row0)
            cursor += 1

    wgmma_cu_cpu, wgmma_starts, wgmma_rows = _build_wgmma_layout_cpu(
        cu_cpu,
        counts,
        block_offsets_cpu,
        block_k=int(block_k),
        align_k=32,
    )
    (
        tma_cu_cpu,
        tma_starts,
        tail_valid_rows,
        tail_wgmma_atoms,
        tail_copy_k,
        has_tail,
        tma_rows,
    ) = _build_tma_layout_cpu(
        counts,
        block_offsets_cpu,
        block_k=int(block_k),
        packed_tail_tma_mode=packed_tail_tma_mode,
    )
    _assert_main_tma_layout(
        mode=packed_tail_tma_mode,
        tma_rows=tma_rows,
        padded_rows=int(padded_cu_cpu[-1].item()),
        wgmma_rows=wgmma_rows,
        tma_cu_cpu=tma_cu_cpu,
    )

    device = cu_seqlens_k.device
    return WgradScaleBlockMeta(
        cu_seqlens_k=cu_seqlens_k,
        padded_cu_seqlens_k=padded_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_cu_seqlens_k=wgmma_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        tma_cu_seqlens_k=tma_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        blocks_per_expert=blocks.to(device=device, dtype=torch.int32).contiguous(),
        block_offsets=block_offsets_cpu.to(device=device, dtype=torch.int32).contiguous(),
        block_row_start=starts.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_block_row_start=wgmma_starts.to(device=device, dtype=torch.int32).contiguous(),
        tma_block_row_start=tma_starts.to(device=device, dtype=torch.int32).contiguous(),
        block_row_count=lens.to(device=device, dtype=torch.int32).contiguous(),
        tail_valid_rows=tail_valid_rows.to(device=device, dtype=torch.int32).contiguous(),
        tail_wgmma_atoms=tail_wgmma_atoms.to(device=device, dtype=torch.int32).contiguous(),
        tail_copy_k=tail_copy_k.to(device=device, dtype=torch.int32).contiguous(),
        has_tail=has_tail.to(device=device, dtype=torch.bool).contiguous(),
        block_k=int(block_k),
        valid_rows=int(cu_cpu[-1].item()),
        padded_rows=int(padded_cu_cpu[-1].item()),
        wgmma_rows=wgmma_rows,
        tma_rows=tma_rows,
        wgmma_k_alignment=32,
        total_blocks=total_blocks,
        packed_tail_tma_mode=packed_tail_tma_mode,
    )


def build_wgrad_tail_block_meta(
    cu_seqlens_k: torch.Tensor,
    *,
    tail_block_k: int = 64,
    full_block_k: int = 128,
) -> WgradTailBlockMeta:
    """Build compact metadata for non-empty per-expert Wgrad tails.

    This is a diagnostic/prototype surface for removing padded 128-row work.
    Full 128-row blocks stay on the main FP8 kernel; only each expert's final
    partial block is compacted and rounded to `tail_block_k`.
    """
    if cu_seqlens_k.ndim != 1:
        raise RuntimeError(f"expected cu_seqlens_k [E+1], got {tuple(cu_seqlens_k.shape)}")
    if not cu_seqlens_k.is_cuda:
        raise RuntimeError("cu_seqlens_k must be a CUDA tensor")
    tail_block_k_i = int(tail_block_k)
    full_block_k_i = int(full_block_k)
    if tail_block_k_i not in (32, 64):
        raise RuntimeError("tail_block_k must be 32 or 64")
    if full_block_k_i != 128:
        raise RuntimeError("full_block_k must stay 128 for the current SM90 Wgrad contract")

    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous()

    cu_cpu = cu_seqlens_k.detach().cpu().to(torch.int64)
    counts = cu_cpu[1:] - cu_cpu[:-1]
    if bool((counts < 0).any()):
        raise RuntimeError("cu_seqlens_k must be monotonic")

    tail_counts = counts % full_block_k_i
    tail_blocks = torch.div(tail_counts + tail_block_k_i - 1, tail_block_k_i, rounding_mode="floor")
    tail_block_offsets_cpu = torch.empty((int(tail_blocks.numel()) + 1,), dtype=torch.int64)
    tail_block_offsets_cpu[0] = 0
    tail_block_offsets_cpu[1:] = torch.cumsum(tail_blocks, dim=0)
    total_tail_blocks = int(tail_block_offsets_cpu[-1].item())
    tail_padded_cu_cpu = tail_block_offsets_cpu * tail_block_k_i

    tail_cu_cpu = torch.empty((int(counts.numel()) + 1,), dtype=torch.int64)
    tail_cu_cpu[0] = 0
    starts = torch.empty((total_tail_blocks,), dtype=torch.int64)
    lens = torch.empty((total_tail_blocks,), dtype=torch.int64)
    tail_indices_parts: list[torch.Tensor] = []
    compact_cursor = 0
    block_cursor = 0
    for e, k_e in enumerate(counts.tolist()):
        expert_start = int(cu_cpu[e].item())
        tail_rows_e = int(tail_counts[e].item())
        full_rows_e = int(k_e) - tail_rows_e
        if tail_rows_e > 0:
            source_tail_start = expert_start + full_rows_e
            tail_indices_parts.append(torch.arange(source_tail_start, source_tail_start + tail_rows_e, dtype=torch.int64))
        for b in range(int(tail_blocks[e].item())):
            row0 = compact_cursor + b * tail_block_k_i
            starts[block_cursor] = row0
            lens[block_cursor] = min(tail_block_k_i, tail_rows_e - b * tail_block_k_i)
            block_cursor += 1
        compact_cursor += tail_rows_e
        tail_cu_cpu[e + 1] = compact_cursor

    if tail_indices_parts:
        tail_indices_cpu = torch.cat(tail_indices_parts, dim=0)
    else:
        tail_indices_cpu = torch.empty((0,), dtype=torch.int64)

    device = cu_seqlens_k.device
    tail_wgmma_block_starts = torch.arange(total_tail_blocks, dtype=torch.int64) * tail_block_k_i
    tail_counts_compact = tail_cu_cpu[1:] - tail_cu_cpu[:-1]
    (
        tma_cu_cpu,
        tma_starts,
        tail_valid_rows,
        tail_wgmma_atoms,
        tail_copy_k,
        has_tail,
        tma_rows,
    ) = _identity_tma_fields_cpu(
        tail_counts_compact,
        tail_padded_cu_cpu,
        starts,
        block_k=tail_block_k_i,
    )
    meta = WgradScaleBlockMeta(
        cu_seqlens_k=tail_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        padded_cu_seqlens_k=tail_padded_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_cu_seqlens_k=tail_padded_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        tma_cu_seqlens_k=tma_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        blocks_per_expert=tail_blocks.to(device=device, dtype=torch.int32).contiguous(),
        block_offsets=tail_block_offsets_cpu.to(device=device, dtype=torch.int32).contiguous(),
        block_row_start=starts.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_block_row_start=tail_wgmma_block_starts.to(device=device, dtype=torch.int32).contiguous(),
        tma_block_row_start=tma_starts.to(device=device, dtype=torch.int32).contiguous(),
        block_row_count=lens.to(device=device, dtype=torch.int32).contiguous(),
        tail_valid_rows=tail_valid_rows.to(device=device, dtype=torch.int32).contiguous(),
        tail_wgmma_atoms=tail_wgmma_atoms.to(device=device, dtype=torch.int32).contiguous(),
        tail_copy_k=tail_copy_k.to(device=device, dtype=torch.int32).contiguous(),
        has_tail=has_tail.to(device=device, dtype=torch.bool).contiguous(),
        block_k=tail_block_k_i,
        valid_rows=int(tail_indices_cpu.numel()),
        padded_rows=int(tail_padded_cu_cpu[-1].item()),
        wgmma_rows=int(tail_padded_cu_cpu[-1].item()),
        tma_rows=tma_rows,
        wgmma_k_alignment=32,
        total_blocks=total_tail_blocks,
        packed_tail_tma_mode=PACKED_TAIL_TMA_MASKED128,
    )
    return WgradTailBlockMeta(
        meta=meta,
        tail_indices=tail_indices_cpu.to(device=device, dtype=torch.long).contiguous(),
        tail_rows=int(tail_indices_cpu.numel()),
        tail_block_k=tail_block_k_i,
    )


def build_wgrad_full_block_split(
    cu_seqlens_k: torch.Tensor,
    *,
    block_k: int = 128,
) -> WgradFullBlockSplit:
    """Build full-block-only FP8 metadata and compact tail indices.

    This intentionally drops each expert's final partial 128-row block from the
    main FP8 kernel. Callers can then feed the compact tail rows to a dedicated
    CuTe tail experiment without doing padded zero work.
    """
    if cu_seqlens_k.ndim != 1:
        raise RuntimeError(f"expected cu_seqlens_k [E+1], got {tuple(cu_seqlens_k.shape)}")
    if not cu_seqlens_k.is_cuda:
        raise RuntimeError("cu_seqlens_k must be a CUDA tensor")
    if int(block_k) != 128:
        raise RuntimeError("SM90 FP8 Wgrad scale block must be 128 for this project")

    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous()

    cu_cpu = cu_seqlens_k.detach().cpu().to(torch.int64)
    counts = cu_cpu[1:] - cu_cpu[:-1]
    if bool((counts < 0).any()):
        raise RuntimeError("cu_seqlens_k must be monotonic")

    block_k_i = int(block_k)
    full_blocks = torch.div(counts, block_k_i, rounding_mode="floor")
    tail_counts = counts - full_blocks * block_k_i

    full_block_offsets_cpu = torch.empty((int(full_blocks.numel()) + 1,), dtype=torch.int64)
    full_block_offsets_cpu[0] = 0
    full_block_offsets_cpu[1:] = torch.cumsum(full_blocks, dim=0)
    total_full_blocks = int(full_block_offsets_cpu[-1].item())
    full_padded_cu_cpu = full_block_offsets_cpu * block_k_i

    full_starts = torch.empty((total_full_blocks,), dtype=torch.int64)
    full_lens = torch.full((total_full_blocks,), block_k_i, dtype=torch.int64)
    cursor = 0
    tail_indices_parts: list[torch.Tensor] = []
    tail_cu_cpu = torch.empty((int(counts.numel()) + 1,), dtype=torch.int64)
    tail_cu_cpu[0] = 0
    for e, k_e in enumerate(counts.tolist()):
        expert_start = int(cu_cpu[e].item())
        full_rows_e = int(full_blocks[e].item()) * block_k_i
        for b in range(int(full_blocks[e].item())):
            full_starts[cursor] = expert_start + b * block_k_i
            cursor += 1
        tail_rows_e = int(tail_counts[e].item())
        if tail_rows_e > 0:
            tail_start = expert_start + full_rows_e
            tail_indices_parts.append(torch.arange(tail_start, tail_start + tail_rows_e, dtype=torch.int64))
        tail_cu_cpu[e + 1] = tail_cu_cpu[e] + tail_rows_e

    if tail_indices_parts:
        tail_indices_cpu = torch.cat(tail_indices_parts, dim=0)
    else:
        tail_indices_cpu = torch.empty((0,), dtype=torch.int64)

    device = cu_seqlens_k.device
    full_wgmma_block_starts = torch.arange(total_full_blocks, dtype=torch.int64) * block_k_i
    full_counts = full_blocks * block_k_i
    (
        tma_cu_cpu,
        tma_starts,
        tail_valid_rows,
        tail_wgmma_atoms,
        tail_copy_k,
        has_tail,
        tma_rows,
    ) = _identity_tma_fields_cpu(
        full_counts,
        full_padded_cu_cpu,
        full_starts,
        block_k=block_k_i,
    )
    full_meta = WgradScaleBlockMeta(
        cu_seqlens_k=cu_seqlens_k,
        padded_cu_seqlens_k=full_padded_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_cu_seqlens_k=full_padded_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        tma_cu_seqlens_k=tma_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        blocks_per_expert=full_blocks.to(device=device, dtype=torch.int32).contiguous(),
        block_offsets=full_block_offsets_cpu.to(device=device, dtype=torch.int32).contiguous(),
        block_row_start=full_starts.to(device=device, dtype=torch.int32).contiguous(),
        wgmma_block_row_start=full_wgmma_block_starts.to(device=device, dtype=torch.int32).contiguous(),
        tma_block_row_start=tma_starts.to(device=device, dtype=torch.int32).contiguous(),
        block_row_count=full_lens.to(device=device, dtype=torch.int32).contiguous(),
        tail_valid_rows=tail_valid_rows.to(device=device, dtype=torch.int32).contiguous(),
        tail_wgmma_atoms=tail_wgmma_atoms.to(device=device, dtype=torch.int32).contiguous(),
        tail_copy_k=tail_copy_k.to(device=device, dtype=torch.int32).contiguous(),
        has_tail=has_tail.to(device=device, dtype=torch.bool).contiguous(),
        block_k=block_k_i,
        valid_rows=int(cu_cpu[-1].item()),
        padded_rows=int(full_padded_cu_cpu[-1].item()),
        wgmma_rows=int(full_padded_cu_cpu[-1].item()),
        tma_rows=tma_rows,
        wgmma_k_alignment=32,
        total_blocks=total_full_blocks,
        packed_tail_tma_mode=PACKED_TAIL_TMA_MASKED128,
    )
    return WgradFullBlockSplit(
        full_meta=full_meta,
        tail_indices=tail_indices_cpu.to(device=device, dtype=torch.long).contiguous(),
        tail_cu_seqlens_k=tail_cu_cpu.to(device=device, dtype=torch.int32).contiguous(),
        full_rows=int(full_blocks.sum().item()) * block_k_i,
        tail_rows=int(tail_counts.sum().item()),
    )
