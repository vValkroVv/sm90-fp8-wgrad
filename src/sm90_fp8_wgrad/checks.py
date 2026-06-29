from __future__ import annotations

import os

import torch

from .block_meta import WgradScaleBlockMeta
from .layout_mode import LAYOUT_MODE_PACKED, LAYOUT_MODE_PADDED, resolve_wgrad_layout_mode
from .quant import Fp8GroupedWgradOperand, packed_kernel_data_block_row_start


def _operand_layout_mode_for_rows(meta: WgradScaleBlockMeta, k_rows: int) -> str:
    if int(k_rows) == int(meta.valid_rows) and int(k_rows) != int(meta.padded_rows):
        return LAYOUT_MODE_PACKED
    if int(k_rows) == int(meta.padded_rows) and int(k_rows) != int(meta.valid_rows):
        return LAYOUT_MODE_PADDED
    return resolve_wgrad_layout_mode()


def _block_data_row0(meta: WgradScaleBlockMeta, k_rows: int, block_idx: int) -> int:
    layout_mode = _operand_layout_mode_for_rows(meta, k_rows)
    if layout_mode == LAYOUT_MODE_PACKED:
        return int(meta.block_row_start[block_idx].item())
    return int(block_idx) * int(meta.block_k)


def _operand_block_rows(
    operand: Fp8GroupedWgradOperand,
    meta: WgradScaleBlockMeta,
    block_idx: int,
) -> torch.Tensor:
    k_rows = int(operand.data.shape[0])
    layout_mode = _operand_layout_mode_for_rows(meta, k_rows)
    row0 = _block_data_row0(meta, k_rows, block_idx)
    rows = int(meta.block_row_count[block_idx].item())
    if layout_mode == LAYOUT_MODE_PACKED and operand.kernel_data is not None:
        kernel_row0 = int(packed_kernel_data_block_row_start(meta)[block_idx].item())
        return operand.kernel_data[:, kernel_row0 : kernel_row0 + rows].mT.float()
    return operand.data[row0 : row0 + rows].float()


def ref_scaled_wgrad(
    lhs: Fp8GroupedWgradOperand,
    rhs: Fp8GroupedWgradOperand,
    meta: WgradScaleBlockMeta,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantized FP8 reference for the grouped scaled Wgrad contract."""
    e_count = int(meta.cu_seqlens_k.numel()) - 1
    m = int(lhs.data.shape[1])
    n = int(rhs.data.shape[1])
    out = torch.zeros((e_count, m, n), device=lhs.data.device, dtype=torch.float32)
    for e in range(e_count):
        b0 = int(meta.block_offsets[e].item())
        b1 = int(meta.block_offsets[e + 1].item())
        for b in range(b0, b1):
            l = _operand_block_rows(lhs, meta, b) * lhs.scales[b].view(1, m)
            r = _operand_block_rows(rhs, meta, b) * rhs.scales[b].view(1, n)
            out[e] += l.T @ r
    return out.to(out_dtype)


def iter_ref_scaled_wgrad_by_expert(
    lhs: Fp8GroupedWgradOperand,
    rhs: Fp8GroupedWgradOperand,
    meta: WgradScaleBlockMeta,
    *,
    out_dtype: torch.dtype,
):
    """Yield per-expert dequantized FP8 references without materializing [E,M,N]."""
    e_count = int(meta.cu_seqlens_k.numel()) - 1
    m = int(lhs.data.shape[1])
    n = int(rhs.data.shape[1])
    for e in range(e_count):
        out = torch.zeros((m, n), device=lhs.data.device, dtype=torch.float32)
        b0 = int(meta.block_offsets[e].item())
        b1 = int(meta.block_offsets[e + 1].item())
        for b in range(b0, b1):
            l = _operand_block_rows(lhs, meta, b) * lhs.scales[b].view(1, m)
            r = _operand_block_rows(rhs, meta, b) * rhs.scales[b].view(1, n)
            out += l.T @ r
        expected = out.to(out_dtype)
        if expected is not out:
            del out
        yield e, expected


def scaled_wgrad_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    max_abs_ref = float(torch.nan_to_num(expected_f.abs(), nan=float("inf")).max().item())
    if not torch.isfinite(actual_f).all() or not torch.isfinite(expected_f).all() or not torch.isfinite(diff).all():
        return float("inf"), float("inf"), max_abs_ref
    max_abs = float(diff.max().item())
    max_rel = float((diff / expected_f.abs().clamp_min(1.0e-6)).max().item())
    return max_abs, max_rel, max_abs_ref


def deepseek_calc_diff(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    require_deepgemm: bool = True,
) -> float:
    """Return DeepSeek/DeepGEMM's official tensor diff metric.

    DeepGEMM's implementation materializes full-size FP32 temporaries. That is
    fine for the Qwen3-30B-A3B shape but can OOM on larger MoE shapes during the
    out-of-band correctness check. For large tensors this function computes the
    same formula in chunks:

        1 - 2 * sum(actual * expected) / sum(actual^2 + expected^2)

    `require_deepgemm=True` still requires the DeepGEMM package to be importable
    so server runs do not silently switch to an unrelated metric.
    """
    try:
        from deep_gemm.testing import calc_diff
    except Exception as exc:
        if require_deepgemm:
            raise RuntimeError("DeepGEMM calc_diff is required but not importable") from exc
        return _deepseek_calc_diff_chunked(actual, expected)

    max_official_elements = int(os.environ.get("SM90_WGRAD_OFFICIAL_CALC_DIFF_MAX_ELEMENTS", "16777216"))
    if int(actual.numel()) > max_official_elements:
        return _deepseek_calc_diff_chunked(actual, expected)

    try:
        actual_f = actual.detach().float()
        expected_f = expected.detach().float()
        return float(calc_diff(actual_f, expected_f))
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _deepseek_calc_diff_chunked(actual, expected)
    except Exception as exc:
        if require_deepgemm:
            raise RuntimeError("DeepGEMM calc_diff failed") from exc
        return _deepseek_calc_diff_chunked(actual, expected)


def _tensor_chunks(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_elements: int,
):
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(f"calc_diff shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
    if int(actual.numel()) <= int(max_elements) or actual.ndim == 0:
        yield actual, expected
        return

    if actual.ndim == 1:
        step = max(1, int(max_elements))
    else:
        elems_per_index = max(1, int(actual[0].numel()))
        step = max(1, int(max_elements) // elems_per_index)

    for start in range(0, int(actual.shape[0]), step):
        stop = min(start + step, int(actual.shape[0]))
        actual_chunk = actual[start:stop]
        expected_chunk = expected[start:stop]
        if int(actual_chunk.numel()) > int(max_elements) and actual_chunk.ndim > 1 and int(actual_chunk.shape[0]) == 1:
            yield from _tensor_chunks(actual_chunk[0], expected_chunk[0], max_elements=max_elements)
        else:
            yield actual_chunk, expected_chunk


def _deepseek_calc_diff_chunked(actual: torch.Tensor, expected: torch.Tensor) -> float:
    max_chunk_elements = int(os.environ.get("SM90_WGRAD_CALC_DIFF_CHUNK_ELEMENTS", "8388608"))
    numerator = 0.0
    denominator = 0.0
    for actual_chunk, expected_chunk in _tensor_chunks(actual, expected, max_elements=max_chunk_elements):
        actual_f = actual_chunk.detach().to(torch.float32)
        expected_f = expected_chunk.detach().to(torch.float32)
        numerator += float((actual_f * expected_f).sum().item())
        denominator += float((actual_f * actual_f + expected_f * expected_f).sum().item())
        del actual_f, expected_f
    if denominator == 0.0:
        return 0.0
    return float(1.0 - (2.0 * numerator / denominator))


def deepseek_calc_diff_stream(
    actual_expected_pairs,
    *,
    require_deepgemm: bool = True,
) -> float:
    """Compute DeepSeek calc_diff over streamed tensor pairs.

    This is the same metric as `deep_gemm.testing.calc_diff`, accumulated over
    chunks so large MoE Wgrad references do not need to exist as one full
    `[experts, M, N]` tensor.
    """
    if require_deepgemm:
        try:
            import deep_gemm.testing  # noqa: F401
        except Exception as exc:
            raise RuntimeError("DeepGEMM calc_diff is required but not importable") from exc

    max_chunk_elements = int(os.environ.get("SM90_WGRAD_CALC_DIFF_CHUNK_ELEMENTS", "8388608"))
    numerator = 0.0
    denominator = 0.0
    with torch.no_grad():
        for actual, expected in actual_expected_pairs:
            for actual_chunk, expected_chunk in _tensor_chunks(actual, expected, max_elements=max_chunk_elements):
                actual_f = actual_chunk.detach().to(torch.float32)
                expected_f = expected_chunk.detach().to(torch.float32)
                numerator += float((actual_f * expected_f).sum().item())
                denominator += float((actual_f * actual_f + expected_f * expected_f).sum().item())
                del actual_f, expected_f
            del actual, expected
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if denominator == 0.0:
        return 0.0
    return float(1.0 - (2.0 * numerator / denominator))


def scaled_wgrad_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    require_deepgemm_calc_diff: bool = True,
) -> dict[str, float]:
    return {
        "deepseek_calc_diff": deepseek_calc_diff(
            actual,
            expected,
            require_deepgemm=require_deepgemm_calc_diff,
        ),
    }


def assert_scaled_wgrad_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    deepseek_diff_limit: float = 1.0e-3,
    require_deepgemm_calc_diff: bool = True,
) -> dict[str, float]:
    stats = scaled_wgrad_stats(
        actual,
        expected,
        require_deepgemm_calc_diff=require_deepgemm_calc_diff,
    )
    _ = out_dtype
    stats["deepseek_diff_limit"] = float(deepseek_diff_limit)
    if stats["deepseek_calc_diff"] > float(deepseek_diff_limit):
        raise RuntimeError(f"custom scaled FP8 Wgrad DeepSeek calc_diff check failed: {stats}")
    return stats
