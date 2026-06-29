from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .block_meta import WgradScaleBlockMeta
from .quant import Fp8GroupedWgradOperand


@dataclass(frozen=True)
class Fp8Quantized:
    data: torch.Tensor
    scales: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data, self.scales


def load_deepgemm_wgrad_api() -> tuple[Any, str]:
    import deep_gemm

    name = "k_grouped_fp8_gemm_nt_contiguous"
    fn = getattr(deep_gemm, name, None)
    if fn is None:
        raise RuntimeError(f"DeepGEMM {name} is required for the SM90 Wgrad baseline")
    return fn, name


def padded_ks_from_meta(meta: WgradScaleBlockMeta) -> tuple[list[int], torch.Tensor]:
    cu = meta.padded_cu_seqlens_k
    ks_tensor = (cu[1:] - cu[:-1]).to(device=cu.device, dtype=torch.int32).contiguous()
    ks_host = [int(v) for v in ks_tensor.detach().cpu().tolist()]
    return ks_host, ks_tensor


def to_deepgemm_k_major_operand(
    operand: Fp8GroupedWgradOperand,
    *,
    ks_host: Sequence[int],
) -> Fp8Quantized:
    """Convert padded logical [K, N] FP8 rows to DeepGEMM flat K-major layout."""
    # DeepGEMM uses its own K-major flattening contract. Use the logical padded
    # data view, not the custom kernel's launch-only packed view.
    if operand.data.ndim != 2:
        raise RuntimeError(f"expected operand.data [K,N], got {tuple(operand.data.shape)}")
    if operand.scales.ndim != 2:
        raise RuntimeError(f"expected operand.scales [K_blocks,N], got {tuple(operand.scales.shape)}")

    total_k, cols = map(int, operand.data.shape)
    if sum(int(k) for k in ks_host) != total_k:
        raise RuntimeError(f"ks sum {sum(int(k) for k in ks_host)} does not match operand rows {total_k}")

    flat_parts: list[torch.Tensor] = []
    cursor = 0
    for k in ks_host:
        k_i = int(k)
        if k_i > 0:
            flat_parts.append(operand.data[cursor : cursor + k_i].transpose(0, 1).contiguous().reshape(-1))
        cursor += k_i
    if flat_parts:
        flat = torch.cat(flat_parts, dim=0)
    else:
        flat = torch.empty((0,), device=operand.data.device, dtype=operand.data.dtype)

    scales = operand.scales.transpose(0, 1).contiguous()
    if int(scales.shape[0]) != cols:
        raise RuntimeError("DeepGEMM scale layout conversion failed")
    return Fp8Quantized(flat, scales)


def run_deepgemm_wgrad(
    *,
    lhs: Fp8Quantized,
    rhs: Fp8Quantized,
    out: torch.Tensor,
    ks_host: Sequence[int],
    ks_tensor: torch.Tensor,
    zero_c: torch.Tensor,
) -> None:
    fn, _name = load_deepgemm_wgrad_api()
    fn(
        lhs.as_tuple(),
        rhs.as_tuple(),
        out,
        [int(v) for v in ks_host],
        ks_tensor.to(device=out.device, dtype=torch.int32).contiguous(),
        zero_c,
        (1, 1, 128),
        "mn",
    )
