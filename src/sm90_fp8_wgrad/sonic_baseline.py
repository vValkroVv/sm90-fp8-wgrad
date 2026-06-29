from __future__ import annotations

import torch


def make_sonic_outputs(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return QuACK/Sonic BF16 Wgrad output buffers in benchmark layout.

    The public Sonic custom op wraps QuACK GEMM with a legacy `[H, I, E]`
    storage contract. Recent QuACK wheels validate the actual output view stride
    more strictly, so this benchmark calls the same BF16 QuACK GEMM directly and
    keeps the down output in `[E, H, I]` and the gate/up output in QuACK's
    natural `[E, H, 2I]` layout. The benchmark transposes gate/up only for the
    out-of-band correctness comparison.
    """
    dw2 = torch.empty((int(experts), int(hidden), int(intermediate)), device=device, dtype=dtype)
    dw1 = torch.empty((int(experts), int(hidden), int(2 * intermediate)), device=device, dtype=dtype)
    return dw2, dw1


def run_sonic_down(
    *,
    dout: torch.Tensor,
    a_prime: torch.Tensor,
    dw2: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
) -> None:
    from quack.gemm_interface import gemm

    gemm(
        dout.T,
        a_prime,
        out=dw2,
        cu_seqlens_k=expert_frequency_offset,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
    )


def run_sonic_gate_up(
    *,
    x: torch.Tensor,
    dh: torch.Tensor,
    dw1: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
) -> None:
    from quack.gemm_interface import gemm

    gemm(
        x.T,
        dh,
        out=dw1,
        cu_seqlens_k=expert_frequency_offset,
        A_idx=x_gather_idx,
        batch_idx_permute=None,
        dynamic_scheduler=False,
    )
