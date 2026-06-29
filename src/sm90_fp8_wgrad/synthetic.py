from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
from typing import Literal

import torch

from .block_meta import WgradScaleBlockMeta, build_wgrad_scale_block_meta
from .checks import scaled_wgrad_error
from .interface import fp8_grouped_wgrad_sm90
from .layout_mode import LAYOUT_MODE_PADDED
from .quant import Fp8GroupedWgradOperand, quantize_grouped_wgrad_operand_128x1


RouteMode = Literal["balanced", "skewed"]


@dataclass
class SyntheticWgradInputs:
    tokens: int
    experts: int
    top_k: int
    hidden: int
    intermediate: int
    dtype: torch.dtype
    route: str
    counts: list[int]
    cu_seqlens_k: torch.Tensor
    x_gather_idx: torch.Tensor
    x_bf16: torch.Tensor
    dout_bf16: torch.Tensor
    hidden_grouped_bf16: torch.Tensor
    grad_gate_up_grouped_bf16: torch.Tensor
    grad_y_grouped_bf16: torch.Tensor
    x_grouped_bf16: torch.Tensor
    meta: WgradScaleBlockMeta
    grad_y_q: Fp8GroupedWgradOperand | None
    hidden_q: Fp8GroupedWgradOperand | None
    grad_gate_up_q: Fp8GroupedWgradOperand | None
    x_q: Fp8GroupedWgradOperand | None
    down_out_bf16: torch.Tensor | None
    gate_up_out_bf16: torch.Tensor | None

    @property
    def valid_rows(self) -> int:
        return int(self.tokens) * int(self.top_k)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise RuntimeError(f"unsupported dtype {name!r}")


def load_counts_file(path: str | Path) -> list[int]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("counts", data.get("expert_counts"))
        if not isinstance(data, list):
            raise RuntimeError("counts JSON must be a list or contain a 'counts' list")
        return [int(v) for v in data]

    counts: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) == 1:
                counts.append(int(row[0]))
            else:
                counts.append(int(row[-1]))
    if not counts:
        raise RuntimeError(f"no counts found in {path}")
    return counts


def make_expert_counts(
    *,
    tokens: int,
    top_k: int,
    experts: int,
    route: str,
    seed: int,
    counts_file: str | None = None,
) -> list[int]:
    valid_rows = int(tokens) * int(top_k)
    if counts_file:
        counts = load_counts_file(counts_file)
        if len(counts) != int(experts):
            raise RuntimeError(f"counts file has {len(counts)} experts, expected {experts}")
        if sum(counts) != valid_rows:
            raise RuntimeError(f"counts sum {sum(counts)} does not match tokens*top_k {valid_rows}")
        return counts

    if str(route) == "balanced":
        base = valid_rows // int(experts)
        rem = valid_rows % int(experts)
        return [base + (1 if e < rem else 0) for e in range(int(experts))]

    if str(route) != "skewed":
        raise RuntimeError("route must be 'balanced' or 'skewed'")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    ranks = torch.arange(1, int(experts) + 1, dtype=torch.float64)
    probs = 1.0 / torch.pow(ranks, 1.15)
    probs = probs / probs.sum()
    perm = torch.randperm(int(experts), generator=gen)
    probs = probs[perm]
    counts_t = torch.multinomial(probs, valid_rows, replacement=True, generator=gen).bincount(minlength=int(experts))
    return [int(v) for v in counts_t.tolist()]


def counts_to_cu_seqlens(counts: list[int], *, device: torch.device) -> torch.Tensor:
    cu = torch.empty((len(counts) + 1,), dtype=torch.int32, device=device)
    cu[0] = 0
    if counts:
        cu[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32, device=device), dim=0)
    return cu.contiguous()


def make_x_gather_idx(
    *,
    counts: list[int],
    tokens: int,
    device: torch.device,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for expert, count in enumerate(counts):
        if int(count) <= 0:
            continue
        rows = torch.arange(int(count), dtype=torch.int64)
        # Offset each expert so repeated token ids are deterministic but not all
        # experts read the same token sequence.
        parts.append(torch.remainder(rows + int(expert) * 7919, int(tokens)))
    if parts:
        idx_cpu = torch.cat(parts, dim=0).to(torch.int32)
    else:
        idx_cpu = torch.empty((0,), dtype=torch.int32)
    return idx_cpu.to(device=device, dtype=torch.int32).contiguous()


def source_bf16_wgrad_ref(
    lhs_src: torch.Tensor,
    rhs_src: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    cu = cu_seqlens_k.detach().cpu().tolist()
    experts = len(cu) - 1
    m = int(lhs_src.shape[1])
    n = int(rhs_src.shape[1])
    out = torch.zeros((experts, m, n), device=lhs_src.device, dtype=torch.float32)
    for expert in range(experts):
        start = int(cu[expert])
        end = int(cu[expert + 1])
        if end > start:
            out[expert] = lhs_src[start:end].float().T @ rhs_src[start:end].float()
    return out.to(out_dtype)


def route_stats(counts: list[int], meta: WgradScaleBlockMeta) -> dict[str, int | float | str]:
    sorted_counts = sorted(int(v) for v in counts)
    median = sorted_counts[len(sorted_counts) // 2] if sorted_counts else 0
    return {
        "experts": len(counts),
        "valid_rows": int(sum(counts)),
        "padded_rows": int(meta.padded_rows),
        "wgmma_rows": int(meta.wgmma_rows),
        "tma_rows": int(meta.tma_rows),
        "total_blocks": int(meta.total_blocks),
        "min_rows_per_expert": int(min(sorted_counts)) if sorted_counts else 0,
        "median_rows_per_expert": int(median),
        "max_rows_per_expert": int(max(sorted_counts)) if sorted_counts else 0,
        "padding_overhead_ratio": float(meta.padded_rows) / float(max(1, sum(counts))),
        "packed_tail_tma_mode": str(meta.packed_tail_tma_mode),
    }


def make_synthetic_wgrad_inputs(
    *,
    tokens: int,
    experts: int = 128,
    top_k: int = 8,
    hidden: int = 2048,
    intermediate: int = 768,
    route: str = "skewed",
    seed: int = 1234,
    counts_file: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    layout_mode: str = LAYOUT_MODE_PADDED,
    build_fp8_operands: bool = True,
    allocate_custom_outputs: bool = True,
) -> SyntheticWgradInputs:
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to build Wgrad benchmark inputs")
        device = torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA device is required")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")

    # The launch path resolves the layout mode from this environment variable.
    # Keep it aligned with the operand layout built below.
    os.environ["KERNEL_LAB_WGRAD_LAYOUT_MODE"] = str(layout_mode)

    counts = make_expert_counts(
        tokens=int(tokens),
        top_k=int(top_k),
        experts=int(experts),
        route=str(route),
        seed=int(seed),
        counts_file=counts_file,
    )
    cu = counts_to_cu_seqlens(counts, device=device)
    x_gather_idx = make_x_gather_idx(counts=counts, tokens=int(tokens), device=device)
    meta = build_wgrad_scale_block_meta(cu, block_k=128)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 17)
    x = torch.randn((int(tokens), int(hidden)), device=device, dtype=dtype, generator=gen)
    dout = torch.randn((int(tokens), int(hidden)), device=device, dtype=dtype, generator=gen)
    hidden_grouped = torch.randn((int(meta.valid_rows), int(intermediate)), device=device, dtype=dtype, generator=gen)
    grad_gate_up_grouped = torch.randn(
        (int(meta.valid_rows), int(2 * intermediate)),
        device=device,
        dtype=dtype,
        generator=gen,
    )
    grad_y_grouped = dout.index_select(0, x_gather_idx.to(torch.long)).contiguous()
    x_grouped = x.index_select(0, x_gather_idx.to(torch.long)).contiguous()

    if build_fp8_operands:
        grad_y_q = quantize_grouped_wgrad_operand_128x1(grad_y_grouped, meta, layout_mode=layout_mode)
        hidden_q = quantize_grouped_wgrad_operand_128x1(hidden_grouped, meta, layout_mode=layout_mode)
        grad_gate_up_q = quantize_grouped_wgrad_operand_128x1(grad_gate_up_grouped, meta, layout_mode=layout_mode)
        x_q = quantize_grouped_wgrad_operand_128x1(x_grouped, meta, layout_mode=layout_mode)
    else:
        grad_y_q = None
        hidden_q = None
        grad_gate_up_q = None
        x_q = None

    if allocate_custom_outputs:
        down_out = torch.empty((int(experts), int(hidden), int(intermediate)), device=device, dtype=torch.bfloat16)
        gate_up_out = torch.empty((int(experts), int(2 * intermediate), int(hidden)), device=device, dtype=torch.bfloat16)
    else:
        down_out = None
        gate_up_out = None

    return SyntheticWgradInputs(
        tokens=int(tokens),
        experts=int(experts),
        top_k=int(top_k),
        hidden=int(hidden),
        intermediate=int(intermediate),
        dtype=dtype,
        route=str(route),
        counts=counts,
        cu_seqlens_k=cu,
        x_gather_idx=x_gather_idx,
        x_bf16=x,
        dout_bf16=dout,
        hidden_grouped_bf16=hidden_grouped,
        grad_gate_up_grouped_bf16=grad_gate_up_grouped,
        grad_y_grouped_bf16=grad_y_grouped,
        x_grouped_bf16=x_grouped,
        meta=meta,
        grad_y_q=grad_y_q,
        hidden_q=hidden_q,
        grad_gate_up_q=grad_gate_up_q,
        x_q=x_q,
        down_out_bf16=down_out,
        gate_up_out_bf16=gate_up_out,
    )


def run_custom_down(inp: SyntheticWgradInputs, *, tuned_config: str = "default") -> None:
    if inp.grad_y_q is None or inp.hidden_q is None or inp.down_out_bf16 is None:
        raise RuntimeError("custom down requires FP8 operands and a down output buffer")
    fp8_grouped_wgrad_sm90(inp.grad_y_q, inp.hidden_q, inp.down_out_bf16, inp.meta, tuned_config=tuned_config)


def run_custom_gate_up(inp: SyntheticWgradInputs, *, tuned_config: str = "default") -> None:
    if inp.grad_gate_up_q is None or inp.x_q is None or inp.gate_up_out_bf16 is None:
        raise RuntimeError("custom gate/up requires FP8 operands and a gate/up output buffer")
    fp8_grouped_wgrad_sm90(inp.grad_gate_up_q, inp.x_q, inp.gate_up_out_bf16, inp.meta, tuned_config=tuned_config)


def run_custom_total(inp: SyntheticWgradInputs, *, tuned_config: str = "default") -> None:
    run_custom_down(inp, tuned_config=tuned_config)
    run_custom_gate_up(inp, tuned_config=tuned_config)


def _finite_chunks(tensor: torch.Tensor, *, max_elements: int):
    if int(tensor.numel()) <= int(max_elements) or tensor.ndim == 0:
        yield tensor
        return

    elems_per_index = max(1, int(tensor[0].numel()))
    step = max(1, int(max_elements) // elems_per_index)
    for start in range(0, int(tensor.shape[0]), step):
        stop = min(start + step, int(tensor.shape[0]))
        chunk = tensor[start:stop]
        if int(chunk.numel()) > int(max_elements) and chunk.ndim > 1 and int(chunk.shape[0]) == 1:
            yield from _finite_chunks(chunk[0], max_elements=max_elements)
        else:
            yield chunk


def finite_status(tensors: dict[str, torch.Tensor]) -> dict[str, bool]:
    max_elements = int(os.environ.get("SM90_WGRAD_FINITE_CHECK_CHUNK_ELEMENTS", "8388608"))
    out: dict[str, bool] = {}
    with torch.no_grad():
        for name, tensor in tensors.items():
            ok = True
            for chunk in _finite_chunks(tensor.detach(), max_elements=max_elements):
                if not bool(torch.isfinite(chunk).all().item()):
                    ok = False
                    break
            out[name] = ok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return out


def error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    max_abs, max_rel, max_abs_ref = scaled_wgrad_error(actual, expected)
    return {"max_abs": max_abs, "max_rel": max_rel, "max_abs_ref": max_abs_ref}
