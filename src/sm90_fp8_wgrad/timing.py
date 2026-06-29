from __future__ import annotations

from dataclasses import dataclass, asdict
import statistics
from typing import Callable

import torch


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    p95_ms: float
    std_ms: float
    repeats: list[float]

    def to_dict(self) -> dict[str, float | list[float]]:
        return asdict(self)


def summarize_ms(values: list[float]) -> TimingStats:
    if not values:
        raise RuntimeError("cannot summarize an empty timing list")
    ordered = sorted(float(v) for v in values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((len(ordered) - 1) * float(p)))
        return ordered[max(0, min(idx, len(ordered) - 1))]

    return TimingStats(
        mean_ms=float(statistics.mean(ordered)),
        median_ms=float(statistics.median(ordered)),
        min_ms=float(min(ordered)),
        p90_ms=float(percentile(0.90)),
        p95_ms=float(percentile(0.95)),
        std_ms=float(statistics.pstdev(ordered)) if len(ordered) > 1 else 0.0,
        repeats=ordered,
    )


def time_cuda(
    fn: Callable[[], None],
    *,
    warmup_iters: int,
    active_iters: int,
    repeat: int,
) -> TimingStats:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for timing")
    if int(active_iters) <= 0:
        raise RuntimeError("active_iters must be positive")
    if int(repeat) <= 0:
        raise RuntimeError("repeat must be positive")

    per_repeat: list[float] = []
    for _ in range(int(repeat)):
        for _warmup in range(int(warmup_iters)):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _active in range(int(active_iters)):
            fn()
        end.record()
        end.synchronize()
        per_repeat.append(float(start.elapsed_time(end)) / float(active_iters))
    return summarize_ms(per_repeat)


def valid_wgrad_flops(valid_rows: int, hidden: int, intermediate: int) -> tuple[float, float, float]:
    down = 2.0 * float(valid_rows) * float(hidden) * float(intermediate)
    gate_up = 2.0 * float(valid_rows) * float(2 * intermediate) * float(hidden)
    return down, gate_up, down + gate_up


def tflops(flops: float, ms: float) -> float:
    if float(ms) <= 0.0:
        return 0.0
    return float(flops) / (float(ms) * 1.0e-3) / 1.0e12
