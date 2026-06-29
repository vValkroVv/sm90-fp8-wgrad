from __future__ import annotations

import os


LAYOUT_MODE_PADDED = "padded"
LAYOUT_MODE_PACKED = "packed"
WGRAD_LAYOUT_MODES = (LAYOUT_MODE_PADDED, LAYOUT_MODE_PACKED)

PACKED_TAIL_TMA_MASKED128 = "masked128"
PACKED_TAIL_TMA_ADAPTIVE64 = "adaptive64"
PACKED_TAIL_TMA_ADAPTIVE32_64_128 = "adaptive32_64_128"
PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE = "adaptive32_64_twophase"
PACKED_TAIL_TMA_MODES = (
    PACKED_TAIL_TMA_MASKED128,
    PACKED_TAIL_TMA_ADAPTIVE64,
    PACKED_TAIL_TMA_ADAPTIVE32_64_128,
    PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
)

KERNEL_DEBUG_TMA_KMAJOR_128_DRAIN = "tma_kmajor_128_drain"
KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN = "tma_kmajor_adaptive64_drain"
PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES = (
    KERNEL_DEBUG_TMA_KMAJOR_128_DRAIN,
    KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN,
)
PACKED_ADAPTIVE64_TMA_KERNEL_DEBUG_MODES = (
    KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN,
    "unscaled",
)
PACKED_ADAPTIVE64_SCALED_TMA_KERNEL_DEBUG_MODES = ("normal", "scale_staged2", "rhs_scale_reuse")

PACKED_SUPPORTED_KERNEL_DEBUG_MODES = ("normal", "scale_staged2", "rhs_scale_reuse")
PACKED_DEBUG_BISECT_KERNEL_DEBUG_MODES = (
    "tma_only",
    *PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES,
    "unscaled",
    "coord_m",
    "coord_n",
    "scale_only",
    "normal",
    "scale_staged2",
    "rhs_scale_reuse",
)


def resolve_wgrad_layout_mode(value: str | None = None, *, default: str = LAYOUT_MODE_PADDED) -> str:
    if value is None:
        value = os.environ.get("KERNEL_LAB_WGRAD_LAYOUT_MODE", default)
    normalized = str(value).strip().lower()
    if normalized not in WGRAD_LAYOUT_MODES:
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_LAYOUT_MODE must be one of "
            f"{WGRAD_LAYOUT_MODES}; got {value!r}"
        )
    return normalized


def resolve_packed_tail_tma_mode(
    value: str | None = None,
    *,
    default: str = PACKED_TAIL_TMA_MASKED128,
) -> str:
    if value is None:
        value = os.environ.get("KERNEL_LAB_WGRAD_PACKED_TAIL_TMA", default)
    normalized = str(value).strip().lower()
    if normalized not in PACKED_TAIL_TMA_MODES:
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_PACKED_TAIL_TMA must be one of "
            f"{PACKED_TAIL_TMA_MODES}; got {value!r}"
        )
    return normalized


def packed_layout_supports_kernel_mode(kernel_debug_mode: str, *, include_debug_bisect: bool = False) -> bool:
    normalized = str(kernel_debug_mode).strip().lower()
    allowed = (
        PACKED_DEBUG_BISECT_KERNEL_DEBUG_MODES
        if include_debug_bisect
        else PACKED_SUPPORTED_KERNEL_DEBUG_MODES
    )
    return normalized in allowed
