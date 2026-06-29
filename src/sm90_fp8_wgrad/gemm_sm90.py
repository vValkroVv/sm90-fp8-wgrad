from functools import partial
import hashlib
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum
import torch

from quack import copy_utils
import quack.cache_utils as quack_cache_utils
import quack.layout_utils as layout_utils
import quack.sm90_utils as quack_sm90_utils
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters, torch2cute_dtype_map
from quack.gemm_config import GemmConfig
from quack.gemm_base import NamedBarrierGemm
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90
from quack.gemm_tvm_ffi_utils import (
    make_fake_scheduler_args,
    make_fake_varlen_args,
    make_scheduler_args,
    make_varlen_args,
)
from quack.pipeline import (
    PipelineAsync as QuackPipelineAsync,
    PipelineTmaAsync as QuackPipelineTmaAsync,
    PipelineTmaCpAsync,
    make_pipeline_state,
)
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments, VarlenManager

from .layout_mode import (
    PACKED_ADAPTIVE64_SCALED_TMA_KERNEL_DEBUG_MODES,
    PACKED_ADAPTIVE64_TMA_KERNEL_DEBUG_MODES,
    KERNEL_DEBUG_TMA_KMAJOR_128_DRAIN,
    KERNEL_DEBUG_TMA_KMAJOR_ADAPTIVE64_DRAIN,
    LAYOUT_MODE_PACKED,
    LAYOUT_MODE_PADDED,
    PACKED_TAIL_TMA_ADAPTIVE32_64_128,
    PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
    PACKED_TAIL_TMA_ADAPTIVE64,
    PACKED_DEBUG_BISECT_KERNEL_DEBUG_MODES,
    PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES,
    PACKED_SUPPORTED_KERNEL_DEBUG_MODES,
    packed_layout_supports_kernel_mode,
    resolve_packed_tail_tma_mode,
    resolve_wgrad_layout_mode,
)


def _patch_torch2cute_fp8_dtype_map() -> None:
    """Keep this repo working with QuACK wheels that predate FP8 dtype mapping."""
    if hasattr(torch, "float8_e4m3fn") and hasattr(cutlass, "Float8E4M3FN"):
        torch2cute_dtype_map.setdefault(torch.float8_e4m3fn, cutlass.Float8E4M3FN)
    if hasattr(torch, "float8_e5m2") and hasattr(cutlass, "Float8E5M2"):
        torch2cute_dtype_map.setdefault(torch.float8_e5m2, cutlass.Float8E5M2)


_patch_torch2cute_fp8_dtype_map()


def _torch_to_cute_dtype(dtype: torch.dtype) -> Type[cutlass.Numeric]:
    try:
        return torch2cute_dtype_map[dtype]
    except KeyError as exc:
        if dtype is getattr(torch, "float8_e4m3fn", None) and hasattr(cutlass, "Float8E4M3FN"):
            return cutlass.Float8E4M3FN
        if dtype is getattr(torch, "float8_e5m2", None) and hasattr(cutlass, "Float8E5M2"):
            return cutlass.Float8E5M2
        supported = ", ".join(str(key) for key in sorted(torch2cute_dtype_map, key=str))
        raise RuntimeError(
            f"Unsupported tensor dtype for CUTLASS/CuTe codegen: {dtype}. "
            f"Known torch dtypes: {supported}"
        ) from exc


_CUSTOM_SOURCE_DIR = Path(__file__).resolve().parent
if all(Path(path).resolve() != _CUSTOM_SOURCE_DIR for path in quack_cache_utils.EXTRA_SOURCE_DIRS):
    quack_cache_utils.EXTRA_SOURCE_DIRS.append(_CUSTOM_SOURCE_DIR)
    cache_clear = getattr(quack_cache_utils._compute_source_fingerprint, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _local_source_fingerprint() -> str:
    h = hashlib.sha256()
    for src in sorted(_CUSTOM_SOURCE_DIR.rglob("*.py")):
        if not src.is_file():
            continue
        content = src.read_bytes()
        h.update(src.relative_to(_CUSTOM_SOURCE_DIR).as_posix().encode())
        h.update(len(content).to_bytes(8, "little"))
        h.update(content)
    return h.hexdigest()


_CUSTOM_SOURCE_FINGERPRINT = _local_source_fingerprint()

SCALE_SMEM_PAD = 8


def _debug_enabled() -> bool:
    return os.environ.get("KERNEL_LAB_WGRAD_DEBUG", "0") == "1"


def _debug_log(msg: str) -> None:
    if _debug_enabled():
        print(f"[quack_fp8_wgrad.gemm_sm90] {msg}", flush=True)


def _host_validation_enabled() -> bool:
    # Host validation intentionally synchronizes CUDA tensors. Keep it out of
    # timed Wgrad sections unless explicitly debugging the Python contract.
    return os.environ.get("KERNEL_LAB_WGRAD_HOST_VALIDATE", "0") == "1" or _debug_enabled()


def _parse_bool_env(name: str, *, default: bool | None) -> bool | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise RuntimeError(f"{name} must be 0/1, true/false, yes/no, or on/off; got {value!r}")


_VALID_KERNEL_DEBUG_MODES = (
    "normal",
    "epilogue_only",
    "tma_only",
    *PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES,
    "no_epilogue",
    "unscaled",
    "scale_only",
    "coord_m",
    "coord_n",
    "raw_scaled",
    "vector_scaled",
    "staged_scale",
    "staged_scale_reg",
    "staged_scale_smem",
    "staged_scale_cached",
    "masked_tail",
    "masked_tail_last_only",
    "masked_tail_early_release",
    "scale_contract_g16",
    "scale_contract_g32",
    "tail64",
    "tail32",
    "scalar_scaled",
    "early_release_scaled",
    "staged_early_release_scaled",
    "staged_scale_early_release",
    "pipelined_scaled",
    "scale_staged2",
    "rhs_scale_reuse",
    "scale_product_prefetch",
    "scale_overlap_pingpong_smallacc",
    "scale_overlap_dbacc_smallm",
    "scale_overlap_coop128_sidecar",
    "scale_sidecar_no_overlap",
    "scale_overlap_wg_pingpong",
)


def _is_kmajor_tma_drain_mode(kernel_debug_mode: str) -> bool:
    return str(kernel_debug_mode).strip().lower() in PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES


def _is_tma_drain_mode(kernel_debug_mode: str) -> bool:
    normalized = str(kernel_debug_mode).strip().lower()
    return normalized == "tma_only" or normalized in PACKED_KMAJOR_TMA_DRAIN_KERNEL_DEBUG_MODES


def _uses_adaptive64_tma_path(
    kernel_debug_mode: str,
    *,
    packed_mode: bool,
    packed_tail_tma_mode: str,
) -> bool:
    normalized = str(kernel_debug_mode).strip().lower()
    if normalized in PACKED_ADAPTIVE64_SCALED_TMA_KERNEL_DEBUG_MODES:
        return bool(packed_mode) and str(packed_tail_tma_mode) == PACKED_TAIL_TMA_ADAPTIVE64
    return (
        bool(packed_mode)
        and str(packed_tail_tma_mode)
        in (
            PACKED_TAIL_TMA_ADAPTIVE64,
            PACKED_TAIL_TMA_ADAPTIVE32_64_128,
            PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
        )
        and normalized in PACKED_ADAPTIVE64_TMA_KERNEL_DEBUG_MODES
    )


def _uses_adaptive32_tma_path(packed_tail_tma_mode: str) -> bool:
    return str(packed_tail_tma_mode) in (
        PACKED_TAIL_TMA_ADAPTIVE32_64_128,
        PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE,
    )


def _uses_adaptive_twophase_tma_path(packed_tail_tma_mode: str) -> bool:
    return str(packed_tail_tma_mode) == PACKED_TAIL_TMA_ADAPTIVE32_64_TWOPHASE


ADAPTIVE64_UNSCALED_CONSUMER_EXACT32 = "exact32"
ADAPTIVE64_UNSCALED_CONSUMER_PHYSICAL = "physical"
ADAPTIVE64_UNSCALED_CONSUMER_MODES = (
    ADAPTIVE64_UNSCALED_CONSUMER_EXACT32,
    ADAPTIVE64_UNSCALED_CONSUMER_PHYSICAL,
)
ADAPTIVE64_K64_DESC_PREFETCH_TAIL = "tail"
ADAPTIVE64_K64_DESC_PREFETCH_NONE = "none"
ADAPTIVE64_K64_DESC_PREFETCH_STARTUP = "startup"
ADAPTIVE64_K64_DESC_PREFETCH_MODES = (
    ADAPTIVE64_K64_DESC_PREFETCH_TAIL,
    ADAPTIVE64_K64_DESC_PREFETCH_NONE,
    ADAPTIVE64_K64_DESC_PREFETCH_STARTUP,
)


def _resolve_adaptive64_unscaled_consumer(value: str | None = None) -> str:
    if value is None:
        value = os.environ.get(
            "KERNEL_LAB_WGRAD_ADAPTIVE64_UNSCALED_CONSUMER",
            ADAPTIVE64_UNSCALED_CONSUMER_PHYSICAL,
        )
    normalized = str(value).strip().lower()
    if normalized not in ADAPTIVE64_UNSCALED_CONSUMER_MODES:
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_ADAPTIVE64_UNSCALED_CONSUMER must be one of "
            f"{ADAPTIVE64_UNSCALED_CONSUMER_MODES}; got {value!r}"
        )
    return normalized


def _resolve_adaptive64_k64_desc_prefetch(value: str | None = None) -> str:
    if value is None:
        value = os.environ.get(
            "KERNEL_LAB_WGRAD_ADAPTIVE64_K64_DESC_PREFETCH",
            ADAPTIVE64_K64_DESC_PREFETCH_TAIL,
        )
    normalized = str(value).strip().lower()
    if normalized not in ADAPTIVE64_K64_DESC_PREFETCH_MODES:
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_ADAPTIVE64_K64_DESC_PREFETCH must be one of "
            f"{ADAPTIVE64_K64_DESC_PREFETCH_MODES}; got {value!r}"
        )
    return normalized


WGRAD_SM90_CONFIGS = {
    "default": GemmConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "optB_coop128_default": GemmConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "down_128x192": GemmConfig(
        tile_m=128,
        tile_n=192,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "down_128x128_pingpong": GemmConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "down_128x256": GemmConfig(
        tile_m=128,
        # This QuACK SM90 path rejects CTA N > 208. Keep the historical
        # config name for CLI compatibility, but clamp the actual tile to the
        # largest valid N so sweeps do not spend rows on guaranteed failures.
        tile_n=208,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "down_64x256": GemmConfig(
        tile_m=64,
        tile_n=208,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "down_128x192_coop": GemmConfig(
        tile_m=128,
        tile_n=192,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "down_128x256_coop": GemmConfig(
        tile_m=128,
        tile_n=208,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "down_64x192": GemmConfig(
        tile_m=64,
        tile_n=192,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "down_64x192_coop": GemmConfig(
        tile_m=64,
        tile_n=192,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "gate_64x256": GemmConfig(
        tile_m=64,
        tile_n=208,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "gate_64x192": GemmConfig(
        tile_m=64,
        tile_n=192,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "gate_128x192": GemmConfig(
        tile_m=128,
        tile_n=192,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "gate_128x256": GemmConfig(
        tile_m=128,
        tile_n=208,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "gate_128x256_coop": GemmConfig(
        tile_m=128,
        tile_n=208,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "gate_256x256_coop": GemmConfig(
        tile_m=256,
        tile_n=208,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "gate_64x256_coop": GemmConfig(
        tile_m=64,
        tile_n=208,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "gate_128x128": GemmConfig(
        tile_m=128,
        tile_n=128,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "gate_256x128": GemmConfig(
        tile_m=256,
        tile_n=128,
        tile_k=128,
        cluster_m=2,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    # Option A: native 64x128 CTA tile, non-pingpong, double-buffered raw accumulators.
    # This halves the M-side accumulator footprint versus the default 128x128 tile
    # so carrying two raw accumulators is affordable enough to test overlap.
    "optA_down_64x128_dbacc": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "optA_gate_64x128_dbacc": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    # Option B: two consumer WGs alternate per-K-block MMA/promotion.
    # Qwen3 layer-24 tile counts stay even per expert with 64x128:
    # down: (2048/64) * (768/128) = 192, gate/up: (1536/64) * (2048/128) = 384.
    "optB_down_64x128_wg_pingpong": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "optB_gate_64x128_wg_pingpong": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=128,
        cluster_m=1,
        cluster_n=1,
        pingpong=True,
        is_dynamic_persistent=False,
    ),
    "tail64_64x128": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=64,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "tail64_64x256": GemmConfig(
        tile_m=64,
        tile_n=208,
        tile_k=64,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
    "tail32_64x128": GemmConfig(
        tile_m=64,
        tile_n=128,
        tile_k=32,
        cluster_m=1,
        cluster_n=1,
        pingpong=False,
        is_dynamic_persistent=False,
    ),
}


def _div_for_dtype(dtype: type[cutlass.Numeric]) -> int:
    return 128 // dtype.width


class Fp8ScaledWgradSm90(GemmDefaultSm90):
    """QuACK/CuTe SM90 FP8 grouped Wgrad with per-128-row scale promotion.

    This is a fork of QuACK's SM90 persistent GEMM path. The key difference is
    the MMA consumer loop: each 128-row K tile is reduced into a temporary FP32
    accumulator, multiplied by the matching per-column lhs/rhs FP32 scales, and
    promoted into the final FP32 accumulator before epilogue store.
    """

    def __init__(
        self,
        *args,
        debug_mode: str = "normal",
        packed_mode: bool = False,
        packed_tail_tma_mode: str = "masked128",
        adaptive64_unscaled_consumer: str = ADAPTIVE64_UNSCALED_CONSUMER_PHYSICAL,
        adaptive64_k64_desc_prefetch: str = ADAPTIVE64_K64_DESC_PREFETCH_TAIL,
        use_ragged_tma: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        debug_mode = str(debug_mode).strip().lower()
        if debug_mode == "scale_overlap_wg_pingpong":
            # Compatibility alias for the deleted pair-N Option B path.
            debug_mode = "scale_overlap_coop128_sidecar"
        self.debug_mode = debug_mode
        self.packed_mode = bool(packed_mode)
        self.packed_tail_tma_mode = resolve_packed_tail_tma_mode(packed_tail_tma_mode)
        self.adaptive64_unscaled_consumer = _resolve_adaptive64_unscaled_consumer(
            adaptive64_unscaled_consumer
        )
        self.adaptive64_k64_desc_prefetch = _resolve_adaptive64_k64_desc_prefetch(
            adaptive64_k64_desc_prefetch
        )
        self.use_adaptive64_tma = _uses_adaptive64_tma_path(
            self.debug_mode,
            packed_mode=self.packed_mode,
            packed_tail_tma_mode=self.packed_tail_tma_mode,
        )
        self.use_adaptive32_tma = self.use_adaptive64_tma and _uses_adaptive32_tma_path(
            self.packed_tail_tma_mode
        )
        self.use_adaptive_twophase_tma = self.use_adaptive64_tma and _uses_adaptive_twophase_tma_path(
            self.packed_tail_tma_mode
        )
        self.use_ragged_tma = use_ragged_tma

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ):
        if const_expr(self.use_adaptive64_tma):
            return self.make_ab_tma_pipeline_with_tx_count(
                tiled_mma,
                cluster_layout_vmnk,
                ab_pipeline_mbar_ptr,
                self.num_tma_load_bytes,
            )
        if const_expr((not self.packed_mode) or self.use_ragged_tma):
            return super().make_ab_pipeline(tiled_mma, cluster_layout_vmnk, ab_pipeline_mbar_ptr)
        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_ab_load_warps * cute.arch.WARP_SIZE,
        )
        consumer_arrive_cnt = tiled_mma.size // cute.arch.WARP_SIZE
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        return QuackPipelineAsync.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            elect_one_release=True,
            syncwarp_before_release=True,
            defer_sync=True,
        )

    def make_ab_tma_pipeline_with_tx_count(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
        tx_count: cutlass.Constexpr[int],
    ):
        producer_cnt = 1 if const_expr(not self.gather_A) else 1 + self.num_ab_load_warps * 32
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, producer_cnt)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        pipeline_cls = QuackPipelineTmaAsync if not self.gather_A else PipelineTmaCpAsync
        return pipeline_cls.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

    @classmethod
    def _compute_stages_with_scale_sidecar(
        cls,
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        d_dtype: Optional[Type[cutlass.Numeric]],
        c_dtype: Optional[Type[cutlass.Numeric]],
        epilogue_args: GemmDefaultSm90.EpilogueArguments,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int, int]:
        """Budget AB stages with the per-stage scale sidecar included.

        The default QuACK heuristic sizes A/B stages against the raw operand
        smem only. The cooperative sidecar mode adds extra FP32 stage storage,
        so reusing the default `ab_stage` can exceed the dynamic smem budget and
        fail the kernel launch with `cudaErrorInvalidValue`.
        """
        epi_stage = 4 if epi_tile[1] <= 16 else 2
        d_bytes_per_stage = cute.size(epi_tile) * d_dtype.width // 8 if d_dtype is not None else 0
        epi_bytes_per_stage = d_bytes_per_stage + cls.epi_smem_bytes_per_stage(
            epilogue_args,
            cta_tile_shape_mnk,
            epi_tile,
        )
        epi_bytes = epi_bytes_per_stage * epi_stage
        epi_c_stage = 0 if c_dtype is None else (4 if epi_tile[1] <= 16 else 2)
        if c_dtype is not None:
            epi_bytes += cute.size(epi_tile) * c_dtype.width // 8 * epi_c_stage

        a_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        )
        scale_sidecar_bytes_per_stage = (Float32.width // 8) * (
            cta_tile_shape_mnk[0] + cta_tile_shape_mnk[1] + 2 * SCALE_SMEM_PAD
        )
        mbar_helpers_bytes = 1024

        remaining_bytes = smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        total_ab_stage_bytes = ab_bytes_per_stage + scale_sidecar_bytes_per_stage
        ab_stage = remaining_bytes // total_ab_stage_bytes
        if ab_stage < 2:
            raise RuntimeError(
                "scale_overlap_coop128_sidecar does not fit the SM90 shared-memory budget "
                f"for tile={cta_tile_shape_mnk}; computed ab_stage={ab_stage}"
            )

        if epi_bytes_per_stage > 0:
            epi_stage += (remaining_bytes - total_ab_stage_bytes * ab_stage) // epi_bytes_per_stage
        return ab_stage, epi_stage, epi_c_stage

    def _setup_attributes(self, epilogue_args: GemmDefaultSm90.EpilogueArguments):
        """Override stage sizing so the cooperative sidecar mode fits in smem."""
        self._setup_tiled_mma()
        self.epi_m_major = self.resolve_epi_m_major(epilogue_args)

        self.cluster_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.epi_tile = self._compute_tile_shape_or_override(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )
        self.epi_tile_shape = cute.ceil_div(self.cta_tile_shape_mnk[:2], self.epi_tile)

        smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_{self.arch}")
        if self.debug_mode == "scale_sidecar_no_overlap":
            self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages_with_scale_sidecar(
                self.cta_tile_shape_mnk,
                self.epi_tile,
                self.a_dtype,
                self.b_dtype,
                self.d_dtype,
                self.c_dtype,
                epilogue_args,
                smem_capacity,
                self.occupancy,
            )
        else:
            self.ab_stage, self.epi_stage, self.epi_c_stage = self._compute_stages(
                self.cta_tile_shape_mnk,
                self.epi_tile,
                self.a_dtype,
                self.b_dtype,
                self.d_dtype,
                self.c_dtype,
                epilogue_args,
                smem_capacity,
                self.occupancy,
            )
        self.sched_stage = 2 if self.pingpong else 1

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_c_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.cta_tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.d_dtype,
            self.d_layout,
            self.epi_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_c_stage,
        )
        self.a_smem_layout_staged_k64 = None
        self.b_smem_layout_staged_k64 = None
        self.a_smem_layout_staged_k32 = None
        self.b_smem_layout_staged_k32 = None
        if self.use_adaptive64_tma:
            (
                self.a_smem_layout_staged_k64,
                self.b_smem_layout_staged_k64,
                _,
                _,
            ) = self._make_smem_layouts(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 64),
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                self.b_dtype,
                self.b_layout,
                self.ab_stage,
                self.d_dtype,
                self.d_layout,
                self.epi_stage,
                self.c_dtype,
                self.c_layout,
                self.epi_c_stage,
            )
        if self.use_adaptive32_tma:
            (
                self.a_smem_layout_staged_k32,
                self.b_smem_layout_staged_k32,
                _,
                _,
            ) = self._make_smem_layouts(
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 32),
                self.epi_tile,
                self.a_dtype,
                self.a_layout,
                self.b_dtype,
                self.b_layout,
                self.ab_stage,
                self.d_dtype,
                self.d_layout,
                self.epi_stage,
                self.c_dtype,
                self.c_layout,
                self.epi_c_stage,
            )

    def _make_shared_storage_type(self, epilogue_params, epi_smem_size: int):
        """Build the CTA shared-storage struct for the current debug mode.

        CuTe DSL codegen rejects branch-local class definitions inside
        `__call__`, so the storage selection lives in a plain Python helper.
        """
        if self.debug_mode == "scale_sidecar_no_overlap":

            @cute.struct
            class SharedStorage:
                ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
                sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
                sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
                sD: cute.struct.Align[
                    cute.struct.MemRange[self.d_dtype, epi_smem_size],
                    self.buffer_align_bytes,
                ]
                epi: self.epi_get_smem_struct(epilogue_params)
                sA: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                    self.buffer_align_bytes,
                ]
                sScaleL: cute.struct.Align[
                    cute.struct.MemRange[
                        Float32,
                        self.ab_stage * (self.cta_tile_shape_mnk[0] + SCALE_SMEM_PAD),
                    ],
                    128,
                ]
                sScaleR: cute.struct.Align[
                    cute.struct.MemRange[
                        Float32,
                        self.ab_stage * (self.cta_tile_shape_mnk[1] + SCALE_SMEM_PAD),
                    ],
                    128,
                ]

            return SharedStorage

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sched_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.sched_stage * 2]
            sched_data: cute.struct.MemRange[Int32, self.sched_stage * 4]
            sD: cute.struct.Align[
                cute.struct.MemRange[self.d_dtype, epi_smem_size],
                self.buffer_align_bytes,
            ]
            epi: self.epi_get_smem_struct(epilogue_params)
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        return SharedStorage

    def make_tma_load_atoms_and_tensors_for_k(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        varlen_k: bool,
        tile_k: int,
    ):
        tma_atom_a, tma_tensor_a = None, None
        if const_expr(not self.gather_A):
            tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
                copy_utils.create_ragged_tensor_for_tma(mA, ragged_dim=1)
                if varlen_k and not self.gather_A
                else mA,
                a_smem_layout,
                (self.cta_tile_shape_mnk[0], int(tile_k)),
                self.cluster_shape_mnk[1],
            )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            copy_utils.create_ragged_tensor_for_tma(mB, ragged_dim=1) if varlen_k else mB,
            b_smem_layout,
            (self.cta_tile_shape_mnk[1], int(tile_k)),
            self.cluster_shape_mnk[0],
        )
        return tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b

    @staticmethod
    def _make_k64_prefix_smem_layout(smem_layout: cute.ComposedLayout) -> cute.ComposedLayout:
        """K=64 TMA layout that keeps the K=128 WGMMA stage addressing.

        `cute.local_tile` is a tensor/view operation, not a legal operation on a
        `ComposedLayout`. Follow the donor pattern from QuACK SM100 gather-TMA:
        rebuild the composed layout with a smaller logical K shape while keeping
        the original outer strides. This lets the K64 TMA write only the logical
        prefix rows but keeps every pipeline stage at the same shared-memory
        address that the normal K128 WGMMA fragment will read.
        """

        outer = smem_layout.outer
        k_mode_shape = outer.shape[1]
        if cute.rank(k_mode_shape) == 1:
            k64_mode_shape = 64
        else:
            k64_mode_shape = (64, *k_mode_shape[1:])
        outer_rank = cute.rank(outer.shape)
        if outer_rank == 2:
            k64_outer_shape = (outer.shape[0], k64_mode_shape)
        elif outer_rank == 3:
            k64_outer_shape = (outer.shape[0], k64_mode_shape, outer.shape[2])
        else:
            raise ValueError(f"unsupported K64 SMEM layout rank: {outer_rank}")
        return cute.make_composed_layout(
            smem_layout.inner,
            0,
            cute.make_layout(
                k64_outer_shape,
                stride=outer.stride,
            ),
        )

    @staticmethod
    def _make_compact_staged_smem_layout(
        compact_smem_layout: cute.ComposedLayout,
        k128_smem_layout: cute.ComposedLayout,
        tile_k: int,
    ) -> cute.ComposedLayout:
        """Compact short-K TMA layout with the normal K128 pipeline stage pitch.

        E1a kept the K128 WGMMA layout for the K64 TMA destination. NCU showed
        that this lowers as many tiny TMA operations. E1b instead uses the
        native compact short-K swizzle/layout for the tail copy, but keeps the K128
        stage stride so stage indices still map to the same pipeline buffers as
        the normal K128 path.
        """

        outer_short = compact_smem_layout.outer
        outer128 = k128_smem_layout.outer
        outer_rank = cute.rank(outer_short.shape)
        if outer_rank != 3:
            raise ValueError(f"expected staged K{int(tile_k)} SMEM layout rank 3, got {outer_rank}")
        if cute.rank(outer128.shape) != 3:
            raise ValueError("expected staged K128 SMEM layout rank 3")
        return cute.make_composed_layout(
            compact_smem_layout.inner,
            0,
            cute.make_layout(
                outer_short.shape,
                stride=(outer_short.stride[0], outer_short.stride[1], outer128.stride[2]),
            ),
        )

    @staticmethod
    def _make_k64_compact_staged_smem_layout(
        k64_smem_layout: cute.ComposedLayout,
        k128_smem_layout: cute.ComposedLayout,
    ) -> cute.ComposedLayout:
        return Fp8ScaledWgradSm90._make_compact_staged_smem_layout(
            k64_smem_layout,
            k128_smem_layout,
            64,
        )

    @staticmethod
    def _logical_tma_tile_bytes(
        dtype: Type[cutlass.Numeric],
        rows: int,
        cols: int,
    ) -> int:
        return int(rows) * int(cols) * (int(dtype.width) // 8)

    @cute.jit
    def __call__(
        self,
        mA_mk: cute.Tensor,
        mB_nk: cute.Tensor,
        mD_mnl: cute.Tensor,
        mC_mnl: Optional[cute.Tensor],
        epilogue_args: tuple,
        scheduler_args: TileSchedulerOptions,
        varlen_args: VarlenArguments,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
        stream: cuda.CUstream,
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        self.a_dtype = mA_mk.element_type
        self.b_dtype = mB_nk.element_type
        self.d_dtype = mD_mnl.element_type
        self.c_dtype = None
        self.a_layout = LayoutEnum.from_tensor(mA_mk)
        self.b_layout = LayoutEnum.from_tensor(mB_nk)
        self.d_layout = LayoutEnum.from_tensor(mD_mnl)
        self.c_layout = None

        if const_expr(self.a_dtype.width != 8 or self.b_dtype.width != 8):
            raise TypeError("Fp8ScaledWgradSm90 requires FP8 A and B")
        if const_expr(
            self.packed_mode
            and self.use_ragged_tma
            and self.debug_mode != "tma_only"
        ):
            # Packed launch tensors are [M_or_N, K_storage] with K-contiguous
            # storage. Keep the SM90 FP8 path on the same K-major operand mode
            # as the working padded path.
            self.a_layout = LayoutEnum.ROW_MAJOR
            self.b_layout = LayoutEnum.ROW_MAJOR
        if const_expr(
            not self.use_ragged_tma
            and self.a_layout.sm90_mma_major_mode() != warpgroup.OperandMajorMode.K
        ):
            raise TypeError("lhs operand must be K-major for SM90 FP8 WGMMA")
        if const_expr(
            not self.use_ragged_tma
            and self.b_layout.sm90_mma_major_mode() != warpgroup.OperandMajorMode.K
        ):
            raise TypeError("rhs operand must be K-major for SM90 FP8 WGMMA")
        if const_expr(varlen_args is None or varlen_args.mCuSeqlensK is None):
            raise TypeError("Fp8ScaledWgradSm90 requires varlen K metadata")
        if const_expr(mC_mnl is not None):
            raise TypeError("C/add epilogue is not implemented for FP8 Wgrad")

        self._setup_attributes(epilogue_args)

        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        a_smem_layout_staged_k64_arg = self.a_smem_layout_staged
        b_smem_layout_staged_k64_arg = self.b_smem_layout_staged
        a_smem_layout_staged_k32_arg = self.a_smem_layout_staged
        b_smem_layout_staged_k32_arg = self.b_smem_layout_staged
        a_smem_layout_k64 = None
        b_smem_layout_k64 = None
        a_smem_layout_k32 = None
        b_smem_layout_k32 = None
        if const_expr(self.use_adaptive64_tma):
            a_smem_layout_staged_k64_arg = self._make_k64_compact_staged_smem_layout(
                self.a_smem_layout_staged_k64,
                self.a_smem_layout_staged,
            )
            b_smem_layout_staged_k64_arg = self._make_k64_compact_staged_smem_layout(
                self.b_smem_layout_staged_k64,
                self.b_smem_layout_staged,
            )
            a_smem_layout_k64 = cute.slice_(a_smem_layout_staged_k64_arg, (None, None, 0))
            b_smem_layout_k64 = cute.slice_(b_smem_layout_staged_k64_arg, (None, None, 0))
            if const_expr(self.use_adaptive32_tma):
                a_smem_layout_staged_k32_arg = self._make_compact_staged_smem_layout(
                    self.a_smem_layout_staged_k32,
                    self.a_smem_layout_staged,
                    32,
                )
                b_smem_layout_staged_k32_arg = self._make_compact_staged_smem_layout(
                    self.b_smem_layout_staged_k32,
                    self.b_smem_layout_staged,
                    32,
                )
                a_smem_layout_k32 = cute.slice_(a_smem_layout_staged_k32_arg, (None, None, 0))
                b_smem_layout_k32 = cute.slice_(b_smem_layout_staged_k32_arg, (None, None, 0))
        # Packed Patch 5 uses per-expert K rounded to the SM90 FP8 WGMMA atom
        # size and stores explicit zero guard rows in kernel_data. For WGMMA
        # modes, keep the source and shared-memory operands K-major and let
        # the masked MMA path ignore the guard rows. The donor ragged TMA
        # descriptor is still useful for tma_only launch validation, but it is
        # MN-major-oriented and is not a valid data path for FP8 WGMMA.
        use_varlen_tma = self.use_ragged_tma
        if const_expr(self.packed_mode and self.debug_mode != "tma_only"):
            use_varlen_tma = False
        tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b = self.make_tma_load_atoms_and_tensors(
            mA_mk,
            mB_nk,
            a_smem_layout,
            b_smem_layout,
            use_varlen_tma,
        )
        tma_atom_a_k64, tma_tensor_a_k64, tma_atom_b_k64, tma_tensor_b_k64 = None, None, None, None
        tma_atom_a_k32, tma_tensor_a_k32, tma_atom_b_k32, tma_tensor_b_k32 = None, None, None, None
        if const_expr(self.use_adaptive64_tma):
            (
                tma_atom_a_k64,
                tma_tensor_a_k64,
                tma_atom_b_k64,
                tma_tensor_b_k64,
            ) = self.make_tma_load_atoms_and_tensors_for_k(
                mA_mk,
                mB_nk,
                a_smem_layout_k64,
                b_smem_layout_k64,
                use_varlen_tma,
                64,
            )
            if const_expr(self.use_adaptive32_tma):
                (
                    tma_atom_a_k32,
                    tma_tensor_a_k32,
                    tma_atom_b_k32,
                    tma_tensor_b_k32,
                ) = self.make_tma_load_atoms_and_tensors_for_k(
                    mA_mk,
                    mB_nk,
                    a_smem_layout_k32,
                    b_smem_layout_k32,
                    use_varlen_tma,
                    32,
                )

        if const_expr(self.use_adaptive64_tma):
            # The K64 descriptor keeps K128 stage strides so TMA lands in the
            # WGMMA-visible prefix. Barrier accounting must still use the
            # logical payload bytes, not the preserved-stride layout span.
            tma_load_bytes_128 = self._logical_tma_tile_bytes(
                self.b_dtype,
                self.cta_tile_shape_mnk[1],
                128,
            )
            tma_load_bytes_k64 = self._logical_tma_tile_bytes(
                self.b_dtype,
                self.cta_tile_shape_mnk[1],
                64,
            )
            tma_load_bytes_k32 = self._logical_tma_tile_bytes(
                self.b_dtype,
                self.cta_tile_shape_mnk[1],
                32,
            )
            if const_expr(not self.gather_A):
                tma_load_bytes_128 += self._logical_tma_tile_bytes(
                    self.a_dtype,
                    self.cta_tile_shape_mnk[0],
                    128,
                )
                tma_load_bytes_k64 += self._logical_tma_tile_bytes(
                    self.a_dtype,
                    self.cta_tile_shape_mnk[0],
                    64,
                )
                tma_load_bytes_k32 += self._logical_tma_tile_bytes(
                    self.a_dtype,
                    self.cta_tile_shape_mnk[0],
                    32,
                )
            # Full 128-row stages are the common case. Use their byte count as
            # the pipeline default so they stay on the normal producer_acquire
            # path, and apply a negative transaction delta only for short tails.
            self.num_tma_load_bytes = tma_load_bytes_128
            tail_tma_extra_tx_count_k64 = tma_load_bytes_k64 - tma_load_bytes_128
            tail_tma_extra_tx_count_k32 = tma_load_bytes_k32 - tma_load_bytes_128
        else:
            tma_load_bytes_128 = cute.size_in_bytes(self.a_dtype, a_smem_layout)
            tma_load_bytes_128 += cute.size_in_bytes(self.b_dtype, b_smem_layout)
            self.num_tma_load_bytes = tma_load_bytes_128
            tail_tma_extra_tx_count_k64 = 0
            tail_tma_extra_tx_count_k32 = 0

        tma_atom_d, tma_tensor_d, _, _ = self.make_tma_epilogue_atoms_and_tensors(
            mD_mnl,
            None,
            epilogue_args,
            False,
        )

        epilogue_params = self.epi_to_underlying_arguments(epilogue_args)
        varlen_params = VarlenManager.to_underlying_arguments(varlen_args)

        TileSchedulerCls = self.get_scheduler_class(varlen_m=False)
        tile_sched_args = self.get_scheduler_arguments(
            mA_mk,
            mB_nk,
            mD_mnl,
            scheduler_args,
            varlen_args,
            epilogue_args,
        )
        tile_sched_params = TileSchedulerCls.to_underlying_arguments(tile_sched_args)
        grid = TileSchedulerCls.get_grid_shape(
            tile_sched_params,
            scheduler_args.max_active_clusters,
        )

        epi_smem_size = cute.cosize(self.epi_smem_layout_staged)
        self.shared_storage = self._make_shared_storage_type(epilogue_params, epi_smem_size)

        self.kernel(
            self.tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_a_k64,
            tma_tensor_a_k64,
            tma_atom_b_k64,
            tma_tensor_b_k64,
            tma_atom_a_k32,
            tma_tensor_a_k32,
            tma_atom_b_k32,
            tma_tensor_b_k32,
            tma_atom_d,
            tma_tensor_d,
            epilogue_params,
            varlen_params,
            self.cluster_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            a_smem_layout_staged_k64_arg,
            b_smem_layout_staged_k64_arg,
            a_smem_layout_staged_k32_arg,
            b_smem_layout_staged_k32_arg,
            self.epi_smem_layout_staged,
            tile_sched_params,
            TileSchedulerCls,
            mLhsScales_bm,
            mRhsScales_bn,
            mBlockOffsets_l,
            mBlockRowCount_b,
            tail_tma_extra_tx_count_k64,
            tail_tma_extra_tx_count_k32,
            trace_ptr,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_a_k64: Optional[cute.CopyAtom],
        mA_mk_k64: Optional[cute.Tensor],
        tma_atom_b_k64: Optional[cute.CopyAtom],
        mB_nk_k64: Optional[cute.Tensor],
        tma_atom_a_k32: Optional[cute.CopyAtom],
        mA_mk_k32: Optional[cute.Tensor],
        tma_atom_b_k32: Optional[cute.CopyAtom],
        mB_nk_k32: Optional[cute.Tensor],
        tma_atom_d: cute.CopyAtom,
        mD_mnl: cute.Tensor,
        epilogue_params,
        varlen_params: VarlenManager.Params,
        cluster_layout_mnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        a_smem_layout_k64: cute.ComposedLayout,
        b_smem_layout_k64: cute.ComposedLayout,
        a_smem_layout_k32: cute.ComposedLayout,
        b_smem_layout_k32: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
        tail_tma_extra_tx_count_k64: cutlass.Constexpr[int],
        tail_tma_extra_tx_count_k32: cutlass.Constexpr[int],
        trace_ptr: Optional[cutlass.Int64] = None,
    ):
        from quack.trace import TraceContext

        tctx = TraceContext.create(trace_ptr)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.ab_load_warp_id:
            # E4: K64 descriptors are only used on physical K64 tails. Keep CTA
            # startup prefetch on the common A/B/D descriptors so K128-only work
            # does not pay K64 setup cost.
            for tma_atom in (tma_atom_a, tma_atom_b, tma_atom_d):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)
            if const_expr(
                self.use_adaptive64_tma
                and self.adaptive64_k64_desc_prefetch
                == ADAPTIVE64_K64_DESC_PREFETCH_STARTUP
            ):
                if const_expr(tma_atom_a_k64 is not None):
                    cpasync.prefetch_descriptor(tma_atom_a_k64)
                if const_expr(tma_atom_b_k64 is not None):
                    cpasync.prefetch_descriptor(tma_atom_b_k64)
                if const_expr(self.use_adaptive32_tma):
                    if const_expr(tma_atom_a_k32 is not None):
                        cpasync.prefetch_descriptor(tma_atom_a_k32)
                    if const_expr(tma_atom_b_k32 is not None):
                        cpasync.prefetch_descriptor(tma_atom_b_k32)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = self.make_ab_pipeline(
            tiled_mma=tiled_mma,
            cluster_layout_vmnk=cute.make_layout((1, *cluster_layout_mnk.shape)),
            ab_pipeline_mbar_ptr=storage.ab_pipeline_array_ptr.data_ptr(),
        )
        sched_pipeline = self.make_sched_pipeline(
            cluster_layout_mnk,
            sched_pipeline_mbar_ptr=storage.sched_pipeline_array_ptr.data_ptr(),
            varlen_k=True,
        )
        sched_data = storage.sched_data.get_tensor((4, self.sched_stage))

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk[:-1], is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sA_k64 = None
        sB_k64 = None
        sA_k32 = None
        sB_k32 = None
        if const_expr(self.use_adaptive64_tma):
            # Match the donor pattern used when one shared allocation has
            # separate load and MMA views: bind the K64 TMA destination from
            # its own composed layout instead of deriving a prefix view from
            # the K128 tensor. The layout keeps the K128 stage strides, so the
            # normal WGMMA path still reads the same shared-memory prefix.
            sA_k64 = storage.sA.get_tensor(
                a_smem_layout_k64.outer,
                swizzle=a_smem_layout_k64.inner,
            )
            sB_k64 = storage.sB.get_tensor(
                b_smem_layout_k64.outer,
                swizzle=b_smem_layout_k64.inner,
            )
            if const_expr(self.use_adaptive32_tma):
                sA_k32 = storage.sA.get_tensor(
                    a_smem_layout_k32.outer,
                    swizzle=a_smem_layout_k32.inner,
                )
                sB_k32 = storage.sB.get_tensor(
                    b_smem_layout_k32.outer,
                    swizzle=b_smem_layout_k32.inner,
                )
        sD = storage.sD.get_tensor(epi_smem_layout.outer, swizzle=epi_smem_layout.inner)
        if const_expr(self.debug_mode == "scale_sidecar_no_overlap"):
            sScaleL = storage.sScaleL.get_tensor(
                (self.ab_stage, self.cta_tile_shape_mnk[0] + SCALE_SMEM_PAD)
            )
            sScaleR = storage.sScaleR.get_tensor(
                (self.ab_stage, self.cta_tile_shape_mnk[1] + SCALE_SMEM_PAD)
            )
        else:
            sScaleL = cute.make_rmem_tensor((1, 1), Float32)
            sScaleR = cute.make_rmem_tensor((1, 1), Float32)
        epi_smem_tensors = self.epi_get_smem_tensors(epilogue_params, storage)

        varlen_manager = VarlenManager.create(
            varlen_params,
            len_m_static=Int32(cute.size(mA_mk, mode=[0])),
            len_k_static=Int32(cute.size(mA_mk, mode=[1])),
        )

        TileSchedulerFactory = partial(
            TileSchedulerCls.create,
            tile_sched_params,
            sched_data,
            sched_pipeline,
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk[:-1])

        if warp_idx >= self.ab_load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            if warp_idx < self.ab_load_warp_id + self.num_ab_load_warps:
                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()

                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk,
                    block_in_cluster_coord_mnk,
                    mode=1,
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk,
                    block_in_cluster_coord_mnk,
                    mode=0,
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                is_scheduler_warp = warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerFactory()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.ab_stage,
                )
                while work_tile.is_valid_tile:
                    tctx.b("tma_load")
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    mA_mk_batch = varlen_manager.offset_batch_A(mA_mk, batch_idx)
                    mB_nk_batch = varlen_manager.offset_batch_B(mB_nk, batch_idx)
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = (
                        self.adaptive_physical_stage_count(len_k)
                        if const_expr(self.use_adaptive64_tma)
                        else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    )
                    if const_expr(self.debug_mode != "epilogue_only"):
                        copy_A, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_a,
                            cta_coord=block_in_cluster_coord_mnk[1],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                            ),
                            src_tensor=cute.local_tile(
                                mA_mk_batch,
                                cute.select(self.cta_tile_shape_mnk, [0, 2]),
                                (tile_coord_mnkl[0], None),
                            ),
                            dst_tensor=sA,
                            mcast_mask=a_mcast_mask,
                        )
                        copy_B, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_b,
                            cta_coord=block_in_cluster_coord_mnk[0],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                            ),
                            src_tensor=cute.local_tile(
                                mB_nk_batch,
                                cute.select(self.cta_tile_shape_mnk, [1, 2]),
                                (tile_coord_mnkl[1], None),
                            ),
                            dst_tensor=sB,
                            mcast_mask=b_mcast_mask,
                        )
                        if const_expr(self.use_adaptive64_tma):
                            full_128_tiles = len_k // Int32(128)
                            tail_rows = len_k - full_128_tiles * Int32(128)
                            if tail_rows == 0:
                                # E3.1: keep K128-only work on the donor fast path.
                                # Do not materialize short-K tensors/copy functions unless
                                # the selected expert has a physical short tail.
                                ab_producer_state = self.load_tma(
                                    ab_pipeline,
                                    ab_producer_state,
                                    [copy_A, copy_B],
                                    full_128_tiles,
                                )
                            else:
                                # E5d/A8: short descriptors are tail-only. Keep
                                # full K128-only experts on the donor producer path.
                                if const_expr(
                                    self.adaptive64_k64_desc_prefetch
                                    == ADAPTIVE64_K64_DESC_PREFETCH_TAIL
                                ):
                                    if warp_idx == self.ab_load_warp_id:
                                        if tail_rows > Int32(32):
                                            if const_expr(tma_atom_a_k64 is not None):
                                                cpasync.prefetch_descriptor(tma_atom_a_k64)
                                            if const_expr(tma_atom_b_k64 is not None):
                                                cpasync.prefetch_descriptor(tma_atom_b_k64)
                                        if const_expr(self.use_adaptive32_tma):
                                            if tail_rows <= Int32(32) or (
                                                const_expr(self.use_adaptive_twophase_tma)
                                                and tail_rows > Int32(64)
                                                and tail_rows <= Int32(96)
                                            ):
                                                if const_expr(tma_atom_a_k32 is not None):
                                                    cpasync.prefetch_descriptor(tma_atom_a_k32)
                                                if const_expr(tma_atom_b_k32 is not None):
                                                    cpasync.prefetch_descriptor(tma_atom_b_k32)
                                mA_mk_batch_k64 = varlen_manager.offset_batch_A(mA_mk_k64, batch_idx)
                                mB_nk_batch_k64 = varlen_manager.offset_batch_B(mB_nk_k64, batch_idx)
                                copy_A_k64, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_a_k64,
                                    cta_coord=block_in_cluster_coord_mnk[1],
                                    cta_layout=cute.make_layout(
                                        cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                                    ),
                                    src_tensor=cute.local_tile(
                                        mA_mk_batch_k64,
                                        (self.cta_tile_shape_mnk[0], 64),
                                        (tile_coord_mnkl[0], None),
                                    ),
                                    dst_tensor=sA_k64,
                                    mcast_mask=a_mcast_mask,
                                )
                                copy_B_k64, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_b_k64,
                                    cta_coord=block_in_cluster_coord_mnk[0],
                                    cta_layout=cute.make_layout(
                                        cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                                    ),
                                    src_tensor=cute.local_tile(
                                        mB_nk_batch_k64,
                                        (self.cta_tile_shape_mnk[1], 64),
                                        (tile_coord_mnkl[1], None),
                                    ),
                                    dst_tensor=sB_k64,
                                    mcast_mask=b_mcast_mask,
                                )
                                if const_expr(self.use_adaptive32_tma):
                                    mA_mk_batch_k32 = varlen_manager.offset_batch_A(mA_mk_k32, batch_idx)
                                    mB_nk_batch_k32 = varlen_manager.offset_batch_B(mB_nk_k32, batch_idx)
                                    copy_A_k32, _, _ = copy_utils.tma_get_copy_fn(
                                        tma_atom_a_k32,
                                        cta_coord=block_in_cluster_coord_mnk[1],
                                        cta_layout=cute.make_layout(
                                            cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                                        ),
                                        src_tensor=cute.local_tile(
                                            mA_mk_batch_k32,
                                            (self.cta_tile_shape_mnk[0], 32),
                                            (tile_coord_mnkl[0], None),
                                        ),
                                        dst_tensor=sA_k32,
                                        mcast_mask=a_mcast_mask,
                                    )
                                    copy_B_k32, _, _ = copy_utils.tma_get_copy_fn(
                                        tma_atom_b_k32,
                                        cta_coord=block_in_cluster_coord_mnk[0],
                                        cta_layout=cute.make_layout(
                                            cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                                        ),
                                        src_tensor=cute.local_tile(
                                            mB_nk_batch_k32,
                                            (self.cta_tile_shape_mnk[1], 32),
                                            (tile_coord_mnkl[1], None),
                                        ),
                                        dst_tensor=sB_k32,
                                        mcast_mask=b_mcast_mask,
                                    )
                                    ab_producer_state = self.load_tma_kmajor_adaptive32_tail_twophase(
                                        ab_pipeline,
                                        ab_producer_state,
                                        [copy_A, copy_B],
                                        [copy_A_k64, copy_B_k64],
                                        [copy_A_k32, copy_B_k32],
                                        full_128_tiles,
                                        tail_rows,
                                        tail_tma_extra_tx_count_k64,
                                        tail_tma_extra_tx_count_k32,
                                    )
                                else:
                                    ab_producer_state = self.load_tma_kmajor_adaptive64_tail64_only(
                                        ab_pipeline,
                                        ab_producer_state,
                                        [copy_A, copy_B],
                                        [copy_A_k64, copy_B_k64],
                                        full_128_tiles,
                                        tail_tma_extra_tx_count_k64,
                                    )
                        else:
                            ab_producer_state = self.load_tma(
                                ab_pipeline,
                                ab_producer_state,
                                [copy_A, copy_B],
                                k_tile_cnt,
                            )
                    tctx.e("tma_load")
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                if warp_idx == self.ab_load_warp_id:
                    ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()

        if warp_idx < self.ab_load_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            is_tma_warp = Boolean(
                (not self.pingpong and warp_idx == 0)
                or (self.pingpong and (warp_idx == 0 or warp_idx == 4))
            )
            tidx, _, _ = cute.arch.thread_idx()
            warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
            if const_expr(self.pingpong):
                tidx = tidx % self.num_threads_per_warp_group

            if const_expr(_is_tma_drain_mode(self.debug_mode)):
                ab_read_state = make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.ab_stage,
                )
                tile_scheduler = TileSchedulerFactory()
                work_tile = tile_scheduler.initial_work_tile_info()
                if const_expr(self.pingpong):
                    if warp_idx >= 4:
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = (
                            self.adaptive_physical_stage_count(len_k)
                            if const_expr(self.use_adaptive64_tma)
                            else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        )
                        ab_read_state = self.advance_pipeline_state(ab_read_state, k_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()

                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = (
                        self.adaptive_physical_stage_count(len_k)
                        if const_expr(self.use_adaptive64_tma)
                        else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    )
                    if const_expr(self.pingpong):
                        self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    tctx.b("tma_drain")
                    ab_read_state = self.drain_ab_pipeline(
                        ab_pipeline,
                        ab_read_state,
                        k_tile_cnt,
                        warp_group_idx,
                    )
                    tctx.e("tma_drain")
                    if const_expr(not self.pingpong):
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = (
                                self.adaptive_physical_stage_count(len_k)
                                if const_expr(self.use_adaptive64_tma)
                                else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            )
                            ab_read_state = self.advance_pipeline_state(ab_read_state, k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                tctx.flush()

            if const_expr(not _is_tma_drain_mode(self.debug_mode)):
                warp_group_thread_layout = cute.make_layout(
                    self.mma_warp_groups if const_expr(not self.pingpong) else 1,
                    stride=self.num_threads_per_warp_group,
                )
                thr_mma = tiled_mma.get_slice(
                    warp_group_thread_layout(warp_group_idx if not self.pingpong else 0)
                )

                acc, tCrA, tCrB = quack_sm90_utils.partition_fragment_ABC(
                    thr_mma,
                    self.cta_tile_shape_mnk,
                    sA,
                    sB,
                )
                tCrA_k64 = tCrA
                tCrB_k64 = tCrB
                tCrA_k32 = tCrA
                tCrB_k32 = tCrB
                if const_expr(self.use_adaptive64_tma):
                    _, tCrA_k64, tCrB_k64 = quack_sm90_utils.partition_fragment_ABC(
                        thr_mma,
                        (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 64),
                        sA_k64,
                        sB_k64,
                    )
                    if const_expr(self.use_adaptive32_tma):
                        _, tCrA_k32, tCrB_k32 = quack_sm90_utils.partition_fragment_ABC(
                            thr_mma,
                            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 32),
                            sA_k32,
                            sB_k32,
                        )
                final_acc = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
                mma_fn = partial(quack_sm90_utils.gemm_w_idx, tiled_mma, acc, tCrA, tCrB)
                scale_tiled_copy_r2s, scale_tRS_rD, _ = self.epilog_smem_store_and_partition(
                    tiled_mma,
                    self.d_layout,
                    self.d_dtype,
                    sD,
                    tidx,
                )
                scale_thr_copy_r2s = scale_tiled_copy_r2s.get_slice(tidx)
                scale_tRS_rCoord = scale_thr_copy_r2s.partition_S(
                    cute.make_identity_tensor(self.epi_tile)
                )
                (
                    scale_acc_epi,
                    scale_final_epi,
                    scale_epi_tile_layout,
                ) = self.make_scale_promotion_plan(
                    acc,
                    final_acc,
                    scale_tiled_copy_r2s,
                    scale_tRS_rD,
                )

                if const_expr(self.pingpong):
                    if warp_group_idx == 0:
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="mma")
                        self.pingpong_barrier_arrive(warp_group_idx=0, stage="epi")

                c_tile_cnt = cute.size(self.epi_tile_shape)
                ab_read_state = make_pipeline_state(pipeline.PipelineUserType.Consumer, self.ab_stage)
                epi_store_pipeline = self.make_epi_store_pipeline()
                epi_read_state = make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.epi_c_stage,
                )
                epi_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.epi_c_stage,
                )
                tile_scheduler = TileSchedulerFactory()
                work_tile = tile_scheduler.initial_work_tile_info()
                if const_expr(self.pingpong):
                    if warp_idx >= 4:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                        k_tile_cnt = (
                            self.adaptive_physical_stage_count(len_k)
                            if const_expr(self.use_adaptive64_tma)
                            else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                        )
                        ab_read_state = self.advance_pipeline_state(ab_read_state, k_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()

                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    len_k = varlen_manager.len_k(batch_idx)
                    k_tile_cnt = (
                        self.adaptive_physical_stage_count(len_k)
                        if const_expr(self.use_adaptive64_tma)
                        else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                    )
                    if const_expr(
                        self.pingpong
                        and self.debug_mode != "epilogue_only"
                    ):
                        self.pingpong_barrier_sync(warp_group_idx, stage="mma")
                    if const_expr(self.debug_mode == "epilogue_only"):
                        final_acc.fill(0.0)
                    elif const_expr(self.debug_mode == "tma_only"):
                        tctx.b("tma_drain")
                        ab_read_state = self.drain_ab_pipeline(
                            ab_pipeline,
                            ab_read_state,
                            k_tile_cnt,
                            warp_group_idx,
                        )
                        tctx.e("tma_drain")
                    elif const_expr(self.debug_mode == "unscaled"):
                        tctx.b("mma_unscaled")
                        if const_expr(self.packed_mode):
                            if const_expr(self.use_adaptive64_tma):
                                if const_expr(
                                    self.adaptive64_unscaled_consumer
                                    == ADAPTIVE64_UNSCALED_CONSUMER_PHYSICAL
                                ):
                                    ab_read_state = self.mma_unscaled_adaptive64_physical_tail_128k(
                                        ab_pipeline,
                                        ab_read_state,
                                        tiled_mma,
                                        mma_fn,
                                        acc,
                                        tCrA,
                                        tCrB,
                                        tCrA_k64,
                                        tCrB_k64,
                                        tCrA_k32,
                                        tCrB_k32,
                                        final_acc,
                                        k_tile_cnt,
                                        len_k,
                                        warp_group_idx,
                                    )
                                else:
                                    ab_read_state = self.mma_unscaled_adaptive64_128k(
                                        ab_pipeline,
                                        ab_read_state,
                                        tiled_mma,
                                        mma_fn,
                                        acc,
                                        tCrA,
                                        tCrB,
                                        tCrA_k64,
                                        tCrB_k64,
                                        tCrA_k32,
                                        tCrB_k32,
                                        final_acc,
                                        k_tile_cnt,
                                        len_k,
                                        warp_group_idx,
                                        tile_coord_mnkl,
                                        mBlockOffsets_l,
                                        mBlockRowCount_b,
                                    )
                            else:
                                ab_read_state = self.mma_unscaled_masked_tail_128k(
                                    ab_pipeline,
                                    ab_read_state,
                                    tiled_mma,
                                    mma_fn,
                                    acc,
                                    tCrA,
                                    tCrB,
                                    final_acc,
                                    k_tile_cnt,
                                    len_k,
                                    warp_group_idx,
                                )
                        else:
                            ab_read_state = self.mma_unscaled_128k(
                                ab_pipeline,
                                ab_read_state,
                                mma_fn,
                                acc,
                                final_acc,
                                k_tile_cnt,
                                warp_group_idx,
                            )
                        tctx.e("mma_unscaled")
                    elif const_expr(self.debug_mode == "scale_only"):
                        tctx.b("scale_only")
                        ab_read_state = self.scale_product_128k(
                            ab_pipeline,
                            ab_read_state,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            thr_mma,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                        )
                        tctx.e("scale_only")
                    elif const_expr(self.debug_mode == "coord_m"):
                        tctx.b("coord_m")
                        ab_read_state = self.coordinate_128k(
                            ab_pipeline,
                            ab_read_state,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            thr_mma,
                            False,
                        )
                        tctx.e("coord_m")
                    elif const_expr(self.debug_mode == "coord_n"):
                        tctx.b("coord_n")
                        ab_read_state = self.coordinate_128k(
                            ab_pipeline,
                            ab_read_state,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            thr_mma,
                            True,
                        )
                        tctx.e("coord_n")
                    elif const_expr(self.debug_mode == "raw_scaled"):
                        tctx.b("mma_raw_scaled")
                        ab_read_state = self.mma_scaled_raw_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            thr_mma,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                        )
                        tctx.e("mma_raw_scaled")
                    elif const_expr(self.debug_mode == "masked_tail"):
                        tctx.b("mma_masked_tail")
                        ab_read_state = self.mma_scaled_masked_tail_128k(
                            ab_pipeline,
                            ab_read_state,
                            tiled_mma,
                            mma_fn,
                            acc,
                            tCrA,
                            tCrB,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            mBlockRowCount_b,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_masked_tail")
                    elif const_expr(self.debug_mode == "masked_tail_last_only"):
                        tctx.b("mma_masked_tail_last_only")
                        ab_read_state = self.mma_scaled_masked_tail_last_only_128k(
                            ab_pipeline,
                            ab_read_state,
                            tiled_mma,
                            mma_fn,
                            acc,
                            tCrA,
                            tCrB,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            mBlockRowCount_b,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_masked_tail_last_only")
                    elif const_expr(self.debug_mode == "masked_tail_early_release"):
                        tctx.b("mma_masked_tail_early_release")
                        ab_read_state = self.mma_masked_tail_early_release_128k(
                            ab_pipeline,
                            ab_read_state,
                            tiled_mma,
                            mma_fn,
                            acc,
                            tCrA,
                            tCrB,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            mBlockRowCount_b,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_masked_tail_early_release")
                    elif const_expr(self.debug_mode == "early_release_scaled"):
                        tctx.b("mma_early_release_scaled")
                        ab_read_state = self.mma_early_release_scaled_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_early_release_scaled")
                    elif const_expr(
                        self.debug_mode == "staged_early_release_scaled"
                        or self.debug_mode == "staged_scale_early_release"
                    ):
                        tctx.b("mma_staged_early_release_scaled")
                        ab_read_state = self.mma_early_release_scaled_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_staged_early_release_scaled")
                    elif const_expr(self.debug_mode == "pipelined_scaled"):
                        tctx.b("mma_pipelined_scaled")
                        pipelined_acc = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
                        pipelined_mma_fn = partial(
                            quack_sm90_utils.gemm_w_idx,
                            tiled_mma,
                            pipelined_acc,
                            tCrA,
                            tCrB,
                        )
                        ab_read_state = self.mma_pipelined_scaled_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            pipelined_mma_fn,
                            acc,
                            pipelined_acc,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_tiled_copy_r2s,
                            scale_thr_copy_r2s,
                            scale_tRS_rD,
                        )
                        tctx.e("mma_pipelined_scaled")
                    elif const_expr(self.debug_mode == "scale_overlap_dbacc_smallm"):
                        tctx.b("mma_scale_overlap_dbacc_smallm")

                        # Option A intentionally pays for the second raw
                        # accumulator only in the 64x128 double-buffer mode.
                        pipe_acc = cute.make_rmem_tensor(acc.shape, self.acc_dtype)
                        pipe_acc_epi = self.epi_retile_acc(
                            pipe_acc,
                            scale_tRS_rD,
                            scale_tiled_copy_r2s,
                        )
                        pipe_mma_fn = partial(
                            quack_sm90_utils.gemm_w_idx,
                            tiled_mma,
                            pipe_acc,
                            tCrA,
                            tCrB,
                        )
                        ab_read_state = self.mma_scale_overlap_dbacc_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            pipe_mma_fn,
                            acc,
                            pipe_acc,
                            final_acc,
                            scale_acc_epi,
                            pipe_acc_epi,
                            scale_final_epi,
                            scale_epi_tile_layout,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_thr_copy_r2s,
                        )
                        tctx.e("mma_scale_overlap_dbacc_smallm")
                    elif const_expr(self.debug_mode == "scale_overlap_coop128_sidecar"):
                        tctx.b("mma_scale_overlap_coop128_sidecar")
                        # Public Option B mode: combine the best current pieces for
                        # SM90 128x128.
                        # - staged2 gives the strongest scale-promotion path
                        # - early release / next-stage poll lowers pipeline slack
                        # - one raw accumulator avoids the reg-pressure cliff from
                        #   the rejected cooperative double-acc sidecar design
                        ab_read_state = self.mma_scale_staged2_release_hybrid_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            final_acc,
                            scale_acc_epi,
                            scale_final_epi,
                            scale_epi_tile_layout,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_thr_copy_r2s,
                        )
                        tctx.e("mma_scale_overlap_coop128_sidecar")
                    elif const_expr(self.debug_mode == "scale_sidecar_no_overlap"):
                        tctx.b("mma_scale_sidecar_no_overlap")
                        ab_read_state = self.mma_scale_sidecar_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            k_tile_cnt,
                            warp_group_idx,
                            scale_acc_epi,
                            scale_final_epi,
                            scale_epi_tile_layout,
                            scale_tRS_rCoord,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            sScaleL,
                            sScaleR,
                        )
                        tctx.e("mma_scale_sidecar_no_overlap")
                    elif const_expr(self.debug_mode == "scale_product_prefetch"):
                        tctx.b("mma_scale_product_prefetch")
                        ab_read_state = self.mma_scale_product_prefetch_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            scale_acc_epi,
                            scale_final_epi,
                            scale_epi_tile_layout,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_thr_copy_r2s,
                        )
                        tctx.e("mma_scale_product_prefetch")
                    elif const_expr(self.debug_mode == "scale_overlap_pingpong_smallacc"):
                        tctx.b("mma_scale_overlap_pingpong_smallacc")
                        ab_read_state = self.mma_scale_overlap_pingpong_smallacc_128k(
                            ab_pipeline,
                            ab_read_state,
                            mma_fn,
                            acc,
                            final_acc,
                            scale_acc_epi,
                            scale_final_epi,
                            scale_epi_tile_layout,
                            k_tile_cnt,
                            warp_group_idx,
                            tile_coord_mnkl,
                            mLhsScales_bm,
                            mRhsScales_bn,
                            mBlockOffsets_l,
                            scale_thr_copy_r2s,
                        )
                        tctx.e("mma_scale_overlap_pingpong_smallacc")
                    else:
                        if const_expr(self.debug_mode == "scalar_scaled"):
                            tctx.b("mma_scalar_scaled")
                        elif const_expr(self.debug_mode == "vector_scaled"):
                            tctx.b("mma_vector_scaled")
                        elif const_expr(self.debug_mode == "staged_scale_cached"):
                            tctx.b("mma_staged_scale_cached")
                        elif const_expr(self.debug_mode == "staged_scale_reg"):
                            tctx.b("mma_staged_scale_reg")
                        elif const_expr(self.debug_mode == "staged_scale_smem"):
                            tctx.b("mma_staged_scale_smem")
                        elif const_expr(self.debug_mode == "rhs_scale_reuse"):
                            tctx.b("mma_rhs_scale_reuse")
                        elif const_expr(self.debug_mode == "scale_staged2"):
                            tctx.b("mma_scale_staged2")
                        else:
                            tctx.b("mma_staged_scale")
                        if const_expr(self.packed_mode):
                            ab_read_state = self.mma_scaled_masked_tail_staged2_128k(
                                ab_pipeline,
                                ab_read_state,
                                tiled_mma,
                                mma_fn,
                                acc,
                                tCrA,
                                tCrB,
                                final_acc,
                                scale_acc_epi,
                                scale_final_epi,
                                scale_epi_tile_layout,
                                k_tile_cnt,
                                len_k,
                                warp_group_idx,
                                tile_coord_mnkl,
                                mLhsScales_bm,
                                mRhsScales_bn,
                                mBlockOffsets_l,
                                scale_tiled_copy_r2s,
                                scale_thr_copy_r2s,
                                scale_tRS_rD,
                            )
                        else:
                            ab_read_state = self.mma_scaled_128k(
                                ab_pipeline,
                                ab_read_state,
                                mma_fn,
                                acc,
                                final_acc,
                                k_tile_cnt,
                                warp_group_idx,
                                tile_coord_mnkl,
                                thr_mma,
                                mLhsScales_bm,
                                mRhsScales_bn,
                                mBlockOffsets_l,
                                scale_tiled_copy_r2s,
                                scale_thr_copy_r2s,
                                scale_tRS_rD,
                                scale_tRS_rCoord,
                                scale_acc_epi,
                                scale_final_epi,
                                scale_epi_tile_layout,
                            )
                        if const_expr(self.debug_mode == "scalar_scaled"):
                            tctx.e("mma_scalar_scaled")
                        elif const_expr(self.debug_mode == "vector_scaled"):
                            tctx.e("mma_vector_scaled")
                        elif const_expr(self.debug_mode == "staged_scale_cached"):
                            tctx.e("mma_staged_scale_cached")
                        elif const_expr(self.debug_mode == "staged_scale_reg"):
                            tctx.e("mma_staged_scale_reg")
                        elif const_expr(self.debug_mode == "staged_scale_smem"):
                            tctx.e("mma_staged_scale_smem")
                        elif const_expr(self.debug_mode == "rhs_scale_reuse"):
                            tctx.e("mma_rhs_scale_reuse")
                        elif const_expr(self.debug_mode == "scale_staged2"):
                            tctx.e("mma_scale_staged2")
                        else:
                            tctx.e("mma_staged_scale")

                    if const_expr(
                        not _is_tma_drain_mode(self.debug_mode)
                        and self.debug_mode != "no_epilogue"
                    ):
                        if const_expr(self.pingpong):
                            self.pingpong_barrier_sync(warp_group_idx, "epi")
                        tctx.b("epilogue")

                        copy_D, _, _ = self.epilog_gmem_copy_and_partition(
                            tma_atom_d,
                            varlen_manager.offset_batch_epi(mD_mnl, batch_idx),
                            self.cta_tile_shape_mnk[:2],
                            self.epi_tile,
                            sD,
                            tile_coord_mnkl,
                        )

                        tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
                            tiled_mma,
                            self.d_layout,
                            self.d_dtype,
                            sD,
                            tidx,
                        )
                        tRS_rAcc = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)
                        if const_expr(
                            self.debug_mode == "coord_m"
                            or self.debug_mode == "coord_n"
                            or self.debug_mode == "scale_only"
                        ):
                            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                            tRS_rCoord = thr_copy_r2s.partition_S(
                                cute.make_identity_tensor(self.epi_tile)
                            )
                            load_acc_subtile = partial(
                                self.epi_load_wgrad_debug_subtile,
                                tRS_rAcc,
                                tRS_rCoord,
                                tile_coord_mnkl,
                                mLhsScales_bm,
                                mRhsScales_bn,
                                mBlockOffsets_l,
                                k_tile_cnt,
                            )
                        else:
                            load_acc_subtile = partial(self.epi_load_acc_subtile, tRS_rAcc)
                        self.epi_visit_acc(epilogue_params, final_acc, tiled_mma, tile_coord_mnkl, tidx)

                        epi_read_state, epi_producer_state = self.epilogue(
                            epilogue_params,
                            epi_smem_tensors,
                            None,
                            epi_store_pipeline,
                            epi_read_state,
                            epi_producer_state,
                            self.epi_tile,
                            load_acc_subtile,
                            tRS_rD,
                            None,
                            None,
                            tiled_copy_r2s,
                            tRS_sD,
                            None,
                            None,
                            None,
                            copy_D,
                            None,
                            tile_coord_mnkl,
                            varlen_manager,
                            self.epilogue_barrier,
                            tile_scheduler,
                            tidx,
                            is_tma_warp,
                        )

                        if const_expr(self.pingpong):
                            if is_tma_warp:
                                epi_store_pipeline.producer_tail()
                            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="epi")
                        tctx.e("epilogue")

                    if const_expr(not self.pingpong):
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                    else:
                        epi_read_state.advance_iters(c_tile_cnt)
                        epi_producer_state.advance_iters(c_tile_cnt)
                        tile_scheduler.advance_to_next_work()
                        work_tile = tile_scheduler.get_current_work()
                        if work_tile.is_valid_tile:
                            len_k = varlen_manager.len_k(batch_idx=work_tile.tile_idx[3])
                            k_tile_cnt = (
                                self.adaptive_physical_stage_count(len_k)
                                if const_expr(self.use_adaptive64_tma)
                                else cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])
                            )
                            ab_read_state = self.advance_pipeline_state(ab_read_state, k_tile_cnt)
                            tile_scheduler.advance_to_next_work()
                            work_tile = tile_scheduler.get_current_work()

                if const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                if const_expr(
                    not self.pingpong
                    and not _is_tma_drain_mode(self.debug_mode)
                    and self.debug_mode != "no_epilogue"
                ):
                    if is_tma_warp:
                        epi_store_pipeline.producer_tail()

        tctx.flush()

    @cute.jit
    def advance_pipeline_state(
        self,
        state: cutlass.pipeline.PipelineState,
        num_iterations: Int32,
    ) -> cutlass.pipeline.PipelineState:
        """Advance a pipeline state without depending on PipelineStateWAdvance.

        `clone()` returns the base CUTLASS PipelineState in this stack, so
        Option B cannot call `advance_iters()` on cloned states.
        """
        for _ in cutlass.range(num_iterations, unroll=1):
            state.advance()
        return state

    @cute.jit
    def adaptive_physical_stage_count(self, len_k: Int32) -> Int32:
        """Return AB pipeline stages for compact packed physical K movement.

        Normal/adaptive64 paths have one pipeline stage per 128-row logical tile.
        A8 twophase splits a 96-row physical tail into K64 + K32 stages so the
        producer can use compact descriptors without overlapping shared memory.
        """

        full_128_tiles = len_k // Int32(128)
        tail_rows = len_k - full_128_tiles * Int32(128)
        stage_cnt = full_128_tiles
        if tail_rows != 0:
            stage_cnt += Int32(1)
            if const_expr(self.use_adaptive_twophase_tma):
                if tail_rows > Int32(64) and tail_rows <= Int32(96):
                    stage_cnt += Int32(1)
        return stage_cnt

    @cute.jit
    def load_tma_kmajor_adaptive64_tail64_only(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy128_fns: Sequence[Optional[Callable]],
        copy64_fns: Sequence[Optional[Callable]],
        full_128_tiles: Int32,
        tail_tma_extra_tx_count: cutlass.Constexpr[int],
    ) -> cutlass.pipeline.PipelineState:
        # E3.1 tail-only helper: callers enter only for physical K64 tails.
        # Keep all full K128 stages on the donor QuACK producer loop.
        ab_producer_state = self.load_tma(
            ab_pipeline,
            ab_producer_state,
            copy128_fns,
            full_128_tiles,
        )

        peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        ab_pipeline.producer_acquire(
            ab_producer_state,
            peek_empty_status,
            extra_tx_count=tail_tma_extra_tx_count,
        )

        tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
        smem_idx = ab_producer_state.index
        tail_src_idx_k64 = full_128_tiles * Int32(2)
        for copy_fn in copy64_fns:
            if const_expr(copy_fn is not None):
                copy_fn(tail_src_idx_k64, smem_idx, tma_bar_ptr=tma_bar_ptr)
        ab_pipeline.producer_commit(ab_producer_state)
        ab_producer_state.advance()

        return ab_producer_state

    @cute.jit
    def load_tma_kmajor_adaptive32_tail_twophase(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy128_fns: Sequence[Optional[Callable]],
        copy64_fns: Sequence[Optional[Callable]],
        copy32_fns: Sequence[Optional[Callable]],
        full_128_tiles: Int32,
        tail_rows: Int32,
        tail_tma_extra_tx_count_k64: cutlass.Constexpr[int],
        tail_tma_extra_tx_count_k32: cutlass.Constexpr[int],
    ) -> cutlass.pipeline.PipelineState:
        """A8 producer for exact32 physical tails.

        1..32 rows use one K32 TMA stage, 33..64 rows use one K64 stage, and
        65..96 rows use two stages: K64 followed by K32. The second stage avoids
        overlapping the K64 shared-memory prefix while preserving exact physical
        movement.
        """

        ab_producer_state = self.load_tma(
            ab_pipeline,
            ab_producer_state,
            copy128_fns,
            full_128_tiles,
        )

        if tail_rows <= Int32(32):
            peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_empty_status,
                extra_tx_count=tail_tma_extra_tx_count_k32,
            )
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            tail_src_idx_k32 = full_128_tiles * Int32(4)
            for copy_fn in copy32_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(tail_src_idx_k32, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
        elif tail_rows <= Int32(64):
            peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_empty_status,
                extra_tx_count=tail_tma_extra_tx_count_k64,
            )
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            tail_src_idx_k64 = full_128_tiles * Int32(2)
            for copy_fn in copy64_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(tail_src_idx_k64, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
        elif tail_rows <= Int32(96):
            peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_empty_status,
                extra_tx_count=tail_tma_extra_tx_count_k64,
            )
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            tail_src_idx_k64 = full_128_tiles * Int32(2)
            for copy_fn in copy64_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(tail_src_idx_k64, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()

            peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            ab_pipeline.producer_acquire(
                ab_producer_state,
                peek_empty_status,
                extra_tx_count=tail_tma_extra_tx_count_k32,
            )
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            tail_src_idx_k32 = full_128_tiles * Int32(4) + Int32(2)
            for copy_fn in copy32_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(tail_src_idx_k32, smem_idx, tma_bar_ptr=tma_bar_ptr)
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
        return ab_producer_state

    @cute.jit
    def load_scale_sidecar_one_k(
        self,
        ab_producer_state: cutlass.pipeline.PipelineState,
        tile_coord_mnkl: cute.Coord,
        k_tile: Int32,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
    ) -> None:
        """Load one 128-row scale block into the sidecar smem for the same AB stage."""
        tidx, _, _ = cute.arch.thread_idx()
        producer_lane = tidx - self.ab_load_warp_id * cute.arch.WARP_SIZE
        producer_threads = self.num_ab_load_warps * cute.arch.WARP_SIZE

        stage_idx = ab_producer_state.index
        scale_block_idx = mBlockOffsets_l[tile_coord_mnkl[3]] + k_tile

        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])

        for vec_i in cutlass.range(
            cute.ceil_div(self.cta_tile_shape_mnk[0] + SCALE_SMEM_PAD, producer_threads),
            unroll=1,
        ):
            local_m = producer_lane + vec_i * producer_threads
            if local_m < self.cta_tile_shape_mnk[0]:
                g_m = tile_m0 + local_m
                if g_m < m_extent:
                    sScaleL[stage_idx, local_m] = mLhsScales_bm[scale_block_idx, g_m]
                else:
                    sScaleL[stage_idx, local_m] = Float32(1.0)
            elif local_m < self.cta_tile_shape_mnk[0] + SCALE_SMEM_PAD:
                sScaleL[stage_idx, local_m] = Float32(1.0)

        for vec_i in cutlass.range(
            cute.ceil_div(self.cta_tile_shape_mnk[1] + SCALE_SMEM_PAD, producer_threads),
            unroll=1,
        ):
            local_n = producer_lane + vec_i * producer_threads
            if local_n < self.cta_tile_shape_mnk[1]:
                g_n = tile_n0 + local_n
                if g_n < n_extent:
                    sScaleR[stage_idx, local_n] = mRhsScales_bn[scale_block_idx, g_n]
                else:
                    sScaleR[stage_idx, local_n] = Float32(1.0)
            elif local_n < self.cta_tile_shape_mnk[1] + SCALE_SMEM_PAD:
                sScaleR[stage_idx, local_n] = Float32(1.0)

    @cute.jit
    def load_tma_with_scale_sidecar(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_fns: Sequence[Optional[Callable]],
        k_tile_cnt: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Produce normal AB stages and populate per-stage scale sidecars."""
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            peek_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
            ab_pipeline.producer_acquire(ab_producer_state, peek_empty_status)

            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            for copy_fn in copy_fns:
                if const_expr(copy_fn is not None):
                    copy_fn(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)

            self.load_scale_sidecar_one_k(
                ab_producer_state,
                tile_coord_mnkl,
                k_tile,
                mLhsScales_bm,
                mRhsScales_bn,
                mBlockOffsets_l,
                sScaleL,
                sScaleR,
            )
            self.scale_sidecar_barrier_arrive(smem_idx)

            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()

        return ab_producer_state

    def scale_sidecar_barrier_sync(self, stage_idx: Int32):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr) + stage_idx,
            number_of_threads=(self.num_epi_warps + self.num_ab_load_warps) * cute.arch.WARP_SIZE,
        )

    def scale_sidecar_barrier_arrive(self, stage_idx: Int32):
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierGemm.TmemPtr) + stage_idx,
            number_of_threads=(self.num_epi_warps + self.num_ab_load_warps) * cute.arch.WARP_SIZE,
        )

    def scale_sidecar_consumer_barrier_sync(self, stage_idx: Int32):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierGemm.TmemPtr) + self.ab_stage + stage_idx,
            number_of_threads=self.num_epi_warps * cute.arch.WARP_SIZE,
        )

    @cute.jit
    def load_scale_sidecar_one_k_consumer(
        self,
        stage_idx: Int32,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
    ) -> None:
        """Consumer-filled sidecar for bisecting sidecar mapping vs producer transport."""
        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(self.pingpong):
            tidx = tidx % self.num_threads_per_warp_group
        lane = tidx

        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])

        if lane < self.cta_tile_shape_mnk[0]:
            g_m = tile_m0 + lane
            if g_m < m_extent:
                sScaleL[stage_idx, lane] = mLhsScales_bm[scale_block_idx, g_m]
            else:
                sScaleL[stage_idx, lane] = Float32(1.0)
        elif lane < self.cta_tile_shape_mnk[0] + SCALE_SMEM_PAD:
            sScaleL[stage_idx, lane] = Float32(1.0)

        if lane < self.cta_tile_shape_mnk[1]:
            g_n = tile_n0 + lane
            if g_n < n_extent:
                sScaleR[stage_idx, lane] = mRhsScales_bn[scale_block_idx, g_n]
            else:
                sScaleR[stage_idx, lane] = Float32(1.0)
        elif lane < self.cta_tile_shape_mnk[1] + SCALE_SMEM_PAD:
            sScaleR[stage_idx, lane] = Float32(1.0)

    @cute.jit
    def drain_ab_pipeline(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
    ) -> cutlass.pipeline.PipelineState:
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def gemm_w_idx_prefix_k(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        zero_init: Boolean,
        atom_limit: cutlass.Constexpr[int],
        A_idx: Optional[Int32] = None,
        B_idx: Optional[Int32] = None,
    ) -> None:
        """Issue a prefix of the four FP8 WGMMA K-atoms in one 128-row slot."""
        if const_expr(atom_limit < 1 or atom_limit > cute.size(tCrA.shape[2])):
            raise RuntimeError("atom_limit must be within the available SM90 FP8 WGMMA K-atoms")

        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]

        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k_atom in cutlass.range_constexpr(atom_limit):
            cute.gemm(mma_atom, acc, rA[None, None, k_atom], rB[None, None, k_atom], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()

    @cute.jit
    def issue_masked_tail_mma_128slot(
        self,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        row_count: Int32,
        ab_stage_idx: Int32,
    ) -> None:
        """Skip padded FP8 WGMMA K-atoms for a final partial 128-row scale block."""
        if row_count > Int32(96):
            mma_fn(A_idx=ab_stage_idx, B_idx=ab_stage_idx, zero_init=Boolean(True))
        elif row_count <= Int32(32):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                Boolean(True),
                1,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        elif row_count <= Int32(64):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                Boolean(True),
                2,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        elif row_count <= Int32(96):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                Boolean(True),
                3,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )

    @cute.jit
    def issue_masked_unscaled_mma_128slot(
        self,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        row_count: Int32,
        ab_stage_idx: Int32,
        zero_init: Boolean,
    ) -> None:
        """Issue only the FP8 WGMMA K-atoms covered by a packed 128-row slot."""
        if row_count > Int32(96):
            mma_fn(A_idx=ab_stage_idx, B_idx=ab_stage_idx, zero_init=zero_init)
        elif row_count <= Int32(32):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                zero_init,
                1,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        elif row_count <= Int32(64):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                zero_init,
                2,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        elif row_count <= Int32(96):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                zero_init,
                3,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )

    @cute.jit
    def issue_masked_unscaled_mma_k64slot(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrA_k64: cute.Tensor,
        tCrB_k64: cute.Tensor,
        row_count: Int32,
        ab_stage_idx: Int32,
        zero_init: Boolean,
    ) -> None:
        """Issue the compact K64 tail WGMMA atoms for an adaptive64 physical tail."""
        if row_count <= Int32(32):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA_k64,
                tCrB_k64,
                zero_init,
                1,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        else:
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA_k64,
                tCrB_k64,
                zero_init,
                2,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )

    @cute.jit
    def mma_unscaled_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)
        zero_init = Boolean(True)
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=zero_init)
            zero_init = Boolean(False)
            warpgroup.wait_group(k_pipe_mmas)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if k_tile_cnt != 0:
            final_acc.store(acc.load())
        return ab_read_state

    @cute.jit
    def mma_unscaled_masked_tail_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        len_k: Int32,
        warp_group_idx: Int32,
    ) -> cutlass.pipeline.PipelineState:
        """Unscaled packed debug path with masked partial 128-row K slots."""
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        zero_init = Boolean(True)
        full_k_tile_cnt = k_tile_cnt
        tail_wgmma_rows = Int32(128)
        if 0 < k_tile_cnt:
            tail_wgmma_rows = len_k - (k_tile_cnt - 1) * Int32(self.cta_tile_shape_mnk[2])
            if tail_wgmma_rows < Int32(128):
                full_k_tile_cnt = k_tile_cnt - 1
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            if k_tile < full_k_tile_cnt:
                mma_fn(
                    A_idx=ab_read_state.index,
                    B_idx=ab_read_state.index,
                    zero_init=zero_init,
                )
            else:
                self.issue_masked_unscaled_mma_128slot(
                    tiled_mma,
                    mma_fn,
                    acc,
                    tCrA,
                    tCrB,
                    tail_wgmma_rows,
                    ab_read_state.index,
                    zero_init,
                )
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            if k_tile < full_k_tile_cnt:
                mma_fn(
                    A_idx=ab_read_state.index,
                    B_idx=ab_read_state.index,
                    zero_init=zero_init,
                )
            else:
                self.issue_masked_unscaled_mma_128slot(
                    tiled_mma,
                    mma_fn,
                    acc,
                    tCrA,
                    tCrB,
                    tail_wgmma_rows,
                    ab_read_state.index,
                    zero_init,
                )
            zero_init = Boolean(False)
            warpgroup.wait_group(k_pipe_mmas)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if k_tile_cnt != 0:
            final_acc.store(acc.load())
        return ab_read_state

    @cute.jit
    def issue_unscaled_adaptive_tail_stage(
        self,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCrA_k64: cute.Tensor,
        tCrB_k64: cute.Tensor,
        tCrA_k32: cute.Tensor,
        tCrB_k32: cute.Tensor,
        k_tile: Int32,
        full_128_tiles: Int32,
        tail_rows: Int32,
        last_row_count: Int32,
        ab_stage_idx: Int32,
        zero_init: Boolean,
        exact32_rows: cutlass.Constexpr[bool],
    ) -> None:
        if k_tile < full_128_tiles:
            mma_fn(A_idx=ab_stage_idx, B_idx=ab_stage_idx, zero_init=zero_init)
        elif tail_rows == 0:
            if const_expr(exact32_rows) and last_row_count <= Int32(96):
                self.issue_masked_unscaled_mma_128slot(
                    tiled_mma,
                    mma_fn,
                    acc,
                    tCrA,
                    tCrB,
                    last_row_count,
                    ab_stage_idx,
                    zero_init,
                )
            else:
                mma_fn(A_idx=ab_stage_idx, B_idx=ab_stage_idx, zero_init=zero_init)
        elif const_expr(self.use_adaptive32_tma) and tail_rows <= Int32(32):
            self.gemm_w_idx_prefix_k(
                tiled_mma,
                acc,
                tCrA_k32,
                tCrB_k32,
                zero_init,
                1,
                A_idx=ab_stage_idx,
                B_idx=ab_stage_idx,
            )
        elif tail_rows <= Int32(64):
            if const_expr(exact32_rows):
                self.issue_masked_unscaled_mma_k64slot(
                    tiled_mma,
                    acc,
                    tCrA_k64,
                    tCrB_k64,
                    last_row_count,
                    ab_stage_idx,
                    zero_init,
                )
            else:
                self.gemm_w_idx_prefix_k(
                    tiled_mma,
                    acc,
                    tCrA_k64,
                    tCrB_k64,
                    zero_init,
                    2,
                    A_idx=ab_stage_idx,
                    B_idx=ab_stage_idx,
                )
        elif const_expr(self.use_adaptive_twophase_tma) and tail_rows <= Int32(96):
            if k_tile == full_128_tiles:
                self.gemm_w_idx_prefix_k(
                    tiled_mma,
                    acc,
                    tCrA_k64,
                    tCrB_k64,
                    zero_init,
                    2,
                    A_idx=ab_stage_idx,
                    B_idx=ab_stage_idx,
                )
            else:
                self.gemm_w_idx_prefix_k(
                    tiled_mma,
                    acc,
                    tCrA_k32,
                    tCrB_k32,
                    zero_init,
                    1,
                    A_idx=ab_stage_idx,
                    B_idx=ab_stage_idx,
                )
        elif const_expr(exact32_rows) and last_row_count <= Int32(96):
            self.issue_masked_unscaled_mma_128slot(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                last_row_count,
                ab_stage_idx,
                zero_init,
            )
        else:
            mma_fn(A_idx=ab_stage_idx, B_idx=ab_stage_idx, zero_init=zero_init)

    @cute.jit
    def mma_unscaled_adaptive64_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCrA_k64: cute.Tensor,
        tCrB_k64: cute.Tensor,
        tCrA_k32: cute.Tensor,
        tCrB_k32: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        len_k: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Unscaled adaptive path with exact32-row WGMMA on the final scale block."""
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        zero_init = Boolean(True)
        block_base = cute.arch.make_warp_uniform(mBlockOffsets_l[tile_coord_mnkl[3]])
        full_128_tiles = len_k // Int32(128)
        tail_rows = len_k - full_128_tiles * Int32(128)
        last_scale_tile = Int32(0)
        last_row_count = Int32(128)
        if 0 < k_tile_cnt:
            if tail_rows == 0:
                last_scale_tile = k_tile_cnt - Int32(1)
            else:
                last_scale_tile = full_128_tiles
            last_row_count = cute.arch.make_warp_uniform(mBlockRowCount_b[block_base + last_scale_tile])
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            self.issue_unscaled_adaptive_tail_stage(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                tCrA_k64,
                tCrB_k64,
                tCrA_k32,
                tCrB_k32,
                k_tile,
                full_128_tiles,
                tail_rows,
                last_row_count,
                ab_read_state.index,
                zero_init,
                True,
            )
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            self.issue_unscaled_adaptive_tail_stage(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                tCrA_k64,
                tCrB_k64,
                tCrA_k32,
                tCrB_k32,
                k_tile,
                full_128_tiles,
                tail_rows,
                last_row_count,
                ab_read_state.index,
                zero_init,
                True,
            )
            zero_init = Boolean(False)
            warpgroup.wait_group(k_pipe_mmas)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if k_tile_cnt != 0:
            final_acc.store(acc.load())
        return ab_read_state

    @cute.jit
    def mma_unscaled_adaptive64_physical_tail_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        tCrA_k64: cute.Tensor,
        tCrB_k64: cute.Tensor,
        tCrA_k32: cute.Tensor,
        tCrB_k32: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        len_k: Int32,
        warp_group_idx: Int32,
    ) -> cutlass.pipeline.PipelineState:
        """Unscaled adaptive consumer that trusts zero-filled physical tails.

        Packed adaptive launch storage is zero-filled outside valid rows.  The
        physical consumer avoids per-block row-count metadata in the MMA loop and
        follows the selected TMA policy: adaptive64 uses K64/K128 tails, while
        A7/A8 can use K32 or K64+K32 physical stages.
        """
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        zero_init = Boolean(True)
        full_128_tiles = len_k // Int32(128)
        tail_rows = len_k - full_128_tiles * Int32(128)
        last_row_count = Int32(128)
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            self.issue_unscaled_adaptive_tail_stage(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                tCrA_k64,
                tCrB_k64,
                tCrA_k32,
                tCrB_k32,
                k_tile,
                full_128_tiles,
                tail_rows,
                last_row_count,
                ab_read_state.index,
                zero_init,
                False,
            )
            zero_init = Boolean(False)
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            self.issue_unscaled_adaptive_tail_stage(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                tCrA_k64,
                tCrB_k64,
                tCrA_k32,
                tCrB_k32,
                k_tile,
                full_128_tiles,
                tail_rows,
                last_row_count,
                ab_read_state.index,
                zero_init,
                False,
            )
            zero_init = Boolean(False)
            warpgroup.wait_group(k_pipe_mmas)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        warpgroup.wait_group(0)
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        if k_tile_cnt != 0:
            final_acc.store(acc.load())
        return ab_read_state

    @cute.jit
    def mma_scaled_masked_tail_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = cute.arch.make_warp_uniform(mBlockOffsets_l[tile_coord_mnkl[3]])
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            row_count = cute.arch.make_warp_uniform(mBlockRowCount_b[scale_block_idx])
            self.issue_masked_tail_mma_128slot(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                row_count,
                ab_read_state.index,
            )
            warpgroup.wait_group(0)
            self.promote_scaled_acc_epilogue_staged(
                acc,
                final_acc,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                tiled_copy_r2s,
                thr_copy_r2s,
                tRS_rD,
            )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            row_count = cute.arch.make_warp_uniform(mBlockRowCount_b[scale_block_idx])
            self.issue_masked_tail_mma_128slot(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                row_count,
                ab_read_state.index,
            )
            warpgroup.wait_group(0)
            self.promote_scaled_acc_epilogue_staged(
                acc,
                final_acc,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                tiled_copy_r2s,
                thr_copy_r2s,
                tRS_rD,
            )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state

    @cute.jit
    def mma_scaled_masked_tail_staged2_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        final_acc: cute.Tensor,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        len_k: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Packed SM90 FP8 path: skip padded 32-row WGMMA atoms in partial blocks."""
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = cute.arch.make_warp_uniform(mBlockOffsets_l[tile_coord_mnkl[3]])
        full_k_tile_cnt = k_tile_cnt
        tail_wgmma_rows = Int32(128)
        if 0 < k_tile_cnt:
            tail_wgmma_rows = len_k - (k_tile_cnt - 1) * Int32(self.cta_tile_shape_mnk[2])
            if tail_wgmma_rows < Int32(128):
                full_k_tile_cnt = k_tile_cnt - 1

        if 0 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base
            if 0 < full_k_tile_cnt:
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            else:
                self.issue_masked_tail_mma_128slot(
                    tiled_mma,
                    mma_fn,
                    acc,
                    tCrA,
                    tCrB,
                    tail_wgmma_rows,
                    ab_read_state.index,
                )
            warpgroup.wait_group(0)
            if const_expr(self.debug_mode == "rhs_scale_reuse"):
                self.promote_scaled_acc_epilogue_rhs_scale_reuse(
                    acc_epi,
                    final_epi,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "scale_staged2"):
                self.promote_scaled_acc_epilogue_staged2(
                    acc_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            else:
                self.promote_scaled_acc_epilogue_staged(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, full_k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)
            if const_expr(self.debug_mode == "rhs_scale_reuse"):
                self.promote_scaled_acc_epilogue_rhs_scale_reuse(
                    acc_epi,
                    final_epi,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "scale_staged2"):
                self.promote_scaled_acc_epilogue_staged2(
                    acc_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            else:
                self.promote_scaled_acc_epilogue_staged(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_idx,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if 0 < full_k_tile_cnt:
            if full_k_tile_cnt < k_tile_cnt:
                ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
                scale_block_idx = scale_block_base + full_k_tile_cnt
                self.issue_masked_tail_mma_128slot(
                    tiled_mma,
                    mma_fn,
                    acc,
                    tCrA,
                    tCrB,
                    tail_wgmma_rows,
                    ab_read_state.index,
                )
                warpgroup.wait_group(0)
                if const_expr(self.debug_mode == "rhs_scale_reuse"):
                    self.promote_scaled_acc_epilogue_rhs_scale_reuse(
                        acc_epi,
                        final_epi,
                        mLhsScales_bm,
                        mRhsScales_bn,
                        scale_block_idx,
                        tile_coord_mnkl,
                        thr_copy_r2s,
                    )
                elif const_expr(self.debug_mode == "scale_staged2"):
                    self.promote_scaled_acc_epilogue_staged2(
                        acc_epi,
                        final_epi,
                        epi_tile_layout,
                        mLhsScales_bm,
                        mRhsScales_bn,
                        scale_block_idx,
                        tile_coord_mnkl,
                        thr_copy_r2s,
                    )
                else:
                    self.promote_scaled_acc_epilogue_staged(
                        acc,
                        final_acc,
                        mLhsScales_bm,
                        mRhsScales_bn,
                        scale_block_idx,
                        tile_coord_mnkl,
                        tiled_copy_r2s,
                        thr_copy_r2s,
                        tRS_rD,
                    )
                ab_pipeline.consumer_release(ab_release_state)
                ab_read_state.advance()
                ab_release_state.advance()

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state

    @cute.jit
    def mma_scaled_masked_tail_last_only_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)

        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = cute.arch.make_warp_uniform(mBlockOffsets_l[tile_coord_mnkl[3]])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile

            # build_wgrad_scale_block_meta makes only the final block for an
            # expert partial; earlier K tiles are always full 128-row blocks.
            is_last_k_tile = k_tile + 1 == k_tile_cnt
            if is_last_k_tile:
                row_count = cute.arch.make_warp_uniform(mBlockRowCount_b[scale_block_idx])
                if row_count < Int32(128):
                    self.issue_masked_tail_mma_128slot(
                        tiled_mma,
                        mma_fn,
                        acc,
                        tCrA,
                        tCrB,
                        row_count,
                        ab_read_state.index,
                    )
                else:
                    mma_fn(
                        A_idx=ab_read_state.index,
                        B_idx=ab_read_state.index,
                        zero_init=Boolean(True),
                    )
            else:
                mma_fn(
                    A_idx=ab_read_state.index,
                    B_idx=ab_read_state.index,
                    zero_init=Boolean(True),
                )

            warpgroup.wait_group(0)
            self.promote_scaled_acc_epilogue_staged(
                acc,
                final_acc,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                tiled_copy_r2s,
                thr_copy_r2s,
                tRS_rD,
            )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        return ab_read_state

    @cute.jit
    def mma_masked_tail_early_release_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        tiled_mma: cute.TiledMma,
        mma_fn: Callable,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        mBlockRowCount_b: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = cute.arch.make_warp_uniform(mBlockOffsets_l[tile_coord_mnkl[3]])
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            row_count = cute.arch.make_warp_uniform(mBlockRowCount_b[scale_block_idx])
            self.issue_masked_tail_mma_128slot(
                tiled_mma,
                mma_fn,
                acc,
                tCrA,
                tCrB,
                row_count,
                ab_read_state.index,
            )
            warpgroup.wait_group(0)

            # Scale promotion reads only registers and scale vectors, so the
            # consumed AB stage can be released before the promotion loop.
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()

            self.promote_scaled_acc_epilogue_staged(
                acc,
                final_acc,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                tiled_copy_r2s,
                thr_copy_r2s,
                tRS_rD,
            )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def scale_product_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_mma: cute.ThrMma,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            self.promote_scale_product(
                final_acc,
                thr_mma,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_base + k_tile,
                tile_coord_mnkl,
            )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def coordinate_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_mma: cute.ThrMma,
        write_n: cutlass.Constexpr[bool],
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_read_state = self.drain_ab_pipeline(ab_pipeline, ab_read_state, k_tile_cnt, warp_group_idx)
        if k_tile_cnt != 0:
            self.promote_coordinate_debug(final_acc, thr_mma, tile_coord_mnkl, write_n)
        return ab_read_state

    def make_scale_promotion_plan(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        tRS_rD: cute.Tensor,
    ):
        """Build scale-promotion views once for the CTA tile.

        The K-loop still changes the scale row (`scale_block_idx`), but the
        accumulator retile and epilogue-subtile traversal are invariant for the
        CTA shape. Hoisting them keeps the hot scale path closer to straight
        scale-vector load plus FMA.
        """
        acc_epi = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        final_epi = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)
        epi_tile_layout = cute.make_ordered_layout(
            self.epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        return acc_epi, final_epi, epi_tile_layout

    def partition_scale_vectors_for_epilogue(
        self,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_copy_r2s: cute.core.ThrCopy,
    ):
        """Partition per-M and per-N scale vectors as zero-stride fragments.

        This stages vectors, not an MxN scale-product tile. The product is kept
        at the epilogue-subtile granularity so register pressure stays close to
        the existing staged path.
        """
        lhs_vec = cute.local_tile(
            mLhsScales_bm[scale_block_idx, None],
            (self.cta_tile_shape_mnk[0],),
            (tile_coord_mnkl[0],),
        )
        rhs_vec = cute.local_tile(
            mRhsScales_bn[scale_block_idx, None],
            (self.cta_tile_shape_mnk[1],),
            (tile_coord_mnkl[1],),
        )
        lhs_scale_mn = cute.make_tensor(
            lhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(1, 0)),
        )
        rhs_scale_mn = cute.make_tensor(
            rhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(0, 1)),
        )
        lhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(lhs_scale_mn, self.epi_tile))
        rhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(rhs_scale_mn, self.epi_tile))
        lhs_scale_epi = cute.group_modes(lhs_scale_epi, 3, cute.rank(lhs_scale_epi))
        rhs_scale_epi = cute.group_modes(rhs_scale_epi, 3, cute.rank(rhs_scale_epi))
        return lhs_scale_epi, rhs_scale_epi

    def partition_scale_sidecar_for_epilogue(
        self,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
        stage_idx: Int32,
        thr_copy_r2s: cute.core.ThrCopy,
    ):
        """Partition scale vectors from smem sidecars instead of global memory."""
        lhs_row_2d = cute.local_tile(
            sScaleL,
            (1, self.cta_tile_shape_mnk[0]),
            (stage_idx, 0),
        )
        rhs_row_2d = cute.local_tile(
            sScaleR,
            (1, self.cta_tile_shape_mnk[1]),
            (stage_idx, 0),
        )
        lhs_vec = lhs_row_2d[0, None]
        rhs_vec = rhs_row_2d[0, None]

        lhs_scale_mn = cute.make_tensor(
            lhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(1, 0)),
        )
        rhs_scale_mn = cute.make_tensor(
            rhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(0, 1)),
        )

        lhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(lhs_scale_mn, self.epi_tile))
        rhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(rhs_scale_mn, self.epi_tile))
        lhs_scale_epi = cute.group_modes(lhs_scale_epi, 3, cute.rank(lhs_scale_epi))
        rhs_scale_epi = cute.group_modes(rhs_scale_epi, 3, cute.rank(rhs_scale_epi))
        return lhs_scale_epi, rhs_scale_epi

    @cute.jit
    def precompute_first_scale_product_fragment(
        self,
        final_epi: cute.Tensor,
        lhs_scale_epi: cute.Tensor,
        rhs_scale_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
    ) -> cute.Tensor:
        epi_coord = epi_tile_layout.get_hier_coord(0)
        final_sub = final_epi[None, None, None, epi_coord]
        lhs_sub_g = lhs_scale_epi[None, None, None, epi_coord]
        rhs_sub_g = rhs_scale_epi[None, None, None, epi_coord]
        lhs_sub = cute.make_rmem_tensor(lhs_sub_g.layout, Float32)
        rhs_sub = cute.make_rmem_tensor(rhs_sub_g.layout, Float32)
        scale_prod = cute.make_rmem_tensor(final_sub.layout, Float32)

        cute.autovec_copy(cute.filter_zeros(lhs_sub_g), cute.filter_zeros(lhs_sub))
        cute.autovec_copy(cute.filter_zeros(rhs_sub_g), cute.filter_zeros(rhs_sub))
        for i in cutlass.range(cute.size(scale_prod), unroll_full=True):
            scale_prod[i] = lhs_sub[i] * rhs_sub[i]
        return scale_prod

    @cute.jit
    def mma_scaled_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_mma: cute.ThrMma,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
        tRS_rCoord: cute.Tensor,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)
            if const_expr(self.debug_mode == "rhs_scale_reuse"):
                self.promote_scaled_acc_epilogue_rhs_scale_reuse(
                    acc_epi,
                    final_epi,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "scale_staged2"):
                self.promote_scaled_acc_epilogue_staged2(
                    acc_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "staged_scale_cached"):
                self.promote_scaled_acc_epilogue_cached(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_rCoord,
                )
            elif const_expr(self.debug_mode != "vector_scaled" and self.debug_mode != "scalar_scaled"):
                self.promote_scaled_acc_epilogue_staged(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            elif const_expr(self.debug_mode == "vector_scaled"):
                self.promote_scaled_acc_epilogue_vector(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            else:
                self.promote_scaled_acc_epilogue(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_rCoord,
                )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)
            if const_expr(self.debug_mode == "rhs_scale_reuse"):
                self.promote_scaled_acc_epilogue_rhs_scale_reuse(
                    acc_epi,
                    final_epi,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "scale_staged2"):
                self.promote_scaled_acc_epilogue_staged2(
                    acc_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            elif const_expr(self.debug_mode == "staged_scale_cached"):
                self.promote_scaled_acc_epilogue_cached(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_rCoord,
                )
            elif const_expr(self.debug_mode != "vector_scaled" and self.debug_mode != "scalar_scaled"):
                self.promote_scaled_acc_epilogue_staged(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            elif const_expr(self.debug_mode == "vector_scaled"):
                self.promote_scaled_acc_epilogue_vector(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            else:
                self.promote_scaled_acc_epilogue(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    tRS_rD,
                    tRS_rCoord,
                )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state

    @cute.jit
    def mma_scale_overlap_dbacc_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn0: Callable,
        mma_fn1: Callable,
        acc0: cute.Tensor,
        acc1: cute.Tensor,
        final_acc: cute.Tensor,
        acc0_epi: cute.Tensor,
        acc1_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> cutlass.pipeline.PipelineState:
        """Option A: overlap scale promotion with the next raw WGMMA.

        The branch is intended for the optA 64x128 configs so the extra raw
        accumulator does not duplicate the default 128x128 register footprint.
        """
        final_acc.fill(0.0)

        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]

        if 0 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn0(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(1, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            if k_tile % 2 == 0:
                mma_fn0(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            else:
                mma_fn1(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))

            # Keep the current WGMMA group in flight and promote only the
            # previous raw accumulator once it is safe to read.
            warpgroup.wait_group(1)
            ab_pipeline.consumer_release(ab_release_state)

            prev_k_tile = k_tile - 1
            if prev_k_tile % 2 == 0:
                self.promote_scaled_acc_epilogue_staged2(
                    acc0_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + prev_k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            else:
                self.promote_scaled_acc_epilogue_staged2(
                    acc1_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + prev_k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )

            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if 0 < k_tile_cnt:
            last_k_tile = k_tile_cnt - 1
            warpgroup.wait_group(0)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            if last_k_tile % 2 == 0:
                self.promote_scaled_acc_epilogue_staged2(
                    acc0_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + last_k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )
            else:
                self.promote_scaled_acc_epilogue_staged2(
                    acc1_epi,
                    final_epi,
                    epi_tile_layout,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + last_k_tile,
                    tile_coord_mnkl,
                    thr_copy_r2s,
                )

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_scale_overlap_coop128_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn0: Callable,
        mma_fn1: Callable,
        acc0_epi: cute.Tensor,
        acc1_epi: cute.Tensor,
        final_acc: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        tRS_rCoord: cute.Tensor,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Cooperative 128x128 Option B with smem scale sidecars."""
        if const_expr(self.ab_stage < 2):
            raise RuntimeError("scale_overlap_coop128_sidecar requires ab_stage >= 2")

        final_acc.fill(0.0)

        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if 0 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn0(
                A_idx=ab_read_state.index,
                B_idx=ab_read_state.index,
                zero_init=Boolean(True),
            )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(1, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)

            if k_tile % 2 == 0:
                mma_fn0(
                    A_idx=ab_read_state.index,
                    B_idx=ab_read_state.index,
                    zero_init=Boolean(True),
                )
            else:
                mma_fn1(
                    A_idx=ab_read_state.index,
                    B_idx=ab_read_state.index,
                    zero_init=Boolean(True),
                )

            warpgroup.wait_group(1)

            prev_stage_idx = ab_release_state.index
            self.scale_sidecar_barrier_sync(prev_stage_idx)
            prev_k_tile = k_tile - 1
            if prev_k_tile % 2 == 0:
                self.promote_scaled_acc_epilogue_sidecar(
                    acc0_epi,
                    final_epi,
                    epi_tile_layout,
                    sScaleL,
                    sScaleR,
                    prev_stage_idx,
                    tRS_rCoord,
                )
            else:
                self.promote_scaled_acc_epilogue_sidecar(
                    acc1_epi,
                    final_epi,
                    epi_tile_layout,
                    sScaleL,
                    sScaleR,
                    prev_stage_idx,
                    tRS_rCoord,
                )

            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()

            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if 0 < k_tile_cnt:
            last_stage_idx = ab_release_state.index
            last_k_tile = k_tile_cnt - 1

            warpgroup.wait_group(0)
            self.scale_sidecar_barrier_sync(last_stage_idx)

            if last_k_tile % 2 == 0:
                self.promote_scaled_acc_epilogue_sidecar(
                    acc0_epi,
                    final_epi,
                    epi_tile_layout,
                    sScaleL,
                    sScaleR,
                    last_stage_idx,
                    tRS_rCoord,
                )
            else:
                self.promote_scaled_acc_epilogue_sidecar(
                    acc1_epi,
                    final_epi,
                    epi_tile_layout,
                    sScaleL,
                    sScaleR,
                    last_stage_idx,
                    tRS_rCoord,
                )

            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()

        return ab_read_state

    @cute.jit
    def mma_scale_sidecar_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        tRS_rCoord: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        """Bisect mode: consumer-filled sidecar without overlap scheduling."""
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]

        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            stage_idx = ab_read_state.index
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=stage_idx, B_idx=stage_idx, zero_init=Boolean(True))
            warpgroup.wait_group(0)

            self.load_scale_sidecar_one_k_consumer(
                stage_idx,
                scale_block_base + k_tile,
                tile_coord_mnkl,
                mLhsScales_bm,
                mRhsScales_bn,
                sScaleL,
                sScaleR,
            )
            self.scale_sidecar_consumer_barrier_sync(stage_idx)

            self.promote_scaled_acc_epilogue_sidecar(
                acc_epi,
                final_epi,
                epi_tile_layout,
                sScaleL,
                sScaleR,
                stage_idx,
                tRS_rCoord,
            )

            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_scale_product_prefetch_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))

            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

            # This scale-vector work is independent of the just-issued WGMMA,
            # so do it only after launch/release and let it overlap with the
            # in-flight WGMMA group plus the producer refill of the released
            # AB stage.
            lhs_scale_epi, rhs_scale_epi = self.partition_scale_vectors_for_epilogue(
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                thr_copy_r2s,
            )
            scale_prod_first = self.precompute_first_scale_product_fragment(
                final_epi,
                lhs_scale_epi,
                rhs_scale_epi,
                epi_tile_layout,
            )
            warpgroup.wait_group(0)

            self.promote_scaled_acc_epilogue_prefetched(
                acc_epi,
                final_epi,
                lhs_scale_epi,
                rhs_scale_epi,
                scale_prod_first,
                epi_tile_layout,
            )

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_scale_staged2_release_hybrid_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        final_acc: cute.Tensor,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> cutlass.pipeline.PipelineState:
        """Composite path: staged2 promotion with early release/next-stage poll.

        This keeps the one-accumulator footprint of the current cooperative
        path, reuses the mathematically clean staged2 promotion, and borrows the
        lower-latency pipeline ordering from the single-acc prefetch variant.
        """
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))

            # Release and start polling the next stage before draining the
            # current WGMMA result. This overlaps producer refill with the
            # eventual scale-promotion work while keeping only one raw
            # accumulator live.
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

            warpgroup.wait_group(0)
            self.promote_scaled_acc_epilogue_staged2(
                acc_epi,
                final_epi,
                epi_tile_layout,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                thr_copy_r2s,
            )

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_scale_overlap_pingpong_smallacc_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            scale_block_idx = scale_block_base + k_tile
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

            lhs_scale_epi, rhs_scale_epi = self.partition_scale_vectors_for_epilogue(
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_idx,
                tile_coord_mnkl,
                thr_copy_r2s,
            )
            scale_prod_first = self.precompute_first_scale_product_fragment(
                final_epi,
                lhs_scale_epi,
                rhs_scale_epi,
                epi_tile_layout,
            )
            warpgroup.wait_group(0)

            # In ping-pong configs, hand the MMA token to the peer before the
            # final scale-promotion subtile work. That turns the last per-K
            # promotion chunk into epilogue-like work and avoids carrying a
            # second full raw accumulator for this tile. Non-pingpong configs
            # still use this mode as a low-risk prefetch/early-release ablation.
            if const_expr(self.pingpong):
                if k_tile + 1 == k_tile_cnt:
                    self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")

            self.promote_scaled_acc_epilogue_prefetched(
                acc_epi,
                final_epi,
                lhs_scale_epi,
                rhs_scale_epi,
                scale_prod_first,
                epi_tile_layout,
            )

        if const_expr(self.pingpong):
            if k_tile_cnt == 0:
                self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_early_release_scaled_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)

            # The scale promotion uses only register accumulators and global
            # scale vectors. Release the consumed AB stage before promotion so
            # the producer can refill the stage while the consumer applies
            # scales. This keeps the one-accumulator register footprint of the
            # normal path, unlike pipelined_scaled.
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()

            if const_expr(
                self.debug_mode == "staged_early_release_scaled"
                or self.debug_mode == "staged_scale_early_release"
            ):
                self.promote_scaled_acc_epilogue_staged(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            else:
                self.promote_scaled_acc_epilogue_vector(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        return ab_read_state

    @cute.jit
    def mma_pipelined_scaled_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        pipelined_mma_fn: Callable,
        acc: cute.Tensor,
        pipelined_acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        ab_release_state = ab_read_state.clone()
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        if 0 < k_tile_cnt:
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(1, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            if k_tile % 2 == 0:
                mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            else:
                pipelined_mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))

            # Wait only for the previous WGMMA group, then promote that raw
            # accumulator while the current group can keep executing.
            warpgroup.wait_group(1)
            ab_pipeline.consumer_release(ab_release_state)
            if (k_tile - 1) % 2 == 0:
                self.promote_scaled_acc_epilogue_vector(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile - 1,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            else:
                self.promote_scaled_acc_epilogue_vector(
                    pipelined_acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + k_tile - 1,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            ab_release_state.advance()
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        if 0 < k_tile_cnt:
            last_k_tile = k_tile_cnt - 1
            warpgroup.wait_group(0)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
            if last_k_tile % 2 == 0:
                self.promote_scaled_acc_epilogue_vector(
                    acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + last_k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
            else:
                self.promote_scaled_acc_epilogue_vector(
                    pipelined_acc,
                    final_acc,
                    mLhsScales_bm,
                    mRhsScales_bn,
                    scale_block_base + last_k_tile,
                    tile_coord_mnkl,
                    tiled_copy_r2s,
                    thr_copy_r2s,
                    tRS_rD,
                )
        return ab_read_state

    @cute.jit
    def mma_scaled_raw_128k(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_read_state: cutlass.pipeline.PipelineState,
        mma_fn: Callable,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        k_tile_cnt: Int32,
        warp_group_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_mma: cute.ThrMma,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
    ) -> cutlass.pipeline.PipelineState:
        final_acc.fill(0.0)
        k_pipe_mmas = 1
        ab_release_state = ab_read_state.clone()
        num_prologue_mma = min(k_pipe_mmas, k_tile_cnt)
        peek_ab_full_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for k_tile in cutlass.range(num_prologue_mma):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)
            self.promote_scaled_acc(
                acc,
                final_acc,
                thr_mma,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_base + k_tile,
                tile_coord_mnkl,
            )
            ab_read_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        for k_tile in cutlass.range(num_prologue_mma, k_tile_cnt, unroll=1):
            ab_pipeline.consumer_wait(ab_read_state, peek_ab_full_status)
            mma_fn(A_idx=ab_read_state.index, B_idx=ab_read_state.index, zero_init=Boolean(True))
            warpgroup.wait_group(0)
            self.promote_scaled_acc(
                acc,
                final_acc,
                thr_mma,
                mLhsScales_bm,
                mRhsScales_bn,
                scale_block_base + k_tile,
                tile_coord_mnkl,
            )
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek_ab_full_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state)

        if const_expr(self.pingpong):
            self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma")
        for k_tile in cutlass.range(num_prologue_mma, unroll=1):
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()
        return ab_read_state

    @cute.jit
    def promote_scaled_acc(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        thr_mma: cute.ThrMma,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
    ) -> None:
        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        coord_acc = thr_mma.partition_C(cAcc)
        # Unscaled mode is correct with final_acc.store(acc.load()), so scale
        # each raw accumulator element with coordinates from the matching raw
        # MMA C partition instead of an MN/frgA view.
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])
        for i in cutlass.range(cute.size(acc), unroll_full=True):
            m = tile_m0 + coord_acc[i][0]
            n = tile_n0 + coord_acc[i][1]
            if m < m_extent and n < n_extent:
                scale = mLhsScales_bm[scale_block_idx, m] * mRhsScales_bn[scale_block_idx, n]
                final_acc[i] = final_acc[i] + acc[i] * scale

    @cute.jit
    def promote_scaled_acc_sidecar(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        thr_mma: cute.ThrMma,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
        stage_idx: Int32,
    ) -> None:
        """Promote one raw accumulator using shared-memory sidecar scales.

        This avoids the more fragile epilogue-subtile scale repartitioning and
        is used for correctness bring-up of the cooperative sidecar path.
        """
        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        coord_acc = thr_mma.partition_C(cAcc)
        for i in cutlass.range(cute.size(acc), unroll_full=True):
            local_m = coord_acc[i][0]
            local_n = coord_acc[i][1]
            scale = sScaleL[stage_idx, local_m] * sScaleR[stage_idx, local_n]
            final_acc[i] = final_acc[i] + acc[i] * scale

    @cute.jit
    def promote_scaled_acc_epilogue(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        tiled_copy_r2s: cute.TiledCopy,
        tRS_rD: cute.Tensor,
        tRS_rCoord: cute.Tensor,
    ) -> None:
        acc_epi = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        final_epi = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]),
            self.epi_tile,
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            subtile_m0 = tile_m0 + epi_coord[0] * self.epi_tile[0]
            subtile_n0 = tile_n0 + epi_coord[1] * self.epi_tile[1]
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                m = subtile_m0 + tRS_rCoord[i][0]
                n = subtile_n0 + tRS_rCoord[i][1]
                if m < m_extent and n < n_extent:
                    scale = mLhsScales_bm[scale_block_idx, m] * mRhsScales_bn[scale_block_idx, n]
                    final_sub[i] = final_sub[i] + acc_sub[i] * scale

    @cute.jit
    def promote_scaled_acc_epilogue_vector(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> None:
        acc_epi = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        final_epi = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)

        lhs_vec = cute.local_tile(
            mLhsScales_bm[scale_block_idx, None],
            (self.cta_tile_shape_mnk[0],),
            (tile_coord_mnkl[0],),
        )
        rhs_vec = cute.local_tile(
            mRhsScales_bn[scale_block_idx, None],
            (self.cta_tile_shape_mnk[1],),
            (tile_coord_mnkl[1],),
        )
        lhs_scale_mn = cute.make_tensor(
            lhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(1, 0)),
        )
        rhs_scale_mn = cute.make_tensor(
            rhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(0, 1)),
        )
        lhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(lhs_scale_mn, self.epi_tile))
        rhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(rhs_scale_mn, self.epi_tile))
        lhs_scale_epi = cute.group_modes(lhs_scale_epi, 3, cute.rank(lhs_scale_epi))
        rhs_scale_epi = cute.group_modes(rhs_scale_epi, 3, cute.rank(rhs_scale_epi))

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]),
            self.epi_tile,
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            lhs_sub = lhs_scale_epi[None, None, None, epi_coord]
            rhs_sub = rhs_scale_epi[None, None, None, epi_coord]
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                final_sub[i] = final_sub[i] + acc_sub[i] * lhs_sub[i] * rhs_sub[i]

    @cute.jit
    def promote_scaled_acc_epilogue_staged(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        tiled_copy_r2s: cute.TiledCopy,
        thr_copy_r2s: cute.core.ThrCopy,
        tRS_rD: cute.Tensor,
    ) -> None:
        acc_epi = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        final_epi = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)

        lhs_vec = cute.local_tile(
            mLhsScales_bm[scale_block_idx, None],
            (self.cta_tile_shape_mnk[0],),
            (tile_coord_mnkl[0],),
        )
        rhs_vec = cute.local_tile(
            mRhsScales_bn[scale_block_idx, None],
            (self.cta_tile_shape_mnk[1],),
            (tile_coord_mnkl[1],),
        )
        lhs_scale_mn = cute.make_tensor(
            lhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(1, 0)),
        )
        rhs_scale_mn = cute.make_tensor(
            rhs_vec.iterator,
            cute.make_layout(self.cta_tile_shape_mnk[:2], stride=(0, 1)),
        )
        lhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(lhs_scale_mn, self.epi_tile))
        rhs_scale_epi = thr_copy_r2s.partition_S(cute.flat_divide(rhs_scale_mn, self.epi_tile))
        lhs_scale_epi = cute.group_modes(lhs_scale_epi, 3, cute.rank(lhs_scale_epi))
        rhs_scale_epi = cute.group_modes(rhs_scale_epi, 3, cute.rank(rhs_scale_epi))

        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]),
            self.epi_tile,
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            lhs_sub_g = lhs_scale_epi[None, None, None, epi_coord]
            rhs_sub_g = rhs_scale_epi[None, None, None, epi_coord]
            lhs_sub = cute.make_rmem_tensor(lhs_sub_g.layout, Float32)
            rhs_sub = cute.make_rmem_tensor(rhs_sub_g.layout, Float32)

            # Keep scale data as broadcast register fragments. filter_zeros
            # preserves the zero-stride vector layout, so this stages the M and
            # N scale vectors without materializing a tile_m x tile_n product.
            cute.autovec_copy(cute.filter_zeros(lhs_sub_g), cute.filter_zeros(lhs_sub))
            cute.autovec_copy(cute.filter_zeros(rhs_sub_g), cute.filter_zeros(rhs_sub))
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                final_sub[i] = final_sub[i] + acc_sub[i] * lhs_sub[i] * rhs_sub[i]

    @cute.jit
    def promote_scaled_acc_epilogue_staged2(
        self,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> None:
        lhs_scale_epi, rhs_scale_epi = self.partition_scale_vectors_for_epilogue(
            mLhsScales_bm,
            mRhsScales_bn,
            scale_block_idx,
            tile_coord_mnkl,
            thr_copy_r2s,
        )
        for epi_idx in cutlass.range_constexpr(cute.size(self.epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            lhs_sub_g = lhs_scale_epi[None, None, None, epi_coord]
            rhs_sub_g = rhs_scale_epi[None, None, None, epi_coord]
            lhs_sub = cute.make_rmem_tensor(lhs_sub_g.layout, Float32)
            rhs_sub = cute.make_rmem_tensor(rhs_sub_g.layout, Float32)

            cute.autovec_copy(cute.filter_zeros(lhs_sub_g), cute.filter_zeros(lhs_sub))
            cute.autovec_copy(cute.filter_zeros(rhs_sub_g), cute.filter_zeros(rhs_sub))
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                final_sub[i] = final_sub[i] + acc_sub[i] * lhs_sub[i] * rhs_sub[i]

    @cute.jit
    def promote_scaled_acc_epilogue_rhs_scale_reuse(
        self,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        thr_copy_r2s: cute.core.ThrCopy,
    ) -> None:
        """Promote a raw FP8 accumulator while reusing RHS scale fragments.

        `rhs_scale[n]` is invariant across all M epilogue subtiles for a fixed
        CTA tile and K128 scale block. Stage each N-subtile RHS fragment once,
        then reuse it while walking the M subtiles. LHS scales remain staged per
        M-subtile. This keeps the data shape O(tile_m + tile_n) and deliberately
        avoids a tile_m x tile_n scale-product cache.
        """
        lhs_scale_epi, rhs_scale_epi = self.partition_scale_vectors_for_epilogue(
            mLhsScales_bm,
            mRhsScales_bn,
            scale_block_idx,
            tile_coord_mnkl,
            thr_copy_r2s,
        )
        for epi_n in cutlass.range_constexpr(self.epi_tile_shape[1]):
            rhs_coord = (0, epi_n)
            rhs_sub_g = rhs_scale_epi[None, None, None, rhs_coord]
            rhs_sub = cute.make_rmem_tensor(rhs_sub_g.layout, Float32)

            cute.autovec_copy(cute.filter_zeros(rhs_sub_g), cute.filter_zeros(rhs_sub))
            for epi_m in cutlass.range_constexpr(self.epi_tile_shape[0]):
                epi_coord = (epi_m, epi_n)
                acc_sub = acc_epi[None, None, None, epi_coord]
                final_sub = final_epi[None, None, None, epi_coord]
                lhs_sub_g = lhs_scale_epi[None, None, None, epi_coord]
                lhs_sub = cute.make_rmem_tensor(lhs_sub_g.layout, Float32)

                cute.autovec_copy(cute.filter_zeros(lhs_sub_g), cute.filter_zeros(lhs_sub))
                for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                    final_sub[i] = final_sub[i] + acc_sub[i] * lhs_sub[i] * rhs_sub[i]

    @cute.jit
    def promote_scaled_acc_epilogue_sidecar(
        self,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        epi_tile_layout: cute.Layout,
        sScaleL: cute.Tensor,
        sScaleR: cute.Tensor,
        stage_idx: Int32,
        tRS_rCoord: cute.Tensor,
    ) -> None:
        """Scale-promotion using smem sidecars with coordinate-driven epilogue mapping.

        The vectorized sidecar repartitioning path proved numerically unstable in
        this codebase. Reuse the same per-element epilogue coordinates as the
        mathematically correct global-scale path, but source scales from smem.
        """
        for epi_idx in cutlass.range_constexpr(cute.size(self.epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            subtile_m0 = epi_coord[0] * self.epi_tile[0]
            subtile_n0 = epi_coord[1] * self.epi_tile[1]
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                local_m = subtile_m0 + tRS_rCoord[i][0]
                local_n = subtile_n0 + tRS_rCoord[i][1]
                if local_m < self.cta_tile_shape_mnk[0] and local_n < self.cta_tile_shape_mnk[1]:
                    scale = sScaleL[stage_idx, local_m] * sScaleR[stage_idx, local_n]
                    final_sub[i] = final_sub[i] + acc_sub[i] * scale

    @cute.jit
    def promote_scaled_acc_epilogue_prefetched(
        self,
        acc_epi: cute.Tensor,
        final_epi: cute.Tensor,
        lhs_scale_epi: cute.Tensor,
        rhs_scale_epi: cute.Tensor,
        scale_prod_first: cute.Tensor,
        epi_tile_layout: cute.Layout,
    ) -> None:
        for epi_idx in cutlass.range_constexpr(cute.size(self.epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            if const_expr(epi_idx == 0):
                for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                    final_sub[i] = final_sub[i] + acc_sub[i] * scale_prod_first[i]
            else:
                lhs_sub_g = lhs_scale_epi[None, None, None, epi_coord]
                rhs_sub_g = rhs_scale_epi[None, None, None, epi_coord]
                lhs_sub = cute.make_rmem_tensor(lhs_sub_g.layout, Float32)
                rhs_sub = cute.make_rmem_tensor(rhs_sub_g.layout, Float32)

                cute.autovec_copy(cute.filter_zeros(lhs_sub_g), cute.filter_zeros(lhs_sub))
                cute.autovec_copy(cute.filter_zeros(rhs_sub_g), cute.filter_zeros(rhs_sub))
                for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                    scale = lhs_sub[i] * rhs_sub[i]
                    final_sub[i] = final_sub[i] + acc_sub[i] * scale

    @cute.jit
    def promote_scaled_acc_epilogue_cached(
        self,
        acc: cute.Tensor,
        final_acc: cute.Tensor,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
        tiled_copy_r2s: cute.TiledCopy,
        tRS_rD: cute.Tensor,
        tRS_rCoord: cute.Tensor,
    ) -> None:
        acc_epi = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)
        final_epi = self.epi_retile_acc(final_acc, tRS_rD, tiled_copy_r2s)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]),
            self.epi_tile,
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = acc_epi[None, None, None, epi_coord]
            final_sub = final_epi[None, None, None, epi_coord]
            subtile_m0 = tile_m0 + epi_coord[0] * self.epi_tile[0]
            subtile_n0 = tile_n0 + epi_coord[1] * self.epi_tile[1]
            last_m = Int32(-1)
            last_n = Int32(-1)
            lhs_cached = Float32(0.0)
            rhs_cached = Float32(0.0)
            for i in cutlass.range(cute.size(final_sub), unroll_full=True):
                m = subtile_m0 + tRS_rCoord[i][0]
                n = subtile_n0 + tRS_rCoord[i][1]
                if m < m_extent and n < n_extent:
                    if m != last_m:
                        lhs_cached = mLhsScales_bm[scale_block_idx, m]
                        last_m = m
                    if n != last_n:
                        rhs_cached = mRhsScales_bn[scale_block_idx, n]
                        last_n = n
                    final_sub[i] = final_sub[i] + acc_sub[i] * lhs_cached * rhs_cached

    @cute.jit
    def promote_scale_product(
        self,
        final_acc: cute.Tensor,
        thr_mma: cute.ThrMma,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        scale_block_idx: Int32,
        tile_coord_mnkl: cute.Coord,
    ) -> None:
        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        coord_acc = thr_mma.partition_C(cAcc)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])
        for i in cutlass.range(cute.size(final_acc), unroll_full=True):
            m = tile_m0 + coord_acc[i][0]
            n = tile_n0 + coord_acc[i][1]
            if m < m_extent and n < n_extent:
                scale = mLhsScales_bm[scale_block_idx, m] * mRhsScales_bn[scale_block_idx, n]
                final_acc[i] = final_acc[i] + scale * Float32(1048576.0)

    @cute.jit
    def promote_coordinate_debug(
        self,
        final_acc: cute.Tensor,
        thr_mma: cute.ThrMma,
        tile_coord_mnkl: cute.Coord,
        write_n: cutlass.Constexpr[bool],
    ) -> None:
        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        coord_acc = thr_mma.partition_C(cAcc)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        for i in cutlass.range(cute.size(final_acc), unroll_full=True):
            coord = tile_n0 + coord_acc[i][1] if const_expr(write_n) else tile_m0 + coord_acc[i][0]
            final_acc[i] = Float32(coord)

    @cute.jit
    def epi_load_wgrad_debug_subtile(
        self,
        tRS_rAcc: cute.Tensor,
        tRS_rCoord: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        mLhsScales_bm: cute.Tensor,
        mRhsScales_bn: cute.Tensor,
        mBlockOffsets_l: cute.Tensor,
        k_tile_cnt: Int32,
        tRS_rD: cute.Tensor,
        epi_coord: cute.Coord,
    ) -> None:
        if const_expr(self.debug_mode != "scale_only"):
            cute.autovec_copy(tRS_rAcc[None, None, None, epi_coord], tRS_rD)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        subtile_m0 = tile_m0 + epi_coord[0] * self.epi_tile[0]
        subtile_n0 = tile_n0 + epi_coord[1] * self.epi_tile[1]
        m_extent = cute.size(mLhsScales_bm, mode=[1])
        n_extent = cute.size(mRhsScales_bn, mode=[1])
        scale_block_base = mBlockOffsets_l[tile_coord_mnkl[3]]
        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            m = subtile_m0 + tRS_rCoord[i][0]
            n = subtile_n0 + tRS_rCoord[i][1]
            value = Float32(0.0)
            if k_tile_cnt != 0:
                if const_expr(self.debug_mode == "coord_m"):
                    value = Float32(m)
                elif const_expr(self.debug_mode == "coord_n"):
                    value = Float32(n)
                elif const_expr(self.debug_mode == "scale_only"):
                    if m < m_extent and n < n_extent:
                        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                            scale = (
                                mLhsScales_bm[scale_block_base + k_tile, m]
                                * mRhsScales_bn[scale_block_base + k_tile, n]
                            )
                            value = value + scale * Float32(1048576.0)
            if const_expr(
                self.debug_mode == "coord_m"
                or self.debug_mode == "coord_n"
                or self.debug_mode == "scale_only"
            ):
                tRS_rD[i] = value

    @cute.jit
    def promote_coordinate_epilogue_debug(
        self,
        tRS_rAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        thr_mma: cute.ThrMma,
        tile_coord_mnkl: cute.Coord,
        write_n: cutlass.Constexpr[bool],
    ) -> None:
        cAcc = cute.make_identity_tensor((self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]))
        coord_acc = thr_mma.partition_C(cAcc)
        tRS_rCoord = self.epi_retile_acc(coord_acc, tRS_rD, tiled_copy_r2s)
        tile_m0 = tile_coord_mnkl[0] * self.cta_tile_shape_mnk[0]
        tile_n0 = tile_coord_mnkl[1] * self.cta_tile_shape_mnk[1]
        epi_tile_shape = cute.zipped_divide(
            cute.make_layout(self.cta_tile_shape_mnk[:2]),
            self.epi_tile,
        ).shape[1]
        epi_tile_layout = cute.make_ordered_layout(
            epi_tile_shape,
            order=(0, 1) if const_expr(self.epi_m_major) else (1, 0),
        )
        for epi_idx in cutlass.range_constexpr(cute.size(epi_tile_shape)):
            epi_coord = epi_tile_layout.get_hier_coord(epi_idx)
            acc_sub = tRS_rAcc[None, None, None, epi_coord]
            coord_sub = tRS_rCoord[None, None, None, epi_coord]
            for i in cutlass.range(cute.size(acc_sub), unroll_full=True):
                coord = tile_n0 + coord_sub[i][1] if const_expr(write_n) else tile_m0 + coord_sub[i][0]
                acc_sub[i] = Float32(coord)


@jit_cache
def _compile_fp8_grouped_wgrad(
    a_dtype,
    b_dtype,
    d_dtype,
    tile_shape_mnk: tuple[int, int, int],
    cluster_shape_mnk: tuple[int, int, int],
    pingpong: bool,
    persistent: bool,
    has_trace_ptr: bool,
    debug_mode: str,
    packed_mode: bool,
    packed_tail_tma_mode: str,
    adaptive64_unscaled_consumer: str,
    adaptive64_k64_desc_prefetch: str,
    use_ragged_tma: bool,
    accumulate: bool,
    static_m: int | None,
    static_n: int | None,
    static_l: int | None,
    custom_source_fingerprint: str,
):
    del custom_source_fingerprint
    m = int(static_m) if static_m is not None else cute.sym_int()
    n = int(static_n) if static_n is not None else cute.sym_int()
    k = cute.sym_int()
    l = int(static_l) if static_l is not None else cute.sym_int()
    total_blocks = cute.sym_int()
    div_a = _div_for_dtype(a_dtype)
    div_b = _div_for_dtype(b_dtype)
    div_d = _div_for_dtype(d_dtype)
    ab_leading_dim = 0 if packed_mode and _packed_uses_donor_tma_layout(debug_mode) else 1
    mA = fake_tensor(a_dtype, (m, k), leading_dim=ab_leading_dim, divisibility=div_a)
    mB = fake_tensor(b_dtype, (n, k), leading_dim=ab_leading_dim, divisibility=div_b)
    mD = fake_tensor(d_dtype, (m, n, l), leading_dim=1, divisibility=div_d)
    mLhsScales = fake_tensor(Float32, (total_blocks, m), leading_dim=1, divisibility=4)
    mRhsScales = fake_tensor(Float32, (total_blocks, n), leading_dim=1, divisibility=4)
    mBlockOffsets = fake_tensor(Int32, (cute.sym_int(),), leading_dim=0, divisibility=4)
    mBlockRowCount = fake_tensor(Int32, (total_blocks,), leading_dim=0, divisibility=4)

    epi_args = Fp8ScaledWgradSm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=bool(accumulate),
    )
    scheduler_args = make_fake_scheduler_args(False, False, l)
    varlen_args = make_fake_varlen_args(False, True, False, None)

    gemm_obj = Fp8ScaledWgradSm90(
        Float32,
        a_dtype,
        tile_shape_mnk,
        cluster_shape_mnk,
        pingpong=pingpong,
        is_persistent=persistent,
        fp8_fast_accum=True,
        gather_A=False,
        concat_layout=None,
        use_pdl=False,
        debug_mode=debug_mode,
        packed_mode=packed_mode,
        packed_tail_tma_mode=packed_tail_tma_mode,
        adaptive64_unscaled_consumer=adaptive64_unscaled_consumer,
        adaptive64_k64_desc_prefetch=adaptive64_k64_desc_prefetch,
        use_ragged_tma=use_ragged_tma,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    trace_ptr = cutlass.Int64(0) if has_trace_ptr else None
    return cute.compile(
        gemm_obj,
        mA,
        mB,
        mD,
        None,
        epi_args,
        scheduler_args,
        varlen_args,
        mLhsScales,
        mRhsScales,
        mBlockOffsets,
        mBlockRowCount,
        stream,
        trace_ptr,
        options="--enable-tvm-ffi",
    )


def _require_wgmma_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be logical [K_rows, cols], got {tuple(tensor.shape)}")
    if int(tensor.stride(0)) != 1:
        raise RuntimeError(
            f"{name} must be SM90 FP8 K-major with stride(0) == 1, "
            f"got stride={tuple(tensor.stride())}"
        )
    if int(tensor.stride(1)) % 16 != 0:
        raise RuntimeError(
            f"{name} leading dimension must be divisible by 16 for SM90 FP8 TMA, "
            f"got stride={tuple(tensor.stride())}"
        )


def _require_varlen_k_wgmma_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be logical [cols, K_rows], got {tuple(tensor.shape)}")
    if int(tensor.stride(1)) != 1:
        raise RuntimeError(
            f"{name} must be packed varlen-K [cols,K] with K-contiguous "
            f"stride(1) == 1 for SM90 FP8 WGMMA, got stride={tuple(tensor.stride())}"
        )
    if int(tensor.stride(0)) % 16 != 0:
        raise RuntimeError(
            f"{name} row pitch must be divisible by 16 for SM90 FP8 ragged TMA, "
            f"got stride={tuple(tensor.stride())}"
        )


def _packed_uses_donor_tma_layout(kernel_debug_mode: str) -> bool:
    return str(kernel_debug_mode).strip().lower() == "tma_only"


def _require_varlen_k_tma_layout(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise RuntimeError(f"{name} must be logical [cols, K_rows], got {tuple(tensor.shape)}")
    if int(tensor.stride(0)) != 1:
        raise RuntimeError(
            f"{name} must use donor SM90 varlen-K TMA layout [cols,K] with "
            f"stride(0) == 1, got stride={tuple(tensor.stride())}"
        )
    if int(tensor.stride(1)) % 16 != 0:
        raise RuntimeError(
            f"{name} K pitch must be divisible by 16 for SM90 ragged TMA, "
            f"got stride={tuple(tensor.stride())}"
        )


def fp8_grouped_wgrad_sm90_impl(
    lhs_q: torch.Tensor,
    lhs_scales: torch.Tensor,
    rhs_q: torch.Tensor,
    rhs_scales: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    block_offsets: torch.Tensor,
    block_row_start: torch.Tensor,
    block_row_count: torch.Tensor,
    *,
    tuned_config: str = "default",
    accumulate: bool = False,
) -> None:
    del block_row_start
    capability = get_device_capacity(out.device)
    if int(capability[0]) != 9:
        raise RuntimeError(f"fp8_grouped_wgrad_sm90_impl requires H100/H200 SM90, got {capability}")
    if lhs_q.dtype != torch.float8_e4m3fn or rhs_q.dtype != torch.float8_e4m3fn:
        raise RuntimeError("lhs_q and rhs_q must be torch.float8_e4m3fn")
    if lhs_scales.dtype != torch.float32 or rhs_scales.dtype != torch.float32:
        raise RuntimeError("FP8 Wgrad scales must be torch.float32")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"out must be BF16 or FP32, got {out.dtype}")
    if cu_seqlens_k.dtype != torch.int32 or block_offsets.dtype != torch.int32:
        raise RuntimeError("cu_seqlens_k and block_offsets must be int32")
    if block_row_count.dtype != torch.int32:
        raise RuntimeError("block_row_count must be int32")
    if block_row_count.ndim != 1:
        raise RuntimeError(f"block_row_count must be [total_blocks], got {tuple(block_row_count.shape)}")
    if not block_row_count.is_contiguous():
        raise RuntimeError("block_row_count must be contiguous")
    if not lhs_scales.is_contiguous() or not rhs_scales.is_contiguous():
        raise RuntimeError("scale tensors must be contiguous")
    if not out.is_contiguous():
        raise RuntimeError("out must be contiguous [E,M,N]")
    if block_offsets.dtype != torch.int32:
        raise RuntimeError("block_offsets must be int32")
    if block_offsets.ndim != 1:
        raise RuntimeError(f"block_offsets must be [E+1], got {tuple(block_offsets.shape)}")
    if not block_offsets.is_contiguous():
        raise RuntimeError("block_offsets must be contiguous")
    experts, m, n = map(int, out.shape)
    if int(cu_seqlens_k.shape[0]) != experts + 1:
        raise RuntimeError("cu_seqlens_k shape must be [E+1]")
    if int(block_offsets.shape[0]) != experts + 1:
        raise RuntimeError("block_offsets shape must be [E+1]")
    if str(tuned_config) not in WGRAD_SM90_CONFIGS:
        raise RuntimeError(f"unknown Wgrad config {tuned_config!r}; choices={sorted(WGRAD_SM90_CONFIGS)}")

    cfg = WGRAD_SM90_CONFIGS[str(tuned_config)]
    layout_mode = resolve_wgrad_layout_mode()
    packed_tail_tma_mode = resolve_packed_tail_tma_mode()
    kernel_debug_mode_raw = os.environ.get("KERNEL_LAB_WGRAD_KERNEL_MODE", "normal").strip().lower()
    kernel_debug_mode = (
        "scale_overlap_coop128_sidecar"
        if kernel_debug_mode_raw == "scale_overlap_wg_pingpong"
        else kernel_debug_mode_raw
    )
    use_adaptive64_tma_path = _uses_adaptive64_tma_path(
        kernel_debug_mode,
        packed_mode=layout_mode == LAYOUT_MODE_PACKED,
        packed_tail_tma_mode=packed_tail_tma_mode,
    )
    adaptive64_unscaled_consumer = _resolve_adaptive64_unscaled_consumer()
    adaptive64_k64_desc_prefetch = _resolve_adaptive64_k64_desc_prefetch()
    if layout_mode == LAYOUT_MODE_PACKED:
        if _packed_uses_donor_tma_layout(kernel_debug_mode):
            _require_varlen_k_tma_layout("lhs_q", lhs_q)
            _require_varlen_k_tma_layout("rhs_q", rhs_q)
        else:
            _require_varlen_k_wgmma_layout("lhs_q", lhs_q)
            _require_varlen_k_wgmma_layout("rhs_q", rhs_q)
        if lhs_q.shape[1] != rhs_q.shape[1]:
            raise RuntimeError(
                "K_rows mismatch: "
                f"lhs_q={tuple(lhs_q.shape)} stride={tuple(lhs_q.stride())} "
                f"rhs_q={tuple(rhs_q.shape)} stride={tuple(rhs_q.stride())}"
            )
        if int(lhs_q.shape[0]) != m or int(rhs_q.shape[0]) != n:
            raise RuntimeError(
                f"packed operand/output mismatch: lhs={tuple(lhs_q.shape)} "
                f"rhs={tuple(rhs_q.shape)} out={tuple(out.shape)}"
            )
    else:
        _require_wgmma_layout("lhs_q", lhs_q)
        _require_wgmma_layout("rhs_q", rhs_q)
        if lhs_q.shape[0] != rhs_q.shape[0]:
            raise RuntimeError(
                "K_rows mismatch: "
                f"lhs_q={tuple(lhs_q.shape)} stride={tuple(lhs_q.stride())} "
                f"rhs_q={tuple(rhs_q.shape)} stride={tuple(rhs_q.stride())}"
            )
        if int(lhs_q.shape[1]) != m or int(rhs_q.shape[1]) != n:
            raise RuntimeError(
                f"operand/output mismatch: lhs={tuple(lhs_q.shape)} "
                f"rhs={tuple(rhs_q.shape)} out={tuple(out.shape)}"
            )
    allow_packed_debug_bisect = _parse_bool_env(
        "KERNEL_LAB_WGRAD_ALLOW_PACKED_DEBUG_BISECT_MODES",
        default=False,
    )
    if kernel_debug_mode not in _VALID_KERNEL_DEBUG_MODES:
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_KERNEL_MODE must be one of "
            f"{_VALID_KERNEL_DEBUG_MODES}; got {kernel_debug_mode!r}"
        )
    if layout_mode == LAYOUT_MODE_PACKED and not packed_layout_supports_kernel_mode(
        kernel_debug_mode,
        include_debug_bisect=bool(allow_packed_debug_bisect),
    ):
        allowed_packed_modes = (
            PACKED_DEBUG_BISECT_KERNEL_DEBUG_MODES
            if bool(allow_packed_debug_bisect)
            else PACKED_SUPPORTED_KERNEL_DEBUG_MODES
        )
        raise RuntimeError(
            "packed FP8 Wgrad currently supports only "
            f"{allowed_packed_modes}; got kernel mode {kernel_debug_mode!r}"
        )
    if _is_kmajor_tma_drain_mode(kernel_debug_mode) and layout_mode != LAYOUT_MODE_PACKED:
        raise RuntimeError(f"{kernel_debug_mode} is defined only for packed K-major Wgrad operands")
    if kernel_debug_mode == "tail64" and cfg.tile_k != 64:
        raise RuntimeError("KERNEL_LAB_WGRAD_KERNEL_MODE=tail64 requires a tile_k=64 tuned config")
    if kernel_debug_mode == "tail32" and cfg.tile_k != 32:
        raise RuntimeError("KERNEL_LAB_WGRAD_KERNEL_MODE=tail32 requires a tile_k=32 tuned config")
    if cfg.tile_k != 128 and kernel_debug_mode not in ("tail64", "tail32"):
        raise RuntimeError("custom FP8 Wgrad tile_k != 128 is only valid for tail64/tail32 modes")
    if kernel_debug_mode == "scale_overlap_dbacc_smallm":
        if cfg.tile_m != 64 or cfg.tile_n != 128 or cfg.tile_k != 128 or cfg.pingpong:
            raise RuntimeError(
                "KERNEL_LAB_WGRAD_KERNEL_MODE=scale_overlap_dbacc_smallm requires "
                "a non-pingpong 64x128x128 config; use optA_down_64x128_dbacc "
                "or optA_gate_64x128_dbacc"
            )
    if kernel_debug_mode in ("scale_overlap_coop128_sidecar", "scale_sidecar_no_overlap"):
        if (
            cfg.tile_m != 128
            or cfg.tile_n != 128
            or cfg.tile_k != 128
            or cfg.cluster_m != 1
            or cfg.cluster_n != 1
            or cfg.cluster_k != 1
            or cfg.pingpong
        ):
            raise RuntimeError(
                "KERNEL_LAB_WGRAD_KERNEL_MODE=scale_overlap_coop128_sidecar/scale_sidecar_no_overlap requires "
                "a non-pingpong 128x128x128 config with cluster=(1,1,1); use "
                "'default' or 'optB_coop128_default'"
            )
    expected_scale_blocks = int(block_offsets[-1].item())
    if int(lhs_scales.shape[0]) != expected_scale_blocks or int(rhs_scales.shape[0]) != expected_scale_blocks:
        raise RuntimeError(
            "scale metadata does not match block_offsets: "
            f"expected {expected_scale_blocks} scale blocks, got "
            f"lhs={int(lhs_scales.shape[0])} rhs={int(rhs_scales.shape[0])}"
        )
    if int(block_row_count.shape[0]) != expected_scale_blocks:
        raise RuntimeError(
            "block_row_count length must match scale block count: "
            f"got {int(block_row_count.shape[0])}, expected {expected_scale_blocks}"
        )
    lhs_k_rows = int(lhs_q.shape[1]) if layout_mode == LAYOUT_MODE_PACKED else int(lhs_q.shape[0])
    if layout_mode == LAYOUT_MODE_PADDED:
        if lhs_k_rows % int(cfg.tile_k) != 0:
            raise RuntimeError(f"padded_TK must be divisible by tile_k={cfg.tile_k}")
        if lhs_k_rows != expected_scale_blocks * int(cfg.tile_k):
            raise RuntimeError(
                "padded operand K_rows must equal total scale blocks * tile_k: "
                f"got K_rows={lhs_k_rows}, blocks={expected_scale_blocks}, tile_k={int(cfg.tile_k)}"
            )
    cu_diffs_host = None
    if _host_validation_enabled():
        cu_diffs_host = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).detach().cpu()
        if layout_mode == LAYOUT_MODE_PADDED and bool((cu_diffs_host % int(cfg.tile_k) != 0).any()):
            raise RuntimeError(
                "SM90 FP8 Wgrad requires padded per-expert K to be divisible by "
                f"tile_k={cfg.tile_k}"
            )
        cu_last = int(cu_seqlens_k[-1].item())
        if layout_mode == LAYOUT_MODE_PACKED:
            if cu_last > lhs_k_rows:
                raise RuntimeError(
                    "packed cu_seqlens_k[-1] must fit inside operand storage K_rows: "
                    f"cu_last={cu_last}, storage_K={lhs_k_rows}"
                )
            if use_adaptive64_tma_path:
                guard_rows = 0
            elif kernel_debug_mode == KERNEL_DEBUG_TMA_KMAJOR_128_DRAIN:
                guard_rows = (int(cfg.tile_k) - (cu_last % int(cfg.tile_k))) % int(cfg.tile_k)
                guard_rows = max(0, guard_rows - 1) if guard_rows > 0 else 0
            else:
                guard_rows = int(cfg.tile_k) - 1
            min_guarded_rows = cu_last + guard_rows if cu_last > 0 else 0
            if lhs_k_rows < min_guarded_rows:
                raise RuntimeError(
                    "packed operand storage must include a tile guard after logical K: "
                    f"storage_K={lhs_k_rows}, required_at_least={min_guarded_rows}"
                )
        elif cu_last != lhs_k_rows:
            raise RuntimeError("cu_seqlens_k[-1] must match operand K_rows")
    tma_mode = os.environ.get("KERNEL_LAB_WGRAD_TMA_MODE", "padded").strip().lower()
    if tma_mode not in ("ragged", "padded"):
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_TMA_MODE must be 'ragged' or 'padded', "
            f"got {tma_mode!r}"
        )
    packed_mode = layout_mode == LAYOUT_MODE_PACKED
    use_ragged_tma = tma_mode == "ragged"
    scheduler_mode = os.environ.get("KERNEL_LAB_WGRAD_SCHEDULER_MODE", "").strip().lower()
    if scheduler_mode and scheduler_mode not in ("persistent", "nonpersistent"):
        raise RuntimeError(
            "KERNEL_LAB_WGRAD_SCHEDULER_MODE must be 'persistent' or 'nonpersistent', "
            f"got {scheduler_mode!r}"
        )
    persistent_override = _parse_bool_env("KERNEL_LAB_WGRAD_PERSISTENT", default=None)
    if persistent_override is not None:
        use_persistent_scheduler = bool(persistent_override)
        scheduler_mode_label = "persistent" if use_persistent_scheduler else "nonpersistent"
    elif scheduler_mode:
        use_persistent_scheduler = scheduler_mode == "persistent"
        scheduler_mode_label = scheduler_mode
    else:
        use_persistent_scheduler = True
        scheduler_mode_label = "persistent_default"
    if layout_mode == LAYOUT_MODE_PACKED and not use_persistent_scheduler:
        raise RuntimeError("packed FP8 Wgrad currently requires persistent scheduler")
    if packed_mode and not use_ragged_tma:
        raise RuntimeError(
            "packed FP8 Wgrad currently requires KERNEL_LAB_WGRAD_TMA_MODE=ragged; "
            "the packed-specific non-ragged producer path was removed because it was incorrect"
        )
    if packed_mode and (
        cfg.cluster_m != 1 or cfg.cluster_n != 1 or cfg.cluster_k != 1 or cfg.pingpong
    ):
        raise RuntimeError(
            "packed FP8 Wgrad currently supports only non-pingpong cluster=(1,1,1) configs; "
            f"got config {tuned_config!r} with cluster="
            f"({cfg.cluster_m},{cfg.cluster_n},{cfg.cluster_k}) pingpong={cfg.pingpong}"
        )
    if cfg.pingpong and not use_persistent_scheduler:
        raise RuntimeError(
            f"Wgrad config {tuned_config!r} uses pingpong and requires persistent scheduler; "
            "choose KERNEL_LAB_WGRAD_PERSISTENT=1, "
            "KERNEL_LAB_WGRAD_SCHEDULER_MODE=persistent, or a non-pingpong config"
        )
    static_shapes = bool(_parse_bool_env("KERNEL_LAB_WGRAD_STATIC_SHAPES", default=False))
    if static_shapes:
        static_m = m
        static_n = n
        static_l = experts
    else:
        static_m = None
        static_n = None
        static_l = None

    if layout_mode == LAYOUT_MODE_PACKED:
        A_mk = lhs_q
        B_nk = rhs_q
    else:
        A_mk = lhs_q.mT
        B_nk = rhs_q.mT
    D_mnl = out.permute(1, 2, 0)
    a_dtype = _torch_to_cute_dtype(A_mk.dtype)
    b_dtype = _torch_to_cute_dtype(B_nk.dtype)
    d_dtype = _torch_to_cute_dtype(D_mnl.dtype)

    compiled_fn = _compile_fp8_grouped_wgrad(
        a_dtype,
        b_dtype,
        d_dtype,
        (cfg.tile_m, cfg.tile_n, cfg.tile_k),
        (cfg.cluster_m, cfg.cluster_n, cfg.cluster_k),
        cfg.pingpong,
        use_persistent_scheduler,
        False,
        kernel_debug_mode,
        packed_mode,
        packed_tail_tma_mode,
        adaptive64_unscaled_consumer,
        adaptive64_k64_desc_prefetch,
        use_ragged_tma,
        bool(accumulate),
        static_m,
        static_n,
        static_l,
        _CUSTOM_SOURCE_FINGERPRINT,
    )

    if quack_cache_utils.COMPILE_ONLY:
        return

    max_active_clusters = get_max_active_clusters(cfg.cluster_m * cfg.cluster_n * cfg.cluster_k)
    epi_args = GemmDefaultEpiMixin.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=None,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(
        max_active_clusters,
        cfg.max_swizzle_size,
        None,
        None,
    )
    varlen_args = make_varlen_args(None, cu_seqlens_k, None)
    if _debug_enabled():
        _debug_log(
            "launch prepared "
            f"config={tuned_config!r} tile=({cfg.tile_m},{cfg.tile_n},{cfg.tile_k}) "
            f"cluster=({cfg.cluster_m},{cfg.cluster_n},{cfg.cluster_k}) "
            f"kernel_mode={kernel_debug_mode!r} "
            f"layout_mode={layout_mode!r} "
            f"packed_tail_tma={packed_tail_tma_mode!r} "
            f"adaptive64_tma_path={use_adaptive64_tma_path} "
            f"adaptive64_unscaled_consumer={adaptive64_unscaled_consumer!r} "
            f"adaptive64_k64_desc_prefetch={adaptive64_k64_desc_prefetch!r} "
            f"tma_mode={tma_mode!r} "
            f"use_ragged_tma={use_ragged_tma} "
            f"scheduler_mode={scheduler_mode_label!r} "
            f"persistent={use_persistent_scheduler} "
            f"accumulate={bool(accumulate)} "
            f"static_shapes={static_shapes} static=({static_m},{static_n},{static_l}) "
            f"cache_enabled={quack_cache_utils.CACHE_ENABLED} "
            f"compile_cache_info={_compile_fp8_grouped_wgrad.cache_info()} "
            f"source_hash={_CUSTOM_SOURCE_FINGERPRINT[:16]}"
        )
        _debug_log(
            f"A_mk shape={tuple(A_mk.shape)} stride={tuple(A_mk.stride())} "
            f"ptr=0x{A_mk.data_ptr():x} dtype={A_mk.dtype}"
        )
        _debug_log(
            f"B_nk shape={tuple(B_nk.shape)} stride={tuple(B_nk.stride())} "
            f"ptr=0x{B_nk.data_ptr():x} dtype={B_nk.dtype}"
        )
        _debug_log(
            f"D_mnl shape={tuple(D_mnl.shape)} stride={tuple(D_mnl.stride())} "
            f"ptr=0x{D_mnl.data_ptr():x} dtype={D_mnl.dtype}"
        )
        _debug_log(
            f"lhs_scales shape={tuple(lhs_scales.shape)} stride={tuple(lhs_scales.stride())} "
            f"ptr=0x{lhs_scales.data_ptr():x}"
        )
        _debug_log(
            f"rhs_scales shape={tuple(rhs_scales.shape)} stride={tuple(rhs_scales.stride())} "
            f"ptr=0x{rhs_scales.data_ptr():x}"
        )
        _debug_log(
            f"block_row_count shape={tuple(block_row_count.shape)} "
            f"ptr=0x{block_row_count.data_ptr():x}"
        )
        _debug_log(
            f"cu_seqlens_k={cu_seqlens_k.detach().cpu().tolist()} "
            f"cu_diffs={[] if cu_diffs_host is None else cu_diffs_host.tolist()} "
            f"block_offsets={block_offsets.detach().cpu().tolist()} "
            f"max_active_clusters={max_active_clusters}"
        )
    compiled_fn(
        A_mk,
        B_nk,
        D_mnl,
        None,
        epi_args,
        scheduler_args,
        varlen_args,
        lhs_scales,
        rhs_scales,
        block_offsets,
        block_row_count,
        None,
    )
