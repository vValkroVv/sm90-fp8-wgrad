#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import sys
import traceback
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from sm90_fp8_wgrad.checks import (
    deepseek_calc_diff_stream,
    iter_ref_scaled_wgrad_by_expert,
    ref_scaled_wgrad,
    scaled_wgrad_stats,
)
from sm90_fp8_wgrad.deepgemm_baseline import (
    load_deepgemm_wgrad_api,
    padded_ks_from_meta,
    run_deepgemm_wgrad,
    to_deepgemm_k_major_operand,
)
from sm90_fp8_wgrad.layout_mode import LAYOUT_MODE_PADDED
from sm90_fp8_wgrad.quant import quantize_grouped_wgrad_operand_128x1
from sm90_fp8_wgrad.sonic_baseline import make_sonic_outputs, run_sonic_down, run_sonic_gate_up
from sm90_fp8_wgrad.synthetic import (
    finite_status,
    make_synthetic_wgrad_inputs,
    route_stats,
    run_custom_down,
    run_custom_gate_up,
    run_custom_total,
    source_bf16_wgrad_ref,
)
from sm90_fp8_wgrad.timing import summarize_ms, tflops, time_cuda, valid_wgrad_flops

TensorSource = torch.Tensor | Callable[[], torch.Tensor]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute-only Wgrad benchmark: Sonic BF16 vs DeepGEMM FP8 vs custom CuTe FP8.")
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=768)
    p.add_argument("--route", choices=("balanced", "skewed"), default="skewed")
    p.add_argument("--counts-file")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--active-iters", type=int, default=1000)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--include-sonic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-deepgemm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-custom", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--custom-layout-mode", choices=("padded", "packed"), default="padded")
    p.add_argument("--custom-tuned-config", default="default")
    p.add_argument("--deepseek-diff-gate", type=float, default=1.0e-3)
    p.add_argument("--require-deepgemm-calc-diff", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="artifacts/wgrad")
    p.add_argument(
        "--results-json",
        help="Optional explicit path for the full machine-readable JSON result. Defaults to <output-dir>/results.json.",
    )
    return p.parse_args()


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "<missing>"


def module_path(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return f"<import failed: {exc!r}>"
    return str(getattr(mod, "__file__", "<builtin>"))


def source_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src" / "sm90_fp8_wgrad").glob("*.py")):
        data = path.read_bytes()
        h.update(path.relative_to(REPO_ROOT).as_posix().encode())
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def collect_env() -> dict:
    env = {
        "python": sys.version.replace("\n", " "),
        "torch": getattr(torch, "__version__", "<unknown>"),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_has_float8_e4m3fn": hasattr(torch, "float8_e4m3fn"),
        "source_hash": source_hash(),
        "packages": {
            "triton": package_version("triton"),
            "cuda-python": package_version("cuda-python"),
            "nvidia-cutlass-dsl": package_version("nvidia-cutlass-dsl"),
            "quack-kernels": package_version("quack-kernels"),
            "sonic-moe": package_version("sonic-moe"),
            "deep_gemm": package_version("deep_gemm"),
        },
        "modules": {
            "quack": module_path("quack"),
            "sonicmoe": module_path("sonicmoe"),
            "deep_gemm": module_path("deep_gemm"),
        },
    }
    if torch.cuda.is_available():
        env["gpu0"] = torch.cuda.get_device_name(0)
        env["capability0"] = tuple(int(v) for v in torch.cuda.get_device_capability(0))
    return env


def error_tail(text: str, lines: int = 20) -> str:
    return "\n".join(str(text).splitlines()[-int(lines) :])


def failed_result(case: str, label: str, exc_text: str) -> dict:
    return {
        "case": case,
        "case_label": label,
        "status": "failed",
        "verdict": "failed",
        "error": str(exc_text),
        "error_tail": error_tail(exc_text),
    }


def release_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def materialize_tensor(source: TensorSource) -> torch.Tensor:
    tensor = source() if callable(source) else source
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"expected a tensor, got {type(tensor)!r}")
    return tensor


def implementation_provenance(case: str, env: dict, *, extra: dict | None = None) -> dict:
    packages = env.get("packages", {})
    modules = env.get("modules", {})
    if case == "sonic_bf16":
        deps = {
            "sonic-moe": packages.get("sonic-moe", "<unknown>"),
            "quack-kernels": packages.get("quack-kernels", "<unknown>"),
        }
        module_paths = {"sonicmoe": modules.get("sonicmoe", "<unknown>"), "quack": modules.get("quack", "<unknown>")}
    elif case == "deepgemm_fp8":
        deps = {"deep_gemm": packages.get("deep_gemm", "<unknown>")}
        module_paths = {"deep_gemm": modules.get("deep_gemm", "<unknown>")}
    elif case == "custom_cute_fp8_bf16_out":
        deps = {
            "quack-kernels": packages.get("quack-kernels", "<unknown>"),
            "nvidia-cutlass-dsl": packages.get("nvidia-cutlass-dsl", "<unknown>"),
            "cuda-python": packages.get("cuda-python", "<unknown>"),
        }
        module_paths = {"quack": modules.get("quack", "<unknown>")}
    else:
        deps = {}
        module_paths = {}

    data = {
        "case": case,
        "source_hash": env.get("source_hash", "<unknown>"),
        "packages": deps,
        "module_paths": module_paths,
    }
    if extra:
        data.update(extra)
    return data


def benchmark_case(
    *,
    case: str,
    label: str,
    down_fn: Callable[[], None],
    gate_up_fn: Callable[[], None],
    total_fn: Callable[[], None],
    finite_tensors: Callable[[], dict[str, torch.Tensor]],
    args: argparse.Namespace,
    env: dict,
    down_flops: float,
    gate_up_flops: float,
    total_flops: float,
    provenance_extra: dict | None = None,
) -> dict:
    try:
        down_fn()
        gate_up_fn()
        torch.cuda.synchronize()
        finite = finite_status(finite_tensors())
        if not all(finite.values()):
            raise RuntimeError(f"non-finite output tensors: {finite}")

        down_stats = time_cuda(down_fn, warmup_iters=args.warmup_iters, active_iters=args.active_iters, repeat=args.repeat)
        gate_stats = time_cuda(
            gate_up_fn,
            warmup_iters=args.warmup_iters,
            active_iters=args.active_iters,
            repeat=args.repeat,
        )
        total_stats = time_cuda(total_fn, warmup_iters=args.warmup_iters, active_iters=args.active_iters, repeat=args.repeat)
        return {
            "case": case,
            "case_label": label,
            "status": "ok",
            "verdict": "ok",
            "correctness_status": "finite_smoke_ok",
            "implementation_provenance": implementation_provenance(case, env, extra=provenance_extra),
            "down_ms": down_stats.mean_ms,
            "down_min_ms": down_stats.min_ms,
            "gate_up_ms": gate_stats.mean_ms,
            "gate_up_min_ms": gate_stats.min_ms,
            "total_ms": total_stats.mean_ms,
            "total_min_ms": total_stats.min_ms,
            "total_p90_ms": total_stats.p90_ms,
            "total_p95_ms": total_stats.p95_ms,
            "total_std_ms": total_stats.std_ms,
            "down_valid_tflops": tflops(down_flops, down_stats.mean_ms),
            "gate_up_valid_tflops": tflops(gate_up_flops, gate_stats.mean_ms),
            "total_valid_tflops": tflops(total_flops, total_stats.mean_ms),
            "finite": finite,
            "timing_repeats_ms": {
                "down": down_stats.repeats,
                "gate_up": gate_stats.repeats,
                "total": total_stats.repeats,
            },
            "error_tail": "",
        }
    except Exception:
        row = failed_result(case, label, traceback.format_exc(limit=12))
        row["implementation_provenance"] = implementation_provenance(case, env, extra=provenance_extra)
        return row


def attach_deepseek_correctness(
    row: dict,
    *,
    down_actual: TensorSource,
    down_expected: TensorSource,
    gate_up_actual: TensorSource,
    gate_up_expected: TensorSource,
    reference: str,
    args: argparse.Namespace,
) -> None:
    if row.get("status") != "ok":
        return
    try:
        down_actual_t = materialize_tensor(down_actual)
        down_expected_t = materialize_tensor(down_expected)
        down_stats = scaled_wgrad_stats(
            down_actual_t,
            down_expected_t,
            require_deepgemm_calc_diff=bool(args.require_deepgemm_calc_diff),
        )
        del down_actual_t, down_expected_t
        release_cuda_cache()

        gate_up_actual_t = materialize_tensor(gate_up_actual)
        gate_up_expected_t = materialize_tensor(gate_up_expected)
        gate_stats = scaled_wgrad_stats(
            gate_up_actual_t,
            gate_up_expected_t,
            require_deepgemm_calc_diff=bool(args.require_deepgemm_calc_diff),
        )
        del gate_up_actual_t, gate_up_expected_t
        release_cuda_cache()
        max_calc_diff = max(float(down_stats["deepseek_calc_diff"]), float(gate_stats["deepseek_calc_diff"]))
        row["correctness"] = {
            "metric": "deep_gemm.testing.calc_diff",
            "reference": reference,
            "deepseek_diff_gate": float(args.deepseek_diff_gate),
            "down": down_stats,
            "gate_up": gate_stats,
            "max_deepseek_calc_diff": max_calc_diff,
        }
        row["down_deepseek_calc_diff"] = float(down_stats["deepseek_calc_diff"])
        row["gate_up_deepseek_calc_diff"] = float(gate_stats["deepseek_calc_diff"])
        row["max_deepseek_calc_diff"] = max_calc_diff
        if max_calc_diff <= float(args.deepseek_diff_gate):
            row["correctness_status"] = "deepseek_calc_diff_ok"
            row["verdict"] = "ok"
        else:
            row["correctness_status"] = "deepseek_calc_diff_failed"
            row["verdict"] = "failed"
            row["status"] = "failed"
            row["error"] = (
                f"DeepSeek calc_diff max={max_calc_diff:.6e} exceeds gate "
                f"{float(args.deepseek_diff_gate):.6e}; "
                f"down={float(down_stats['deepseek_calc_diff']):.6e}; "
                f"gate_up={float(gate_stats['deepseek_calc_diff']):.6e}"
            )
            row["error_tail"] = row["error"]
    except Exception:
        row["correctness_status"] = "deepseek_calc_diff_error"
        row["verdict"] = "failed"
        row["status"] = "failed"
        row["error"] = traceback.format_exc(limit=12)
        row["error_tail"] = error_tail(row["error"])


def source_bf16_wgrad_pairs(
    actual: torch.Tensor,
    lhs_src: torch.Tensor,
    rhs_src: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    actual_transpose: bool = False,
):
    cu = cu_seqlens_k.detach().cpu().tolist()
    for expert in range(len(cu) - 1):
        start = int(cu[expert])
        end = int(cu[expert + 1])
        m = int(lhs_src.shape[1])
        n = int(rhs_src.shape[1])
        expected = torch.zeros((m, n), device=lhs_src.device, dtype=torch.float32)
        if end > start:
            expected = lhs_src[start:end].float().T @ rhs_src[start:end].float()
        actual_e = actual[expert].T if actual_transpose else actual[expert]
        expected_out = expected.to(out_dtype)
        if expected_out is not expected:
            del expected
        yield actual_e, expected_out


def scaled_fp8_wgrad_pairs(
    actual: torch.Tensor,
    lhs,
    rhs,
    meta,
    *,
    out_dtype: torch.dtype,
):
    for expert, expected in iter_ref_scaled_wgrad_by_expert(lhs, rhs, meta, out_dtype=out_dtype):
        yield actual[int(expert)], expected


def attach_deepseek_correctness_stream(
    row: dict,
    *,
    down_pairs: Callable[[], object],
    gate_up_pairs: Callable[[], object],
    reference: str,
    args: argparse.Namespace,
) -> None:
    if row.get("status") != "ok":
        return
    try:
        down_diff = deepseek_calc_diff_stream(
            down_pairs(),
            require_deepgemm=bool(args.require_deepgemm_calc_diff),
        )
        release_cuda_cache()
        gate_up_diff = deepseek_calc_diff_stream(
            gate_up_pairs(),
            require_deepgemm=bool(args.require_deepgemm_calc_diff),
        )
        release_cuda_cache()
        down_stats = {"deepseek_calc_diff": float(down_diff)}
        gate_stats = {"deepseek_calc_diff": float(gate_up_diff)}
        max_calc_diff = max(float(down_diff), float(gate_up_diff))
        row["correctness"] = {
            "metric": "deep_gemm.testing.calc_diff",
            "reference": reference,
            "reference_mode": "streamed_by_expert",
            "deepseek_diff_gate": float(args.deepseek_diff_gate),
            "down": down_stats,
            "gate_up": gate_stats,
            "max_deepseek_calc_diff": max_calc_diff,
        }
        row["down_deepseek_calc_diff"] = float(down_diff)
        row["gate_up_deepseek_calc_diff"] = float(gate_up_diff)
        row["max_deepseek_calc_diff"] = max_calc_diff
        if max_calc_diff <= float(args.deepseek_diff_gate):
            row["correctness_status"] = "deepseek_calc_diff_ok"
            row["verdict"] = "ok"
        else:
            row["correctness_status"] = "deepseek_calc_diff_failed"
            row["verdict"] = "failed"
            row["status"] = "failed"
            row["error"] = (
                f"DeepSeek calc_diff max={max_calc_diff:.6e} exceeds gate "
                f"{float(args.deepseek_diff_gate):.6e}; "
                f"down={float(down_diff):.6e}; gate_up={float(gate_up_diff):.6e}"
            )
            row["error_tail"] = row["error"]
    except Exception:
        row["correctness_status"] = "deepseek_calc_diff_error"
        row["verdict"] = "failed"
        row["status"] = "failed"
        row["error"] = traceback.format_exc(limit=12)
        row["error_tail"] = error_tail(row["error"])


def _attach_output_paths_and_logs(out_dir: Path, summary: dict, *, results_json: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary["output_paths"] = {
        "output_dir": str(out_dir),
        "results_json": str(results_json),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
        "results_csv": str(out_dir / "results.csv"),
        "env_json": str(out_dir / "env.json"),
        "logs_dir": str(logs_dir),
    }
    for row in summary.get("results", []):
        row.setdefault("implementation_provenance", implementation_provenance(str(row.get("case", "")), summary["env"]))
        if row.get("status") != "ok":
            log_path = logs_dir / f"{row.get('case', 'unknown')}.error.log"
            log_text = str(row.get("error", row.get("error_tail", "")))
            log_path.write_text(log_text.rstrip() + "\n", encoding="utf-8")
            row["error_log_path"] = str(log_path)


def write_outputs(out_dir: Path, summary: dict, *, results_json: Path) -> None:
    _attach_output_paths_and_logs(out_dir, summary, results_json=results_json)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "env.json").write_text(json.dumps(summary["env"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_route_stats = summary.get("route_stats") or {}

    fields = [
        "case",
        "case_label",
        "status",
        "tokens",
        "valid_rows",
        "padded_rows",
        "min_rows_per_expert",
        "median_rows_per_expert",
        "max_rows_per_expert",
        "down_ms",
        "gate_up_ms",
        "total_ms",
        "speed_vs_sonic",
        "down_valid_tflops",
        "gate_up_valid_tflops",
        "total_valid_tflops",
        "gpu_memory_peak_gb",
        "gpu_memory_peak_bytes",
        "down_deepseek_calc_diff",
        "gate_up_deepseek_calc_diff",
        "max_deepseek_calc_diff",
        "correctness_status",
        "verdict",
        "error_log_path",
        "error_tail",
    ]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Wgrad Benchmark Summary",
        "",
        f"status: `{summary['status']}`",
        f"tokens: `{summary['tokens']}`",
        f"route: `{summary['route']}`",
        f"valid_rows: `{summary_route_stats.get('valid_rows', '')}`",
        f"source_hash: `{summary['env']['source_hash']}`",
        f"deepseek_diff_gate: `{summary['deepseek_diff_gate']}`",
        f"results_json: `{summary['output_paths']['results_json']}`",
        "",
        "| case | status | down ms | gate/up ms | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        speed = row.get("speed_vs_sonic")
        speed_s = "" if speed is None or speed == "" else f"{float(speed):.4f}x"
        calc_diff = row.get("max_deepseek_calc_diff")
        calc_diff_s = "" if calc_diff is None or calc_diff == "" else f"{float(calc_diff):.6e}"
        if row.get("status") == "ok":
            lines.append(
                f"| {row['case_label']} | ok | {row['down_ms']:.6f} | {row['gate_up_ms']:.6f} | "
                f"{row['total_ms']:.6f} | {speed_s} | {row['total_valid_tflops']:.3f} | {calc_diff_s} |"
            )
        else:
            lines.append(f"| {row['case_label']} | failed |  |  |  |  |  | {calc_diff_s} |")
    failed = [row for row in summary["results"] if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.extend(
                [
                    f"### {row['case_label']}",
                    "",
                    f"log: `{row.get('error_log_path', '')}`",
                    "",
                    "```text",
                    row.get("error_tail", ""),
                    "```",
                    "",
                ]
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_cases(args: argparse.Namespace) -> list[str]:
    cases: list[str] = []
    if bool(args.include_sonic):
        cases.append("sonic_bf16")
    if bool(args.include_custom):
        cases.append("custom_cute_fp8_bf16_out")
    if bool(args.include_deepgemm):
        cases.append("deepgemm_fp8")
    return cases


def make_inputs_for_case(args: argparse.Namespace, case: str):
    needs_custom_buffers = case == "custom_cute_fp8_bf16_out"
    return make_synthetic_wgrad_inputs(
        tokens=args.tokens,
        experts=args.experts,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        route=args.route,
        seed=args.seed,
        counts_file=args.counts_file,
        dtype=torch.bfloat16,
        layout_mode=args.custom_layout_mode,
        build_fp8_operands=needs_custom_buffers,
        allocate_custom_outputs=needs_custom_buffers,
    )


def require_tensor(value: torch.Tensor | None, name: str) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"{name} was not allocated")
    return value


def require_operand(value, name: str):
    if value is None:
        raise RuntimeError(f"{name} was not allocated")
    return value


def annotate_row(args: argparse.Namespace, row: dict, inp, stats: dict) -> None:
    row.update(
        {
            "tokens": int(args.tokens),
            "valid_rows": int(inp.valid_rows),
            "padded_rows": int(inp.meta.padded_rows),
            "route_stats": stats,
            "min_rows_per_expert": stats["min_rows_per_expert"],
            "median_rows_per_expert": stats["median_rows_per_expert"],
            "max_rows_per_expert": stats["max_rows_per_expert"],
        }
    )
    if torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated())
        row["gpu_memory_peak_bytes"] = peak_bytes
        row["gpu_memory_peak_gb"] = float(peak_bytes) / 1.0e9


def run_one_case(
    *,
    case: str,
    args: argparse.Namespace,
    env: dict,
) -> tuple[dict, dict]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inp = None
    stats: dict = {"valid_rows": int(args.tokens) * int(args.top_k)}
    try:
        inp = make_inputs_for_case(args, case)
        stats = route_stats(inp.counts, inp.meta)
        down_flops, gate_up_flops, total_flops = valid_wgrad_flops(inp.valid_rows, args.hidden, args.intermediate)

        if case == "sonic_bf16":
            sonic_dw2, sonic_dw1 = make_sonic_outputs(
                experts=args.experts,
                hidden=args.hidden,
                intermediate=args.intermediate,
                device=inp.x_bf16.device,
                dtype=torch.bfloat16,
            )
            row = benchmark_case(
                case="sonic_bf16",
                label="Sonic BF16",
                down_fn=lambda: run_sonic_down(
                    dout=inp.dout_bf16,
                    a_prime=inp.hidden_grouped_bf16,
                    dw2=sonic_dw2,
                    expert_frequency_offset=inp.cu_seqlens_k,
                    x_gather_idx=inp.x_gather_idx,
                ),
                gate_up_fn=lambda: run_sonic_gate_up(
                    x=inp.x_bf16,
                    dh=inp.grad_gate_up_grouped_bf16,
                    dw1=sonic_dw1,
                    expert_frequency_offset=inp.cu_seqlens_k,
                    x_gather_idx=inp.x_gather_idx,
                ),
                total_fn=lambda: (
                    run_sonic_down(
                        dout=inp.dout_bf16,
                        a_prime=inp.hidden_grouped_bf16,
                        dw2=sonic_dw2,
                        expert_frequency_offset=inp.cu_seqlens_k,
                        x_gather_idx=inp.x_gather_idx,
                    ),
                    run_sonic_gate_up(
                        x=inp.x_bf16,
                        dh=inp.grad_gate_up_grouped_bf16,
                        dw1=sonic_dw1,
                        expert_frequency_offset=inp.cu_seqlens_k,
                        x_gather_idx=inp.x_gather_idx,
                    ),
                ),
                finite_tensors=lambda: {"sonic_dw2": sonic_dw2, "sonic_dw1": sonic_dw1},
                args=args,
                env=env,
                down_flops=down_flops,
                gate_up_flops=gate_up_flops,
                total_flops=total_flops,
            )
            attach_deepseek_correctness_stream(
                row,
                down_pairs=lambda: source_bf16_wgrad_pairs(
                    sonic_dw2,
                    inp.grad_y_grouped_bf16,
                    inp.hidden_grouped_bf16,
                    inp.cu_seqlens_k,
                    out_dtype=torch.bfloat16,
                ),
                gate_up_pairs=lambda: source_bf16_wgrad_pairs(
                    sonic_dw1,
                    inp.grad_gate_up_grouped_bf16,
                    inp.x_grouped_bf16,
                    inp.cu_seqlens_k,
                    out_dtype=torch.bfloat16,
                    actual_transpose=True,
                ),
                reference="bf16_grouped_source",
                args=args,
            )
        elif case == "custom_cute_fp8_bf16_out":
            down_out = require_tensor(inp.down_out_bf16, "custom down output")
            gate_up_out = require_tensor(inp.gate_up_out_bf16, "custom gate/up output")
            row = benchmark_case(
                case="custom_cute_fp8_bf16_out",
                label="Custom CuTe FP8 BF16-out",
                down_fn=lambda: run_custom_down(inp, tuned_config=args.custom_tuned_config),
                gate_up_fn=lambda: run_custom_gate_up(inp, tuned_config=args.custom_tuned_config),
                total_fn=lambda: run_custom_total(inp, tuned_config=args.custom_tuned_config),
                finite_tensors=lambda: {"custom_down": down_out, "custom_gate_up": gate_up_out},
                args=args,
                env=env,
                down_flops=down_flops,
                gate_up_flops=gate_up_flops,
                total_flops=total_flops,
                provenance_extra={
                    "custom_layout_mode": str(args.custom_layout_mode),
                    "custom_tuned_config": str(args.custom_tuned_config),
                },
            )
            attach_deepseek_correctness_stream(
                row,
                down_pairs=lambda: scaled_fp8_wgrad_pairs(
                    down_out,
                    require_operand(inp.grad_y_q, "custom grad_y_q"),
                    require_operand(inp.hidden_q, "custom hidden_q"),
                    inp.meta,
                    out_dtype=torch.bfloat16,
                ),
                gate_up_pairs=lambda: scaled_fp8_wgrad_pairs(
                    gate_up_out,
                    require_operand(inp.grad_gate_up_q, "custom grad_gate_up_q"),
                    require_operand(inp.x_q, "custom x_q"),
                    inp.meta,
                    out_dtype=torch.bfloat16,
                ),
                reference="dequantized_fp8_operands",
                args=args,
            )
        elif case == "deepgemm_fp8":
            _fn, dg_name = load_deepgemm_wgrad_api()
            ks_host, ks_tensor = padded_ks_from_meta(inp.meta)
            dg_grad_y_q = quantize_grouped_wgrad_operand_128x1(
                inp.grad_y_grouped_bf16,
                inp.meta,
                layout_mode=LAYOUT_MODE_PADDED,
            )
            dg_hidden_q = quantize_grouped_wgrad_operand_128x1(
                inp.hidden_grouped_bf16,
                inp.meta,
                layout_mode=LAYOUT_MODE_PADDED,
            )
            dg_grad_gate_up_q = quantize_grouped_wgrad_operand_128x1(
                inp.grad_gate_up_grouped_bf16,
                inp.meta,
                layout_mode=LAYOUT_MODE_PADDED,
            )
            dg_x_q = quantize_grouped_wgrad_operand_128x1(
                inp.x_grouped_bf16,
                inp.meta,
                layout_mode=LAYOUT_MODE_PADDED,
            )
            dg_grad_y = to_deepgemm_k_major_operand(dg_grad_y_q, ks_host=ks_host)
            dg_hidden = to_deepgemm_k_major_operand(dg_hidden_q, ks_host=ks_host)
            dg_grad_gate_up = to_deepgemm_k_major_operand(dg_grad_gate_up_q, ks_host=ks_host)
            dg_x = to_deepgemm_k_major_operand(dg_x_q, ks_host=ks_host)
            dg_down_out = torch.empty((args.experts, args.hidden, args.intermediate), device=inp.x_bf16.device, dtype=torch.float32)
            dg_down_zero = torch.zeros_like(dg_down_out)
            down_fn = lambda: run_deepgemm_wgrad(
                lhs=dg_grad_y,
                rhs=dg_hidden,
                out=dg_down_out,
                ks_host=ks_host,
                ks_tensor=ks_tensor,
                zero_c=dg_down_zero,
            )
            down_fn()
            torch.cuda.synchronize()
            down_finite = finite_status({"deepgemm_down": dg_down_out})
            if not all(down_finite.values()):
                raise RuntimeError(f"non-finite DeepGEMM down output: {down_finite}")
            down_stats = time_cuda(
                down_fn,
                warmup_iters=args.warmup_iters,
                active_iters=args.active_iters,
                repeat=args.repeat,
            )
            del dg_down_zero
            release_cuda_cache()
            down_diff = deepseek_calc_diff_stream(
                scaled_fp8_wgrad_pairs(dg_down_out, dg_grad_y_q, dg_hidden_q, inp.meta, out_dtype=torch.float32),
                require_deepgemm=bool(args.require_deepgemm_calc_diff),
            )
            del dg_down_out
            release_cuda_cache()

            dg_gate_out = torch.empty((args.experts, 2 * args.intermediate, args.hidden), device=inp.x_bf16.device, dtype=torch.float32)
            dg_gate_zero = torch.zeros_like(dg_gate_out)
            gate_fn = lambda: run_deepgemm_wgrad(
                lhs=dg_grad_gate_up,
                rhs=dg_x,
                out=dg_gate_out,
                ks_host=ks_host,
                ks_tensor=ks_tensor,
                zero_c=dg_gate_zero,
            )
            gate_fn()
            torch.cuda.synchronize()
            gate_finite = finite_status({"deepgemm_gate_up": dg_gate_out})
            if not all(gate_finite.values()):
                raise RuntimeError(f"non-finite DeepGEMM gate/up output: {gate_finite}")
            gate_stats = time_cuda(
                gate_fn,
                warmup_iters=args.warmup_iters,
                active_iters=args.active_iters,
                repeat=args.repeat,
            )
            del dg_gate_zero
            release_cuda_cache()
            gate_diff = deepseek_calc_diff_stream(
                scaled_fp8_wgrad_pairs(dg_gate_out, dg_grad_gate_up_q, dg_x_q, inp.meta, out_dtype=torch.float32),
                require_deepgemm=bool(args.require_deepgemm_calc_diff),
            )
            del dg_gate_out
            release_cuda_cache()

            total_repeats = [float(d) + float(g) for d, g in zip(down_stats.repeats, gate_stats.repeats)]
            total_stats = summarize_ms(total_repeats or [float(down_stats.mean_ms) + float(gate_stats.mean_ms)])
            max_calc_diff = max(float(down_diff), float(gate_diff))
            row = {
                "case": "deepgemm_fp8",
                "case_label": "DeepGEMM FP8",
                "status": "ok",
                "verdict": "ok",
                "correctness_status": "deepseek_calc_diff_ok",
                "implementation_provenance": implementation_provenance(
                    "deepgemm_fp8",
                    env,
                    extra={"deepgemm_api": dg_name, "stage_execution": "streamed_down_then_gate_up"},
                ),
                "deepgemm_api": dg_name,
                "stage_execution": "streamed_down_then_gate_up",
                "total_timing_mode": "sum_down_gate_up_repeats",
                "down_ms": down_stats.mean_ms,
                "down_min_ms": down_stats.min_ms,
                "gate_up_ms": gate_stats.mean_ms,
                "gate_up_min_ms": gate_stats.min_ms,
                "total_ms": total_stats.mean_ms,
                "total_min_ms": total_stats.min_ms,
                "total_p90_ms": total_stats.p90_ms,
                "total_p95_ms": total_stats.p95_ms,
                "total_std_ms": total_stats.std_ms,
                "down_valid_tflops": tflops(down_flops, down_stats.mean_ms),
                "gate_up_valid_tflops": tflops(gate_up_flops, gate_stats.mean_ms),
                "total_valid_tflops": tflops(total_flops, total_stats.mean_ms),
                "finite": {**down_finite, **gate_finite},
                "timing_repeats_ms": {
                    "down": down_stats.repeats,
                    "gate_up": gate_stats.repeats,
                    "total": total_stats.repeats,
                },
                "correctness": {
                    "metric": "deep_gemm.testing.calc_diff",
                    "reference": "dequantized_fp8_operands_deepgemm_padded",
                    "reference_mode": "streamed_by_expert",
                    "deepseek_diff_gate": float(args.deepseek_diff_gate),
                    "down": {"deepseek_calc_diff": float(down_diff)},
                    "gate_up": {"deepseek_calc_diff": float(gate_diff)},
                    "max_deepseek_calc_diff": max_calc_diff,
                },
                "down_deepseek_calc_diff": float(down_diff),
                "gate_up_deepseek_calc_diff": float(gate_diff),
                "max_deepseek_calc_diff": max_calc_diff,
                "error_tail": "",
            }
            if max_calc_diff > float(args.deepseek_diff_gate):
                row["correctness_status"] = "deepseek_calc_diff_failed"
                row["verdict"] = "failed"
                row["status"] = "failed"
                row["error"] = (
                    f"DeepSeek calc_diff max={max_calc_diff:.6e} exceeds gate "
                    f"{float(args.deepseek_diff_gate):.6e}; "
                    f"down={float(down_diff):.6e}; gate_up={float(gate_diff):.6e}"
                )
                row["error_tail"] = row["error"]
        else:
            raise RuntimeError(f"unsupported case {case!r}")

        annotate_row(args, row, inp, stats)
        return row, stats
    except Exception:
        labels = {
            "sonic_bf16": "Sonic BF16",
            "custom_cute_fp8_bf16_out": "Custom CuTe FP8 BF16-out",
            "deepgemm_fp8": "DeepGEMM FP8",
        }
        row = failed_result(case, labels.get(case, case), traceback.format_exc(limit=12))
        row["implementation_provenance"] = implementation_provenance(case, env)
        if inp is not None:
            annotate_row(args, row, inp, stats)
        else:
            row.update(
                {
                    "tokens": int(args.tokens),
                    "valid_rows": int(args.tokens) * int(args.top_k),
                    "padded_rows": "",
                    "route_stats": stats,
                    "min_rows_per_expert": "",
                    "median_rows_per_expert": "",
                    "max_rows_per_expert": "",
                }
            )
            if torch.cuda.is_available():
                peak_bytes = int(torch.cuda.max_memory_allocated())
                row["gpu_memory_peak_bytes"] = peak_bytes
                row["gpu_memory_peak_gb"] = float(peak_bytes) / 1.0e9
        return row, stats
    finally:
        release_cuda_cache()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    env = collect_env()
    cases = selected_cases(args)

    if not torch.cuda.is_available():
        row = failed_result("all", "all", "CUDA is required")
        row.update(
            {
                "tokens": int(args.tokens),
                "valid_rows": int(args.tokens) * int(args.top_k),
                "padded_rows": "",
                "min_rows_per_expert": "",
                "median_rows_per_expert": "",
                "max_rows_per_expert": "",
                "speed_vs_sonic": None,
            }
        )
        summary = {
            "status": "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "tokens": int(args.tokens),
            "route": str(args.route),
            "deepgemm_metric": "deep_gemm.testing.calc_diff",
            "deepseek_diff_gate": float(args.deepseek_diff_gate),
            "require_deepgemm_calc_diff": bool(args.require_deepgemm_calc_diff),
            "env": env,
            "route_stats": {"valid_rows": int(args.tokens) * int(args.top_k)},
            "results": [row],
        }
        write_outputs(out_dir, summary, results_json=results_json)
        return 1

    results: list[dict] = []
    stats: dict | None = None

    if not cases:
        row = failed_result("all", "all", "no benchmark cases selected")
        row.update(
            {
                "tokens": int(args.tokens),
                "valid_rows": int(args.tokens) * int(args.top_k),
                "padded_rows": "",
                "min_rows_per_expert": "",
                "median_rows_per_expert": "",
                "max_rows_per_expert": "",
                "speed_vs_sonic": None,
            }
        )
        results.append(row)
    else:
        for case in cases:
            row, case_stats = run_one_case(case=case, args=args, env=env)
            release_cuda_cache()
            stats = case_stats
            results.append(row)

    sonic = next((row for row in results if row.get("case") == "sonic_bf16" and row.get("status") == "ok"), None)
    sonic_total = float(sonic["total_ms"]) if sonic is not None else None
    for row in results:
        if row.get("status") == "ok" and sonic_total is not None:
            row["speed_vs_sonic"] = sonic_total / float(row["total_ms"])
        else:
            row["speed_vs_sonic"] = None
    route_summary = stats if stats is not None else {"valid_rows": int(args.tokens) * int(args.top_k)}

    summary = {
        "status": "ok" if any(row.get("status") == "ok" for row in results) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "execution_mode": "isolated_per_case",
        "selected_cases": cases,
        "tokens": int(args.tokens),
        "route": str(args.route),
        "seed": int(args.seed),
        "deepgemm_metric": "deep_gemm.testing.calc_diff",
        "deepseek_diff_gate": float(args.deepseek_diff_gate),
        "require_deepgemm_calc_diff": bool(args.require_deepgemm_calc_diff),
        "shape": {
            "experts": int(args.experts),
            "top_k": int(args.top_k),
            "hidden": int(args.hidden),
            "intermediate": int(args.intermediate),
        },
        "route_stats": route_summary,
        "env": env,
        "results": results,
    }
    write_outputs(out_dir, summary, results_json=results_json)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
