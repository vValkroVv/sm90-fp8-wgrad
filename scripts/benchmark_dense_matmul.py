#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata as md
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense CUDA matmul TFLOP/s sanity benchmark.")
    parser.add_argument(
        "--sizes",
        help="Comma-separated square GEMM sizes. Example: 8192,16384,32768. Overrides --m/--n/--k.",
    )
    parser.add_argument("--m", type=int, default=16384)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "tf32", "fp32"), default="bf16")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--active-iters", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="artifacts/dense_matmul")
    parser.add_argument("--results-json", help="Default: <output-dir>/results.json")
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "<missing>"


def parse_sizes(value: str | None, *, m: int, n: int, k: int) -> list[tuple[int, int, int]]:
    if not value:
        return [(int(m), int(n), int(k))]
    shapes: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    for raw in value.split(","):
        item = raw.strip().lower().replace("_", "")
        if not item:
            continue
        if item.endswith("k"):
            size = int(float(item[:-1]) * 1024)
        else:
            size = int(float(item))
        if size <= 0:
            raise RuntimeError(f"matrix size must be positive, got {size}")
        if size not in seen:
            shapes.append((size, size, size))
            seen.add(size)
    if not shapes:
        raise RuntimeError("--sizes did not contain any valid size")
    return shapes


def dtype_config(torch: Any, dtype_name: str) -> tuple[Any, str, int]:
    if dtype_name == "bf16":
        return torch.bfloat16, "BF16", 2
    if dtype_name == "fp16":
        return torch.float16, "FP16", 2
    if dtype_name == "tf32":
        return torch.float32, "TF32", 4
    if dtype_name == "fp32":
        return torch.float32, "FP32", 4
    raise RuntimeError(f"unsupported dtype {dtype_name}")


def configure_matmul_mode(torch: Any, dtype_name: str) -> None:
    if dtype_name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    elif dtype_name == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def summarize_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise RuntimeError("no timing samples")
    ordered = sorted(float(item) for item in samples)
    return {
        "min_ms": ordered[0],
        "mean_ms": sum(ordered) / len(ordered),
        "max_ms": ordered[-1],
    }


def time_cuda_ms(torch: Any, fn: Callable[[], Any], *, warmup_iters: int, active_iters: int) -> float:
    for _ in range(int(warmup_iters)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(active_iters)):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(active_iters)


def tflops(flops: float, ms: float) -> float:
    return float(flops) / (float(ms) * 1.0e-3) / 1.0e12


def error_tail(text: str, lines: int = 30) -> str:
    return "\n".join(str(text).splitlines()[-int(lines) :])


def collect_env(torch: Any) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "torch": getattr(torch, "__version__", "<unknown>"),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": {
            "triton": package_version("triton"),
        },
    }
    if torch.cuda.is_available():
        env["gpu0"] = torch.cuda.get_device_name(0)
        env["capability0"] = tuple(int(v) for v in torch.cuda.get_device_capability(0))
    return env


def benchmark_one(torch: Any, args: argparse.Namespace, *, m: int, n: int, k: int) -> dict[str, Any]:
    dtype, dtype_label, bytes_per_element = dtype_config(torch, args.dtype)
    configure_matmul_mode(torch, args.dtype)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    a = torch.empty((int(m), int(k)), device="cuda", dtype=dtype).normal_(mean=0.0, std=0.02)
    b = torch.empty((int(k), int(n)), device="cuda", dtype=dtype).normal_(mean=0.0, std=0.02)
    c = torch.empty((int(m), int(n)), device="cuda", dtype=dtype)
    torch.cuda.synchronize()

    def matmul() -> None:
        torch.mm(a, b, out=c)

    samples = [
        time_cuda_ms(torch, matmul, warmup_iters=int(args.warmup_iters), active_iters=int(args.active_iters))
        for _ in range(int(args.repeat))
    ]
    stats = summarize_ms(samples)
    flops = 2.0 * float(m) * float(n) * float(k)
    hbm_bytes = float(m * k + k * n + m * n) * float(bytes_per_element)
    row = {
        "status": "ok",
        "dtype": str(args.dtype),
        "dtype_label": dtype_label,
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "flops": flops,
        "hbm_bytes_lower_bound": hbm_bytes,
        "arithmetic_intensity_flops_per_byte": flops / hbm_bytes,
        "warmup_iters": int(args.warmup_iters),
        "active_iters": int(args.active_iters),
        "repeat": int(args.repeat),
        "samples_ms": samples,
        **stats,
        "best_tflops": tflops(flops, stats["min_ms"]),
        "mean_tflops": tflops(flops, stats["mean_ms"]),
        "worst_tflops": tflops(flops, stats["max_ms"]),
        "gpu_memory_allocated_gb": float(torch.cuda.memory_allocated()) / 1.0e9,
        "gpu_memory_peak_gb": float(torch.cuda.max_memory_allocated()) / 1.0e9,
    }
    del a, b, c
    torch.cuda.empty_cache()
    return row


def failed_row(args: argparse.Namespace, *, m: int, n: int, k: int, exc_text: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "dtype": str(args.dtype),
        "m": int(m),
        "n": int(n),
        "k": int(k),
        "error": exc_text,
        "error_tail": error_tail(exc_text),
    }


def write_outputs(out_dir: Path, summary: dict[str, Any], *, results_json: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["output_paths"] = {
        "output_dir": str(out_dir),
        "results_json": str(results_json),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
        "results_csv": str(out_dir / "results.csv"),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "status",
        "dtype",
        "m",
        "n",
        "k",
        "mean_ms",
        "min_ms",
        "max_ms",
        "mean_tflops",
        "best_tflops",
        "worst_tflops",
        "hbm_bytes_lower_bound",
        "arithmetic_intensity_flops_per_byte",
        "gpu_memory_peak_gb",
        "error_tail",
    ]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary["results"]:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Dense Matmul Benchmark",
        "",
        f"created_at: `{summary['created_at']}`",
        f"results_json: `{summary['output_paths']['results_json']}`",
        "",
        "FLOPs are counted as `2 * M * N * K`. HBM bytes are the lower bound: read A, read B, write C.",
        "",
        "| dtype | M | N | K | status | mean ms | best TFLOP/s | mean TFLOP/s | peak GB |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        if row.get("status") == "ok":
            lines.append(
                f"| {row['dtype_label']} | {row['m']} | {row['n']} | {row['k']} | ok | "
                f"{float(row['mean_ms']):.6f} | {float(row['best_tflops']):.3f} | "
                f"{float(row['mean_tflops']):.3f} | {float(row['gpu_memory_peak_gb']):.3f} |"
            )
        else:
            lines.append(
                f"| {row.get('dtype', '')} | {row.get('m', '')} | {row.get('n', '')} | "
                f"{row.get('k', '')} | failed |  |  |  |  |"
            )
    failed = [row for row in summary["results"] if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## failures", ""])
        for row in failed:
            lines.extend(
                [
                    f"### {row.get('dtype', '')} M={row.get('m')} N={row.get('n')} K={row.get('k')}",
                    "",
                    "```text",
                    str(row.get("error_tail", "")),
                    "```",
                    "",
                ]
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    import torch

    out_dir = Path(args.output_dir)
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense matmul benchmarking")

    rows: list[dict[str, Any]] = []
    for m, n, k in parse_sizes(args.sizes, m=args.m, n=args.n, k=args.k):
        try:
            rows.append(benchmark_one(torch, args, m=m, n=n, k=k))
        except Exception:
            exc_text = traceback.format_exc()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            rows.append(failed_row(args, m=m, n=n, k=k, exc_text=exc_text))

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": "ok" if any(row.get("status") == "ok" for row in rows) else "failed",
        "metric": "dense_matmul_tflops",
        "flop_formula": "2 * M * N * K",
        "hbm_lower_bound": "A read + B read + C write",
        "env": collect_env(torch),
        "results": rows,
    }
    write_outputs(out_dir, summary, results_json=results_json)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
