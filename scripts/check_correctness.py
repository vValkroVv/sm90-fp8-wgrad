#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from sm90_fp8_wgrad.checks import assert_scaled_wgrad_close, ref_scaled_wgrad
from sm90_fp8_wgrad.synthetic import (
    dtype_from_name,
    make_synthetic_wgrad_inputs,
    route_stats,
    run_custom_down,
    run_custom_gate_up,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check custom SM90 FP8 Wgrad against dequantized FP8 references.")
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=768)
    p.add_argument("--route", choices=("balanced", "skewed"), default="skewed")
    p.add_argument("--counts-file")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--layout-mode", choices=("padded", "packed"), default="padded")
    p.add_argument("--tuned-config", default="default")
    p.add_argument("--deepseek-diff-gate", type=float, default=1.0e-3)
    p.add_argument("--require-deepgemm-calc-diff", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="artifacts/correctness")
    return p.parse_args()


def write_outputs(output_dir: Path, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary.get("status") != "ok":
        lines = [
            "# Correctness Summary",
            "",
            f"status: `{summary.get('status', 'failed')}`",
            "",
            "```text",
            str(summary.get("error", "")),
            "```",
        ]
        (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# Correctness Summary",
        "",
        f"status: `{summary['status']}`",
        f"tokens: `{summary['tokens']}`",
        f"route: `{summary['route']}`",
        f"valid_rows: `{summary['route_stats']['valid_rows']}`",
        f"deepseek_diff_gate: `{summary['deepseek_diff_gate']}`",
        "",
        "| stage | scaled-FP8 status | DeepSeek calc_diff |",
        "| --- | --- | ---: |",
    ]
    for stage in ("down", "gate_up"):
        item = summary["stages"][stage]
        lines.append(
            f"| {stage} | {item['scaled_fp8_status']} | "
            f"{item['scaled_fp8_error']['deepseek_calc_diff']:.6e} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dtype = dtype_from_name(args.output_dtype)
    try:
        inp = make_synthetic_wgrad_inputs(
            tokens=args.tokens,
            experts=args.experts,
            top_k=args.top_k,
            hidden=args.hidden,
            intermediate=args.intermediate,
            route=args.route,
            seed=args.seed,
            counts_file=args.counts_file,
            dtype=torch.bfloat16,
            layout_mode=args.layout_mode,
        )

        run_custom_down(inp, tuned_config=args.tuned_config)
        run_custom_gate_up(inp, tuned_config=args.tuned_config)
        torch.cuda.synchronize()

        down_scaled_ref = ref_scaled_wgrad(inp.grad_y_q, inp.hidden_q, inp.meta, out_dtype=out_dtype)
        gate_scaled_ref = ref_scaled_wgrad(inp.grad_gate_up_q, inp.x_q, inp.meta, out_dtype=out_dtype)
        down_scaled_stats = assert_scaled_wgrad_close(
            inp.down_out_bf16.to(out_dtype),
            down_scaled_ref,
            out_dtype=out_dtype,
            deepseek_diff_limit=float(args.deepseek_diff_gate),
            require_deepgemm_calc_diff=bool(args.require_deepgemm_calc_diff),
        )
        gate_scaled_stats = assert_scaled_wgrad_close(
            inp.gate_up_out_bf16.to(out_dtype),
            gate_scaled_ref,
            out_dtype=out_dtype,
            deepseek_diff_limit=float(args.deepseek_diff_gate),
            require_deepgemm_calc_diff=bool(args.require_deepgemm_calc_diff),
        )

        summary = {
            "status": "ok",
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
            "route_stats": route_stats(inp.counts, inp.meta),
            "stages": {
                "down": {
                    "scaled_fp8_status": "ok",
                    "scaled_fp8_error": dict(down_scaled_stats),
                },
                "gate_up": {
                    "scaled_fp8_status": "ok",
                    "scaled_fp8_error": dict(gate_scaled_stats),
                },
            },
        }
    except Exception:
        summary = {
            "status": "failed",
            "tokens": int(args.tokens),
            "route": str(args.route),
            "error": traceback.format_exc(limit=12),
        }
        write_outputs(Path(args.output_dir), summary)
        print(summary["error"], file=sys.stderr)
        return 1

    write_outputs(Path(args.output_dir), summary)
    print((Path(args.output_dir) / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
