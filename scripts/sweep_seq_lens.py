#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "scripts" / "benchmark_wgrad.py"


def parse_token_count(value: str) -> int:
    text = str(value).strip().lower().replace("_", "")
    if not text:
        raise RuntimeError("empty token count")
    scale = 1
    for suffix in ("kib", "ki", "k"):
        if text.endswith(suffix):
            scale = 1024
            text = text[: -len(suffix)]
            break
    return int(float(text) * scale)


def parse_tokens_list(value: str) -> list[int]:
    tokens: list[int] = []
    seen: set[int] = set()
    for part in str(value).split(","):
        if not part.strip():
            continue
        parsed = parse_token_count(part)
        if parsed <= 0:
            raise RuntimeError(f"token count must be positive, got {parsed}")
        if parsed not in seen:
            tokens.append(parsed)
            seen.add(parsed)
    if not tokens:
        raise RuntimeError("--tokens-list must not be empty")
    return tokens


def token_label(tokens: int) -> str:
    if int(tokens) % 1024 == 0:
        return f"{int(tokens) // 1024}k"
    return str(int(tokens))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep Wgrad benchmark sequence lengths.")
    p.add_argument("--tokens-list", default="4096,8192,16384,32768,65536")
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=768)
    p.add_argument("--route", choices=("balanced", "skewed"), default="skewed")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup-iters", type=int, default=100)
    p.add_argument("--active-iters", type=int, default=1000)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--include-sonic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-deepgemm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-custom", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--isolate-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each selected implementation in its own benchmark process.",
    )
    p.add_argument("--custom-layout-mode", choices=("padded", "packed"), default="padded")
    p.add_argument("--custom-tuned-config", default="default")
    p.add_argument("--deepseek-diff-gate", type=float, default=1.0e-3)
    p.add_argument("--require-deepgemm-calc-diff", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-dir", default="artifacts/seq_sweep")
    p.add_argument(
        "--results-json",
        help="Optional explicit path for the aggregate machine-readable JSON. Defaults to <output-dir>/results.json.",
    )
    return p.parse_args()


def selected_cases(args: argparse.Namespace) -> list[str]:
    cases: list[str] = []
    if bool(args.include_sonic):
        cases.append("sonic_bf16")
    if bool(args.include_custom):
        cases.append("custom_cute_fp8_bf16_out")
    if bool(args.include_deepgemm):
        cases.append("deepgemm_fp8")
    return cases


def include_flags_for_case(args: argparse.Namespace, case: str | None) -> tuple[bool, bool, bool]:
    if case is None:
        return bool(args.include_sonic), bool(args.include_custom), bool(args.include_deepgemm)
    return case == "sonic_bf16", case == "custom_cute_fp8_bf16_out", case == "deepgemm_fp8"


def build_cmd(args: argparse.Namespace, *, tokens: int, run_dir: Path, case: str | None = None) -> list[str]:
    include_sonic, include_custom, include_deepgemm = include_flags_for_case(args, case)
    cmd = [
        sys.executable,
        str(BENCH),
        "--tokens",
        str(int(tokens)),
        "--experts",
        str(int(args.experts)),
        "--top-k",
        str(int(args.top_k)),
        "--hidden",
        str(int(args.hidden)),
        "--intermediate",
        str(int(args.intermediate)),
        "--route",
        str(args.route),
        "--seed",
        str(int(args.seed)),
        "--warmup-iters",
        str(int(args.warmup_iters)),
        "--active-iters",
        str(int(args.active_iters)),
        "--repeat",
        str(int(args.repeat)),
        "--custom-layout-mode",
        str(args.custom_layout_mode),
        "--custom-tuned-config",
        str(args.custom_tuned_config),
        "--deepseek-diff-gate",
        str(float(args.deepseek_diff_gate)),
        "--output-dir",
        str(run_dir),
        "--results-json",
        str(run_dir / "results.json"),
    ]
    cmd.append("--include-sonic" if include_sonic else "--no-include-sonic")
    cmd.append("--include-deepgemm" if include_deepgemm else "--no-include-deepgemm")
    cmd.append("--include-custom" if include_custom else "--no-include-custom")
    cmd.append("--require-deepgemm-calc-diff" if bool(args.require_deepgemm_calc_diff) else "--no-require-deepgemm-calc-diff")
    return cmd


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-int(lines) :])


def flatten_rows(summary: dict, *, run_dir: Path, log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for row in summary.get("results", []):
        flat = dict(row)
        flat["run_dir"] = str(run_dir)
        flat["results_json"] = str(run_dir / "results.json")
        flat["summary_json"] = str(run_dir / "summary.json")
        flat["log_path"] = str(log_path)
        rows.append(flat)
    return rows


def write_outputs(out_dir: Path, aggregate: dict, *, results_json: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate["output_paths"] = {
        "output_dir": str(out_dir),
        "results_json": str(results_json),
        "summary_json": str(out_dir / "summary.json"),
        "summary_md": str(out_dir / "summary.md"),
        "summary_csv": str(out_dir / "summary.csv"),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "tokens",
        "token_label",
        "case",
        "case_label",
        "status",
        "valid_rows",
        "padded_rows",
        "down_ms",
        "gate_up_ms",
        "total_ms",
        "speed_vs_sonic",
        "total_valid_tflops",
        "gpu_memory_peak_gb",
        "gpu_memory_peak_bytes",
        "down_deepseek_calc_diff",
        "gate_up_deepseek_calc_diff",
        "max_deepseek_calc_diff",
        "correctness_status",
        "run_dir",
        "results_json",
        "summary_json",
        "log_path",
        "error_log_path",
        "error_tail",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in aggregate["rows"]:
            writer.writerow({field: row.get(field, "") for field in fields})

    lines = [
        "# Wgrad Sequence-Length Sweep",
        "",
        f"created_at: `{aggregate['created_at']}`",
        f"deepseek_diff_gate: `{aggregate['deepseek_diff_gate']}`",
        f"results_json: `{aggregate['output_paths']['results_json']}`",
        "",
        "| tokens | case | status | total ms | speed vs Sonic | total valid TFLOP/s | max DeepSeek calc_diff |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["rows"]:
        speed = row.get("speed_vs_sonic")
        speed_s = "" if speed is None or speed == "" else f"{float(speed):.4f}x"
        calc_diff = row.get("max_deepseek_calc_diff")
        calc_diff_s = "" if calc_diff is None or calc_diff == "" else f"{float(calc_diff):.6e}"
        if row.get("status") == "ok":
            lines.append(
                f"| {row['tokens']} | {row['case_label']} | ok | {float(row['total_ms']):.6f} | "
                f"{speed_s} | {float(row['total_valid_tflops']):.3f} | {calc_diff_s} |"
            )
        else:
            lines.append(f"| {row.get('tokens', '')} | {row.get('case_label', '')} | failed |  |  |  | {calc_diff_s} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_aggregate_speed_vs_sonic(rows: list[dict]) -> None:
    sonic_by_tokens: dict[int, float] = {}
    for row in rows:
        if row.get("case") == "sonic_bf16" and row.get("status") == "ok" and row.get("total_ms") not in (None, ""):
            sonic_by_tokens[int(row["tokens"])] = float(row["total_ms"])
    for row in rows:
        sonic_total = sonic_by_tokens.get(int(row.get("tokens", -1)))
        if row.get("status") == "ok" and sonic_total is not None and row.get("total_ms") not in (None, ""):
            row["speed_vs_sonic"] = sonic_total / float(row["total_ms"])


def run_benchmark_process(args: argparse.Namespace, *, tokens: int, run_dir: Path, case: str | None) -> tuple[int, list[dict]]:
    label = token_label(tokens)
    log_path = run_dir / "run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(args, tokens=tokens, run_dir=run_dir, case=case)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)

    result_path = run_dir / "results.json"
    summary_path = run_dir / "summary.json"
    if result_path.exists() or summary_path.exists():
        loaded_path = result_path if result_path.exists() else summary_path
        summary = json.loads(loaded_path.read_text(encoding="utf-8"))
        run_rows = flatten_rows(summary, run_dir=run_dir, log_path=log_path)
        for row in run_rows:
            row["token_label"] = label
        # Baseline failures, such as DeepGEMM correctness failures, are
        # represented as failed rows in JSON. The sweep itself only fails when
        # a child process leaves no parseable result file.
        _ = proc
        return 0, run_rows

    case_label = case or "all"
    return 1, [
        {
            "tokens": int(tokens),
            "token_label": label,
            "case": case_label,
            "case_label": case_label,
            "status": "failed",
            "run_dir": str(run_dir),
            "results_json": str(result_path),
            "log_path": str(log_path),
            "error_tail": tail(log_path),
        }
    ]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    results_json = Path(args.results_json) if args.results_json else out_dir / "results.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures = 0

    cases = selected_cases(args)
    for tokens in parse_tokens_list(args.tokens_list):
        label = token_label(tokens)
        if bool(args.isolate_cases):
            if not cases:
                failures += 1
                rows.append(
                    {
                        "tokens": int(tokens),
                        "token_label": label,
                        "case": "all",
                        "case_label": "all",
                        "status": "failed",
                        "error_tail": "no benchmark cases selected",
                    }
                )
                continue
            for case in cases:
                run_dir = out_dir / f"tokens_{label}_{case}"
                failed, run_rows = run_benchmark_process(args, tokens=tokens, run_dir=run_dir, case=case)
                failures += failed
                rows.extend(run_rows)
        else:
            run_dir = out_dir / f"tokens_{label}"
            failed, run_rows = run_benchmark_process(args, tokens=tokens, run_dir=run_dir, case=None)
            failures += failed
            rows.extend(run_rows)

    apply_aggregate_speed_vs_sonic(rows)

    any_ok = any(row.get("status") == "ok" for row in rows)
    aggregate_status = "failed" if failures or not any_ok else "ok"
    aggregate = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "status": aggregate_status,
        "execution_mode": "isolated_per_case_process" if bool(args.isolate_cases) else "per_token_process",
        "selected_cases": cases,
        "deepgemm_metric": "deep_gemm.testing.calc_diff",
        "deepseek_diff_gate": float(args.deepseek_diff_gate),
        "require_deepgemm_calc_diff": bool(args.require_deepgemm_calc_diff),
        "rows": rows,
    }
    write_outputs(out_dir, aggregate, results_json=results_json)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0 if aggregate_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
