#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any


RESULTS_BEGIN = "<!-- SM90_WGRAD_RESULTS_BEGIN -->"
RESULTS_END = "<!-- SM90_WGRAD_RESULTS_END -->"

CASE_LABELS = {
    "sonic_bf16": "Sonic BF16",
    "custom_cute_fp8_bf16_out": "Custom CuTe FP8",
    "deepgemm_fp8": "DeepGEMM FP8",
}

TOKEN_ORDER = [4096, 8192, 16384, 32768]
PROJECT_DIR_NAME = "sm90-fp8-wgrad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render model-shape matrix results as publishable Markdown.")
    parser.add_argument(
        "--input",
        required=True,
        help="Matrix output directory or matrix_runs.tsv path.",
    )
    parser.add_argument(
        "--output",
        help="Markdown output path. Default: <matrix-dir>/results_for_readme.md.",
    )
    parser.add_argument(
        "--csv",
        help="Flat CSV output path. Default: <matrix-dir>/results_flat.csv.",
    )
    parser.add_argument(
        "--readme",
        help="Optional README path. The generated block replaces SM90_WGRAD_RESULTS markers.",
    )
    parser.add_argument(
        "--standalone-output",
        action="store_true",
        help="Write --output as a standalone Markdown document instead of a README marker block.",
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def token_label(tokens: int | None) -> str:
    if tokens is None:
        return ""
    if tokens % 1024 == 0:
        return f"{tokens // 1024}k"
    return str(tokens)


def parse_token_list(value: str) -> list[int]:
    out: list[int] = []
    for raw in str(value or "").split(","):
        item = raw.strip().lower().replace("_", "")
        if not item:
            continue
        if item.endswith("k"):
            out.append(int(float(item[:-1]) * 1024))
        else:
            out.append(int(float(item)))
    return out


def format_ms(value: Any) -> str:
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.3f}"


def format_speed(value: Any) -> str:
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.4f}x"


def format_tflops(value: Any) -> str:
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.1f}"


def format_diff(value: Any) -> str:
    parsed = as_float(value)
    return "" if parsed is None else f"{parsed:.2e}"


def short_error(row: dict[str, Any]) -> str:
    text = str(row.get("error_tail") or row.get("error") or "")
    if not text:
        return ""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped[:160]
    return ""


def load_matrix_rows(matrix_path: Path) -> tuple[Path, list[dict[str, str]]]:
    if matrix_path.is_dir():
        root = matrix_path
        manifest = root / "matrix_runs.tsv"
    else:
        manifest = matrix_path
        root = manifest.parent
    if not manifest.exists():
        raise FileNotFoundError(f"matrix manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return root, rows


def resolve_artifact_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path or "")
    if not raw_path:
        return path
    if path.exists():
        return path
    parts = path.parts
    if root.name in parts:
        suffix = Path(*parts[parts.index(root.name) + 1 :])
        relocated = root / suffix
        if relocated.exists():
            return relocated
    if not path.is_absolute():
        relocated = root / path
        if relocated.exists():
            return relocated
    return path


def artifact_display_root(root: Path) -> Path:
    if root.parent.name == "artifacts":
        return Path(root.parent.name, root.name)
    return Path(root.name)


def relative_artifact_path(root: Path, raw_path: Any) -> str:
    if raw_path in (None, ""):
        return ""
    text = str(raw_path)
    path = Path(text)
    parts = path.parts
    display_root = artifact_display_root(root)
    if root.name in parts:
        return Path(display_root, *parts[parts.index(root.name) + 1 :]).as_posix()
    try:
        return Path(display_root, path.relative_to(root)).as_posix()
    except ValueError:
        return text


def relative_embedded_paths(root: Path, raw_text: Any) -> str:
    if raw_text in (None, ""):
        return ""
    text = str(raw_text)
    replacements = (
        (root.name, artifact_display_root(root).as_posix()),
        (PROJECT_DIR_NAME, PROJECT_DIR_NAME),
    )
    for marker, replacement in replacements:
        text = re.sub(
            rf"(?<![\w.-])/(?:[^/\s\"'`|<>]+/)+{re.escape(marker)}/",
            f"{replacement}/",
            text,
        )
    return text


def make_row_paths_relative(root: Path, row: dict[str, Any]) -> None:
    for key in (
        "output_dir",
        "results_json",
        "summary_json",
        "log_path",
        "error_log_path",
        "source_results_json",
        "missing_results_json",
    ):
        if key in row:
            row[key] = relative_artifact_path(root, row.get(key))
    for key in ("error", "error_tail"):
        if key in row:
            row[key] = relative_embedded_paths(root, row.get(key))


def global_token_map(manifest_row: dict[str, str]) -> dict[int, int]:
    global_tokens = parse_token_list(manifest_row.get("global_tokens", ""))
    local_tokens = parse_token_list(manifest_row.get("local_tokens", ""))
    mapping: dict[int, int] = {}
    for global_token, local_token in zip(global_tokens, local_tokens):
        mapping[int(local_token)] = int(global_token)
    return mapping


def load_result_rows(root: Path, manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_manifest_row in manifest_rows:
        manifest_row = dict(raw_manifest_row)
        for key in ("output_dir", "results_json"):
            if key in manifest_row:
                manifest_row[key] = relative_artifact_path(root, manifest_row.get(key))
        token_map = global_token_map(manifest_row)
        local_by_global = {global_token: local_token for local_token, global_token in token_map.items()}
        results_json = resolve_artifact_path(root, raw_manifest_row.get("results_json") or "")

        if manifest_row.get("run_kind") == "deepgemm_skipped":
            for global_token in parse_token_list(manifest_row.get("global_tokens", "")):
                local_token = local_by_global.get(global_token, global_token)
                row = {
                    **manifest_row,
                    "case": "deepgemm_fp8",
                    "case_label": CASE_LABELS["deepgemm_fp8"],
                    "status": "skipped",
                    "tokens": local_token,
                    "global_tokens_value": global_token,
                    "token_label": token_label(global_token),
                    "error_tail": "Skipped by DeepGEMM local token cap.",
                }
                make_row_paths_relative(root, row)
                rows.append(row)
            continue

        if not results_json.exists():
            if manifest_row.get("run_kind") == "deepgemm":
                cases = ["deepgemm_fp8"]
            elif manifest_row.get("run_kind") == "sonic_custom":
                cases = ["sonic_bf16", "custom_cute_fp8_bf16_out"]
            else:
                cases = [str(manifest_row.get("run_kind") or "unknown")]
            for global_token in parse_token_list(manifest_row.get("global_tokens", "")) or [0]:
                local_token = local_by_global.get(global_token, global_token)
                for case in cases:
                    row = {
                        **manifest_row,
                        "case": case,
                        "case_label": CASE_LABELS.get(case, case),
                        "status": "missing",
                        "tokens": local_token,
                        "global_tokens_value": global_token or "",
                        "local_tokens_value": local_token or "",
                        "token_label": token_label(global_token) if global_token else "",
                        "error_tail": "Missing results file.",
                        "missing_results_json": str(results_json),
                    }
                    make_row_paths_relative(root, row)
                    rows.append(row)
            continue

        payload = json.loads(results_json.read_text(encoding="utf-8"))
        for result_row in payload.get("rows", []):
            local_token = as_int(result_row.get("tokens"))
            global_token = token_map.get(local_token or -1, local_token)
            case = str(result_row.get("case") or "")
            row = {
                **manifest_row,
                **result_row,
                "case": case,
                "case_label": CASE_LABELS.get(case, str(result_row.get("case_label") or case)),
                "global_tokens_value": global_token,
                "token_label": token_label(global_token),
                "local_tokens_value": local_token,
                "source_results_json": str(results_json),
            }
            make_row_paths_relative(root, row)
            rows.append(row)
    return rows


def add_custom_speed_from_sonic(rows: list[dict[str, Any]]) -> None:
    sonic: dict[tuple[str, str, str, int | None], float] = {}
    for row in rows:
        if row.get("case") != "sonic_bf16" or row.get("status") != "ok":
            continue
        key = (
            str(row.get("model_key")),
            str(row.get("route")),
            str(row.get("ep")),
            as_int(row.get("global_tokens_value")),
        )
        total_ms = as_float(row.get("total_ms"))
        if total_ms is not None:
            sonic[key] = total_ms

    for row in rows:
        if row.get("case") != "custom_cute_fp8_bf16_out" or row.get("status") != "ok":
            continue
        if as_float(row.get("speed_vs_sonic")) is not None:
            continue
        key = (
            str(row.get("model_key")),
            str(row.get("route")),
            str(row.get("ep")),
            as_int(row.get("global_tokens_value")),
        )
        total_ms = as_float(row.get("total_ms"))
        sonic_ms = sonic.get(key)
        if total_ms is not None and sonic_ms is not None:
            row["speed_vs_sonic"] = sonic_ms / total_ms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_key",
        "model_label",
        "route",
        "ep",
        "run_kind",
        "device",
        "global_experts",
        "local_experts",
        "top_k",
        "hidden",
        "intermediate",
        "global_tokens_value",
        "local_tokens_value",
        "case",
        "case_label",
        "status",
        "total_ms",
        "speed_vs_sonic",
        "total_valid_tflops",
        "max_deepseek_calc_diff",
        "gpu_memory_peak_gb",
        "output_dir",
        "source_results_json",
        "error_log_path",
        "error_tail",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def ok_custom_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("case") == "custom_cute_fp8_bf16_out" and row.get("status") == "ok"
    ]


def count_by_case(rows: list[dict[str, Any]]) -> list[tuple[str, int, int, int, int]]:
    cases = ["sonic_bf16", "custom_cute_fp8_bf16_out", "deepgemm_fp8"]
    out: list[tuple[str, int, int, int, int]] = []
    for case in cases:
        case_rows = [row for row in rows if row.get("case") == case]
        ok = sum(1 for row in case_rows if row.get("status") == "ok")
        failed = sum(1 for row in case_rows if row.get("status") not in ("ok", "skipped"))
        skipped = sum(1 for row in case_rows if row.get("status") == "skipped")
        out.append((CASE_LABELS[case], ok, failed, skipped, len(case_rows)))
    return out


def speed_matrix(rows: list[dict[str, Any]]) -> list[str]:
    custom_by_key: dict[tuple[str, str, str, int | None], dict[str, Any]] = {}
    groups: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row.get("case") != "custom_cute_fp8_bf16_out":
            continue
        group = (
            str(row.get("model_label") or row.get("model_key")),
            str(row.get("model_key")),
            str(row.get("route")),
            str(row.get("ep")),
        )
        groups.add(group)
        key = (group[1], group[2], group[3], as_int(row.get("global_tokens_value")))
        custom_by_key[key] = row

    lines = [
        "| model | route | EP | 4k | 8k | 16k | 32k |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_label, model_key, route, ep in sorted(groups, key=lambda item: (item[1], int(item[3]), item[2])):
        cells: list[str] = []
        for token in TOKEN_ORDER:
            row = custom_by_key.get((model_key, route, ep, token))
            if not row:
                cells.append("")
            elif row.get("status") != "ok":
                cells.append("failed")
            else:
                cells.append(format_speed(row.get("speed_vs_sonic")))
        lines.append(f"| {model_label} | {route} | {ep} | " + " | ".join(cells) + " |")
    return lines


def sonic_custom_comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    by_key: dict[tuple[str, str, str, int | None, str], dict[str, Any]] = {}
    custom_rows = [row for row in rows if row.get("case") == "custom_cute_fp8_bf16_out"]
    for row in rows:
        case = str(row.get("case") or "")
        if case not in ("sonic_bf16", "custom_cute_fp8_bf16_out"):
            continue
        key = (
            str(row.get("model_key")),
            str(row.get("route")),
            str(row.get("ep")),
            as_int(row.get("global_tokens_value")),
            case,
        )
        by_key[key] = row

    custom = sorted(
        custom_rows,
        key=lambda row: (
            str(row.get("model_key")),
            int(row.get("ep") or 0),
            str(row.get("route")),
            as_int(row.get("global_tokens_value")) or 0,
        ),
    )
    lines = [
        "| model | route | EP | tokens | Sonic ms | Custom ms | speed vs Sonic | custom TFLOP/s | max calc_diff | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in custom:
        base_key = (
            str(row.get("model_key")),
            str(row.get("route")),
            str(row.get("ep")),
            as_int(row.get("global_tokens_value")),
        )
        sonic = by_key.get((*base_key, "sonic_bf16"), {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model_label") or row.get("model_key")),
                    str(row.get("route")),
                    str(row.get("ep")),
                    token_label(as_int(row.get("global_tokens_value"))),
                    format_ms(sonic.get("total_ms")),
                    format_ms(row.get("total_ms")),
                    format_speed(row.get("speed_vs_sonic")),
                    format_tflops(row.get("total_valid_tflops")),
                    format_diff(row.get("max_deepseek_calc_diff")),
                    str(row.get("status")),
                ]
            )
            + " |"
        )
    return lines


def deepgemm_table(rows: list[dict[str, Any]]) -> list[str]:
    deepgemm = sorted(
        [row for row in rows if row.get("case") == "deepgemm_fp8"],
        key=lambda row: (
            str(row.get("model_key")),
            int(row.get("ep") or 0),
            str(row.get("route")),
            as_int(row.get("global_tokens_value")) or 0,
        ),
    )
    lines = [
        "| model | route | EP | tokens | status | total ms | max calc_diff | note |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in deepgemm:
        note = "" if row.get("status") == "ok" else short_error(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model_label") or row.get("model_key")),
                    str(row.get("route")),
                    str(row.get("ep")),
                    token_label(as_int(row.get("global_tokens_value"))),
                    str(row.get("status")),
                    format_ms(row.get("total_ms")),
                    format_diff(row.get("max_deepseek_calc_diff")),
                    note.replace("|", "/"),
                ]
            )
            + " |"
        )
    return lines


def render_markdown(root: Path, rows: list[dict[str, Any]]) -> str:
    add_custom_speed_from_sonic(rows)
    custom_ok = ok_custom_rows(rows)
    speeds = [as_float(row.get("speed_vs_sonic")) for row in custom_ok]
    speeds = [value for value in speeds if value is not None]
    diffs = [as_float(row.get("max_deepseek_calc_diff")) for row in custom_ok]
    diffs = [value for value in diffs if value is not None]
    best = max(custom_ok, key=lambda row: as_float(row.get("speed_vs_sonic")) or -1, default=None)
    worst = min(custom_ok, key=lambda row: as_float(row.get("speed_vs_sonic")) or 10**9, default=None)
    generated_at = datetime.now(timezone.utc).isoformat()
    source_hashes = sorted({str(row.get("source_hash")) for row in rows if row.get("source_hash")})

    lines: list[str] = [
        RESULTS_BEGIN,
        "",
        f"Generated at `{generated_at}` from matrix artifact `{artifact_display_root(root).as_posix()}`.",
        "",
        "### run status",
        "",
        "| implementation | ok | failed | skipped | total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, ok, failed, skipped, total in count_by_case(rows):
        lines.append(f"| {label} | {ok} | {failed} | {skipped} | {total} |")

    lines.extend(
        [
            "",
            "### headline",
            "",
        ]
    )
    if speeds:
        lines.append(f"- Custom CuTe FP8 average speed vs Sonic BF16 across valid rows: `{mean(speeds):.4f}x`.")
    if best:
        lines.append(
            "- Best custom row: "
            f"`{best.get('model_label')}` route `{best.get('route')}`, EP `{best.get('ep')}`, "
            f"{token_label(as_int(best.get('global_tokens_value')))} tokens, "
            f"`{format_speed(best.get('speed_vs_sonic'))}`."
        )
    if worst:
        lines.append(
            "- Slowest valid custom row: "
            f"`{worst.get('model_label')}` route `{worst.get('route')}`, EP `{worst.get('ep')}`, "
            f"{token_label(as_int(worst.get('global_tokens_value')))} tokens, "
            f"`{format_speed(worst.get('speed_vs_sonic'))}`."
        )
    if diffs:
        lines.append(f"- Worst custom DeepSeek `calc_diff`: `{max(diffs):.3e}`.")
    if source_hashes:
        joined_hashes = ", ".join(f"`{item}`" for item in source_hashes[:3])
        suffix = "" if len(source_hashes) <= 3 else f" and {len(source_hashes) - 3} more"
        lines.append(f"- Source hash: {joined_hashes}{suffix}.")
    lines.append("- DeepGEMM failed or skipped rows are reported below and are not used for custom speedup claims.")

    lines.extend(
        [
            "",
            "### custom speedup matrix",
            "",
            *speed_matrix(rows),
            "",
            "### Sonic BF16 vs custom CuTe FP8",
            "",
            *sonic_custom_comparison_table(rows),
            "",
            "### DeepGEMM public baseline",
            "",
            *deepgemm_table(rows),
            "",
            RESULTS_END,
            "",
        ]
    )
    return "\n".join(lines)


def update_readme(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    begin = text.find(RESULTS_BEGIN)
    end = text.find(RESULTS_END)
    if begin < 0 or end < 0 or end < begin:
        raise RuntimeError(f"README does not contain {RESULTS_BEGIN}/{RESULTS_END} markers")
    end += len(RESULTS_END)
    path.write_text(text[:begin] + block.rstrip() + text[end:] + ("\n" if text.endswith("\n") else ""), encoding="utf-8")


def standalone_markdown(root: Path, block: str) -> str:
    artifact = artifact_display_root(root).as_posix()
    body = block.replace(f"{RESULTS_BEGIN}\n\n", "").replace(f"\n{RESULTS_END}\n", "\n")
    header = "\n".join(
        [
            "# benchmark results",
            "",
            f"Source artifact: `{artifact}`.",
            "",
            f"Run manifests: `{artifact}/matrix_runs.tsv` and `{artifact}/matrix_jobs.tsv`.",
            "",
            "Correctness gate: max DeepSeek `calc_diff <= 1e-3`. Failed DeepGEMM rows are baseline diagnostics and are not used for custom speedup claims.",
            "",
        ]
    )
    return header + "\n" + body.lstrip()


def main() -> int:
    args = parse_args()
    root, manifest_rows = load_matrix_rows(Path(args.input))
    rows = load_result_rows(root, manifest_rows)
    add_custom_speed_from_sonic(rows)

    output = Path(args.output) if args.output else root / "results_for_readme.md"
    csv_output = Path(args.csv) if args.csv else root / "results_flat.csv"
    markdown = render_markdown(root, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    output_text = standalone_markdown(root, markdown) if args.standalone_output else markdown
    output.write_text(output_text, encoding="utf-8")
    write_csv(csv_output, rows)
    if args.readme:
        update_readme(Path(args.readme), markdown)

    print(f"wrote {output}")
    print(f"wrote {csv_output}")
    if args.readme:
        print(f"updated {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
