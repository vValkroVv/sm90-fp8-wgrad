#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON:-python}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARTIFACT_ROOT="${STORAGE_ROOT:-$REPO_ROOT}/artifacts"
OUTPUT_ROOT="$ARTIFACT_ROOT/sm90_fp8_wgrad_model_shape_matrix_$TIMESTAMP"

MODEL_LIST="all"
EP_LIST="1"
ROUTE_SELECTION="both"
TOKENS_LIST="4096,8192,16384,32768"
WARMUP_ITERS=5
ACTIVE_ITERS=100
REPEAT=1
SEED=1234
CUSTOM_LAYOUT_MODE="padded"
CUSTOM_TUNED_CONFIG="default"
DEEPSEEK_DIFF_GATE="0.001"
DEEPGEMM_LOCAL_TOKEN_CAP=8192
RUN_DEEPGEMM=1
DRY_RUN=0
DEVICES=""
PARALLEL=0
JOB_GRANULARITY="auto"
EP_TOKEN_MODE="real"
CASE_ISOLATION="isolated"
CLEAN_CHILD_OUTPUTS=1

MODEL_SPECS=(
  "qwen3_30b|Qwen3-30B-A3B|128|8|2048|768"
  "deepseek_v3|DeepSeek-V3|256|8|7168|2048"
  "minimax_m3|MiniMax-M3|128|4|6144|3072"
  "qwen3_5_397b_a17b|Qwen3.5-397B-A17B|512|10|4096|1024"
)

usage() {
  cat <<'EOF'
Run SM90 FP8 Wgrad model-shape benchmark matrix.

By default this runs all model shapes, EP=1, and both balanced/skewed routes.
Use --all-eps for the full requested EP matrix: 1,8,16.

Options:
  --output-dir DIR              Output root. Default: $STORAGE_ROOT/artifacts/sm90_fp8_wgrad_model_shape_matrix_<ts>
  --models LIST                 Comma list: all,qwen3_30b,deepseek_v3,minimax_m3,qwen3_5_397b_a17b
  --ep LIST                     Comma list of EP sizes. Default: 1
  --all-eps                     Shortcut for --ep 1,8,16
  --route ROUTE                 balanced, skewed, or both. Default: both
  --tokens-list LIST            Global token counts. Default: 4096,8192,16384,32768
  --warmup-iters N              Default: 5
  --active-iters N              Default: 100
  --repeat N                    Default: 1
  --seed N                      Default: 1234
  --custom-layout-mode MODE     padded or packed. Default: padded
  --custom-tuned-config NAME    Default: default
  --deepseek-diff-gate FLOAT    Default: 0.001
  --deepgemm-local-token-cap N  Run DeepGEMM only for local_tokens <= N. Default: 8192
  --no-deepgemm-token-cap       Run DeepGEMM for every local token count.
  --no-deepgemm                 Skip DeepGEMM entirely.
  --devices LIST                Comma list of GPU ids, e.g. 0,1,2,3. Implies --parallel.
  --parallel                    Run queued benchmark jobs across devices.
  --sequential                  Force sequential execution.
  --job-granularity MODE        auto, sweep, or token. Default: auto.
  --case-isolation MODE         isolated or grouped. Default: isolated.
  --keep-child-outputs          Keep per-case JSON/Markdown/CSV outputs.
  --ep-token-mode MODE          real or stress. Default: real.
  --dry-run                     Print commands without running them.
  -h, --help                    Show this help.

EP proxy:
  local_experts = global_experts / EP
  real mode:   local_tokens = global_tokens
  stress mode: local_tokens = global_tokens * EP

Outputs:
  <output-dir>/matrix_runs.tsv
  <output-dir>/<model>/ep<EP>/<route>/sonic_custom/
  <output-dir>/<model>/ep<EP>/<route>/deepgemm/

Parallel execution:
  --devices 0,1,2,3 runs one worker per listed GPU and sets each child process
  to CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>. By default each
  model/EP/route sweep is one queued job, so outputs stay compact.
EOF
}

fail() {
  echo "error: $*" >&2
  return 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_ROOT="${2:-}"
      shift 2
      ;;
    --models)
      MODEL_LIST="${2:-}"
      shift 2
      ;;
    --ep)
      EP_LIST="${2:-}"
      shift 2
      ;;
    --all-eps)
      EP_LIST="1,8,16"
      shift
      ;;
    --route)
      ROUTE_SELECTION="${2:-}"
      shift 2
      ;;
    --tokens-list)
      TOKENS_LIST="${2:-}"
      shift 2
      ;;
    --warmup-iters)
      WARMUP_ITERS="${2:-}"
      shift 2
      ;;
    --active-iters)
      ACTIVE_ITERS="${2:-}"
      shift 2
      ;;
    --repeat)
      REPEAT="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --custom-layout-mode)
      CUSTOM_LAYOUT_MODE="${2:-}"
      shift 2
      ;;
    --custom-tuned-config)
      CUSTOM_TUNED_CONFIG="${2:-}"
      shift 2
      ;;
    --deepseek-diff-gate)
      DEEPSEEK_DIFF_GATE="${2:-}"
      shift 2
      ;;
    --deepgemm-local-token-cap)
      DEEPGEMM_LOCAL_TOKEN_CAP="${2:-}"
      shift 2
      ;;
    --no-deepgemm-token-cap)
      DEEPGEMM_LOCAL_TOKEN_CAP=0
      shift
      ;;
    --no-deepgemm)
      RUN_DEEPGEMM=0
      shift
      ;;
    --devices)
      DEVICES="${2:-}"
      PARALLEL=1
      shift 2
      ;;
    --parallel)
      PARALLEL=1
      shift
      ;;
    --sequential)
      PARALLEL=0
      shift
      ;;
    --job-granularity)
      JOB_GRANULARITY="${2:-}"
      shift 2
      ;;
    --case-isolation)
      CASE_ISOLATION="${2:-}"
      shift 2
      ;;
    --keep-child-outputs)
      CLEAN_CHILD_OUTPUTS=0
      shift
      ;;
    --ep-token-mode)
      EP_TOKEN_MODE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      fail "unknown argument $1"
      exit 2
      ;;
  esac
done

case "$ROUTE_SELECTION" in
  balanced) ROUTES=("balanced") ;;
  skewed) ROUTES=("skewed") ;;
  both) ROUTES=("balanced" "skewed") ;;
  *)
    usage >&2
    fail "--route must be balanced, skewed, or both"
    exit 2
    ;;
esac

case "$JOB_GRANULARITY" in
  auto|sweep|token) ;;
  *)
    usage >&2
    fail "--job-granularity must be auto, sweep, or token"
    exit 2
    ;;
esac

case "$CASE_ISOLATION" in
  grouped|isolated) ;;
  *)
    usage >&2
    fail "--case-isolation must be grouped or isolated"
    exit 2
    ;;
esac

case "$EP_TOKEN_MODE" in
  real|stress) ;;
  *)
    usage >&2
    fail "--ep-token-mode must be real or stress"
    exit 2
    ;;
esac

if [[ "$JOB_GRANULARITY" == "auto" ]]; then
  JOB_GRANULARITY="sweep"
fi

token_to_int() {
  local raw="$1"
  local lower
  lower="$(printf "%s" "$raw" | tr '[:upper:]' '[:lower:]')"
  lower="${lower//_/}"
  if [[ "$lower" == *k ]]; then
    local base="${lower%k}"
    echo $((base * 1024))
  else
    echo "$lower"
  fi
}

token_label() {
  local tokens="$1"
  if (( tokens % 1024 == 0 )); then
    echo "$((tokens / 1024))k"
  else
    echo "$tokens"
  fi
}

scale_one_token_for_ep() {
  local global_token="$1"
  local ep="$2"
  if [[ "$EP_TOKEN_MODE" == "stress" ]]; then
    echo $((global_token * ep))
  else
    echo "$global_token"
  fi
}

scale_tokens_list() {
  local tokens_csv="$1"
  local ep="$2"
  local out=""
  local part global local
  local parts
  if [[ -z "$tokens_csv" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a parts <<< "$tokens_csv"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    global="$(token_to_int "$part")"
    local="$(scale_one_token_for_ep "$global" "$ep")"
    if [[ -n "$out" ]]; then
      out+=","
    fi
    out+="$local"
  done
  echo "$out"
}

filter_tokens_le_cap() {
  local tokens_csv="$1"
  local cap="$2"
  local out=""
  local part tokens
  local parts
  if [[ -z "$tokens_csv" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a parts <<< "$tokens_csv"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    tokens="$(token_to_int "$part")"
    if (( cap == 0 || tokens <= cap )); then
      if [[ -n "$out" ]]; then
        out+=","
      fi
      out+="$tokens"
    fi
  done
  echo "$out"
}

filter_global_tokens_by_local_cap() {
  local tokens_csv="$1"
  local ep="$2"
  local cap="$3"
  local out=""
  local part global local
  local parts
  if [[ -z "$tokens_csv" ]]; then
    echo ""
    return 0
  fi
  IFS=',' read -r -a parts <<< "$tokens_csv"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    global="$(token_to_int "$part")"
    local="$(scale_one_token_for_ep "$global" "$ep")"
    if (( cap == 0 || local <= cap )); then
      if [[ -n "$out" ]]; then
        out+=","
      fi
      out+="$global"
    fi
  done
  echo "$out"
}

model_selected() {
  local key="$1"
  if [[ "$MODEL_LIST" == "all" ]]; then
    return 0
  fi
  local item
  IFS=',' read -r -a selected <<< "$MODEL_LIST"
  for item in "${selected[@]}"; do
    if [[ "$item" == "$key" ]]; then
      return 0
    fi
  done
  return 1
}

shell_join() {
  local out=""
  local arg
  for arg in "$@"; do
    if [[ -n "$out" ]]; then
      out+=" "
    fi
    printf -v quoted "%q" "$arg"
    out+="$quoted"
  done
  echo "$out"
}

acquire_lock() {
  local lock_dir="$1"
  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 0.1
  done
}

release_lock() {
  local lock_dir="$1"
  rmdir "$lock_dir" 2>/dev/null || true
}

append_manifest_line() {
  local line="$1"
  acquire_lock "$MATRIX_LOCK_DIR"
  printf "%s\n" "$line" >> "$MATRIX_TSV"
  release_lock "$MATRIX_LOCK_DIR"
}

append_manifest() {
  local model_key="$1"
  local model_label="$2"
  local route="$3"
  local ep="$4"
  local run_kind="$5"
  local device="$6"
  local global_experts="$7"
  local local_experts="$8"
  local top_k="$9"
  local hidden="${10}"
  local intermediate="${11}"
  local global_tokens="${12}"
  local local_tokens="${13}"
  local run_dir="${14}"
  local results_json="${15}"
  local rc="${16}"
  local command_text="${17}"
  local line
  line="$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" \
    "$model_key" "$model_label" "$route" "$ep" "$run_kind" "$device" \
    "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
    "$global_tokens" "$local_tokens" "$run_dir" "$results_json" "$rc" "$command_text")"
  append_manifest_line "$line"
}

cleanup_child_outputs() {
  local run_dir="$1"
  if [[ "$CLEAN_CHILD_OUTPUTS" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "$run_dir" ]]; then
    return 0
  fi

  # Keep the aggregate files in run_dir and keep logs under child directories.
  # Remove per-case machine/markdown summaries so users see one result table per
  # model/EP/route job instead of hundreds of leaf summaries.
  find "$run_dir" -mindepth 2 -type f \
    \( -name 'summary.md' -o -name 'summary.json' -o -name 'results.json' -o -name 'results.csv' -o -name 'env.json' \) \
    -delete
}

run_sweep() {
  local run_kind="$1"
  local model_key="$2"
  local model_label="$3"
  local route="$4"
  local ep="$5"
  local global_experts="$6"
  local local_experts="$7"
  local top_k="$8"
  local hidden="$9"
  local intermediate="${10}"
  local global_tokens="${11}"
  local local_tokens="${12}"
  local include_sonic="${13}"
  local include_custom="${14}"
  local include_deepgemm="${15}"
  local run_dir="${16}"
  local device="${17:-}"

  mkdir -p "$run_dir"
  local results_json="$run_dir/results.json"
  local cmd=(
    "$PYTHON_BIN" "$REPO_ROOT/scripts/sweep_seq_lens.py"
    --tokens-list "$local_tokens"
    --experts "$local_experts"
    --top-k "$top_k"
    --hidden "$hidden"
    --intermediate "$intermediate"
    --route "$route"
    --seed "$SEED"
    --warmup-iters "$WARMUP_ITERS"
    --active-iters "$ACTIVE_ITERS"
    --repeat "$REPEAT"
    --custom-layout-mode "$CUSTOM_LAYOUT_MODE"
    --custom-tuned-config "$CUSTOM_TUNED_CONFIG"
    --deepseek-diff-gate "$DEEPSEEK_DIFF_GATE"
    --output-dir "$run_dir"
    --results-json "$results_json"
  )
  if [[ "$include_sonic" == "1" ]]; then cmd+=(--include-sonic); else cmd+=(--no-include-sonic); fi
  if [[ "$include_custom" == "1" ]]; then cmd+=(--include-custom); else cmd+=(--no-include-custom); fi
  if [[ "$include_deepgemm" == "1" ]]; then cmd+=(--include-deepgemm); else cmd+=(--no-include-deepgemm); fi
  if [[ "$CASE_ISOLATION" == "isolated" ]]; then cmd+=(--isolate-cases); else cmd+=(--no-isolate-cases); fi

  local command_text
  if [[ -n "$device" ]]; then
    command_text="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device $(shell_join "${cmd[@]}")"
  else
    command_text="$(shell_join "${cmd[@]}")"
  fi
  printf "%s\n" "$command_text" > "$run_dir/command.txt"
  echo
  echo "================================================================================"
  echo "model=$model_key route=$route ep=$ep kind=$run_kind device=${device:-current}"
  echo "global_experts=$global_experts local_experts=$local_experts global_tokens=$global_tokens local_tokens=$local_tokens"
  echo "output=$run_dir"
  echo "command: $command_text"

  local rc=0
  if [[ "$DRY_RUN" == "1" ]]; then
    rc=0
  elif [[ -n "$device" ]]; then
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$device" "${cmd[@]}"
    rc=$?
  else
    "${cmd[@]}"
    rc=$?
  fi

  cleanup_child_outputs "$run_dir"

  append_manifest \
    "$model_key" "$model_label" "$route" "$ep" "$run_kind" "$device" \
    "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
    "$global_tokens" "$local_tokens" "$run_dir" "$results_json" "$rc" "$command_text"

  return "$rc"
}

enqueue_sweep() {
  local line
  line="$(printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" "$@")"
  printf "%s\n" "$line" >> "$JOBS_TSV"
}

run_job_line() {
  local line="$1"
  local device="${2:-}"
  local run_kind model_key model_label route ep global_experts local_experts top_k hidden intermediate
  local global_tokens local_tokens include_sonic include_custom include_deepgemm run_dir
  IFS=$'\t' read -r \
    run_kind model_key model_label route ep global_experts local_experts top_k hidden intermediate \
    global_tokens local_tokens include_sonic include_custom include_deepgemm run_dir <<< "$line"
  run_sweep \
    "$run_kind" "$model_key" "$model_label" "$route" "$ep" \
    "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
    "$global_tokens" "$local_tokens" \
    "$include_sonic" "$include_custom" "$include_deepgemm" "$run_dir" "$device"
}

next_job_line() {
  local idx next total
  total="$(wc -l < "$JOBS_TSV" | tr -d ' ')"
  acquire_lock "$QUEUE_LOCK_DIR"
  idx="$(cat "$QUEUE_COUNTER")"
  if (( idx >= total )); then
    release_lock "$QUEUE_LOCK_DIR"
    return 1
  fi
  next=$((idx + 1))
  printf "%s\n" "$next" > "$QUEUE_COUNTER"
  release_lock "$QUEUE_LOCK_DIR"
  sed -n "${next}p" "$JOBS_TSV"
  return 0
}

worker_loop() {
  local device="$1"
  local line run_dir rc
  while line="$(next_job_line)"; do
    run_dir="$(printf "%s\n" "$line" | awk -F '\t' '{print $16}')"
    mkdir -p "$run_dir"
    echo "[worker gpu=$device] start $run_dir"
    run_job_line "$line" "$device" > "$run_dir/runner.log" 2>&1
    rc=$?
    echo "[worker gpu=$device] done rc=$rc $run_dir"
  done
}

export SM90_WGRAD_CALC_DIFF_CHUNK_ELEMENTS="${SM90_WGRAD_CALC_DIFF_CHUNK_ELEMENTS:-1048576}"
export SM90_WGRAD_FINITE_CHECK_CHUNK_ELEMENTS="${SM90_WGRAD_FINITE_CHECK_CHUNK_ELEMENTS:-1048576}"

mkdir -p "$OUTPUT_ROOT"
MATRIX_TSV="$OUTPUT_ROOT/matrix_runs.tsv"
JOBS_TSV="$OUTPUT_ROOT/matrix_jobs.tsv"
MATRIX_LOCK_DIR="$OUTPUT_ROOT/.matrix_lock"
QUEUE_LOCK_DIR="$OUTPUT_ROOT/.queue_lock"
QUEUE_COUNTER="$OUTPUT_ROOT/.queue_counter"
rm -rf "$MATRIX_LOCK_DIR" "$QUEUE_LOCK_DIR"
printf "model_key\tmodel_label\troute\tep\trun_kind\tdevice\tglobal_experts\tlocal_experts\ttop_k\thidden\tintermediate\tglobal_tokens\tlocal_tokens\toutput_dir\tresults_json\trc\tcommand\n" > "$MATRIX_TSV"
: > "$JOBS_TSV"
printf "0\n" > "$QUEUE_COUNTER"

cat > "$OUTPUT_ROOT/README.txt" <<EOF
SM90 FP8 Wgrad model-shape matrix
created_at_utc: $TIMESTAMP
models: $MODEL_LIST
ep_list: $EP_LIST
route: $ROUTE_SELECTION
global_tokens_list: $TOKENS_LIST
warmup_iters: $WARMUP_ITERS
active_iters: $ACTIVE_ITERS
repeat: $REPEAT
deepgemm_local_token_cap: $DEEPGEMM_LOCAL_TOKEN_CAP
parallel: $PARALLEL
devices: ${DEVICES:-<current>}
job_granularity: $JOB_GRANULARITY
case_isolation: $CASE_ISOLATION
clean_child_outputs: $CLEAN_CHILD_OUTPUTS
ep_token_mode: $EP_TOKEN_MODE

EP proxy:
local_experts = global_experts / EP
real mode local_tokens = global_tokens
stress mode local_tokens = global_tokens * EP

DeepGEMM is run separately from Sonic/custom. By default it is skipped for
local token counts above the local cap because the public DeepGEMM Wgrad API
has known failures for larger sequences in this benchmark.
EOF

failure_count=0
skip_count=0

enqueue_token_jobs() {
  local run_kind="$1"
  local model_key="$2"
  local model_label="$3"
  local route="$4"
  local ep="$5"
  local global_experts="$6"
  local local_experts="$7"
  local top_k="$8"
  local hidden="$9"
  local intermediate="${10}"
  local global_tokens_csv="${11}"
  local local_tokens_csv="${12}"
  local include_sonic="${13}"
  local include_custom="${14}"
  local include_deepgemm="${15}"
  local run_dir_base="${16}"
  local part global_token local_token global_label local_label
  IFS=',' read -r -a global_parts <<< "$global_tokens_csv"
  for part in "${global_parts[@]}"; do
    [[ -z "$part" ]] && continue
    global_token="$(token_to_int "$part")"
    local_token="$(scale_one_token_for_ep "$global_token" "$ep")"
    global_label="$(token_label "$global_token")"
    local_label="$(token_label "$local_token")"
    enqueue_sweep \
      "$run_kind" "$model_key" "$model_label" "$route" "$ep" \
      "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
      "$global_token" "$local_token" \
      "$include_sonic" "$include_custom" "$include_deepgemm" \
      "$run_dir_base/global_${global_label}_local_${local_label}"
  done
  _="$local_tokens_csv"
}

enqueue_jobs_for_sweep() {
  local run_kind="$1"
  local model_key="$2"
  local model_label="$3"
  local route="$4"
  local ep="$5"
  local global_experts="$6"
  local local_experts="$7"
  local top_k="$8"
  local hidden="$9"
  local intermediate="${10}"
  local global_tokens_csv="${11}"
  local local_tokens_csv="${12}"
  local include_sonic="${13}"
  local include_custom="${14}"
  local include_deepgemm="${15}"
  local run_dir_base="${16}"
  if [[ "$JOB_GRANULARITY" == "token" ]]; then
    enqueue_token_jobs \
      "$run_kind" "$model_key" "$model_label" "$route" "$ep" \
      "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
      "$global_tokens_csv" "$local_tokens_csv" \
      "$include_sonic" "$include_custom" "$include_deepgemm" "$run_dir_base"
  else
    enqueue_sweep \
      "$run_kind" "$model_key" "$model_label" "$route" "$ep" \
      "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
      "$global_tokens_csv" "$local_tokens_csv" \
      "$include_sonic" "$include_custom" "$include_deepgemm" "$run_dir_base"
  fi
}

IFS=',' read -r -a eps <<< "$EP_LIST"
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_key model_label global_experts top_k hidden intermediate <<< "$spec"
  if ! model_selected "$model_key"; then
    continue
  fi

  for ep_raw in "${eps[@]}"; do
    ep="${ep_raw// /}"
    [[ -z "$ep" ]] && continue
    if (( ep <= 0 )); then
      echo "Skipping invalid EP=$ep for $model_key" >&2
      failure_count=$((failure_count + 1))
      continue
    fi
    if (( global_experts % ep != 0 )); then
      echo "Skipping $model_key EP=$ep: global experts $global_experts is not divisible by EP" >&2
      failure_count=$((failure_count + 1))
      continue
    fi

    local_experts=$((global_experts / ep))
    local_tokens="$(scale_tokens_list "$TOKENS_LIST" "$ep")"

    for route in "${ROUTES[@]}"; do
      run_base="$OUTPUT_ROOT/$model_key/ep${ep}/$route"
      sonic_custom_dir="$run_base/sonic_custom"
      enqueue_jobs_for_sweep \
        "sonic_custom" "$model_key" "$model_label" "$route" "$ep" \
        "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
        "$TOKENS_LIST" "$local_tokens" \
        1 1 0 "$sonic_custom_dir"

      if [[ "$RUN_DEEPGEMM" == "1" ]]; then
        dg_global_tokens="$(filter_global_tokens_by_local_cap "$TOKENS_LIST" "$ep" "$DEEPGEMM_LOCAL_TOKEN_CAP")"
        dg_tokens="$(scale_tokens_list "$dg_global_tokens" "$ep")"
        if [[ -z "$dg_tokens" ]]; then
          skip_count=$((skip_count + 1))
          dg_dir="$run_base/deepgemm"
          mkdir -p "$dg_dir"
          echo "DeepGEMM skipped: no local token count <= cap $DEEPGEMM_LOCAL_TOKEN_CAP" > "$dg_dir/SKIPPED.txt"
          append_manifest \
            "$model_key" "$model_label" "$route" "$ep" "deepgemm_skipped" "" \
            "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
            "$TOKENS_LIST" "$local_tokens" "$dg_dir" "" "skip" "DeepGEMM skipped by local token cap"
        else
          dg_dir="$run_base/deepgemm"
          enqueue_jobs_for_sweep \
            "deepgemm" "$model_key" "$model_label" "$route" "$ep" \
            "$global_experts" "$local_experts" "$top_k" "$hidden" "$intermediate" \
            "$dg_global_tokens" "$dg_tokens" \
            0 0 1 "$dg_dir"
        fi
      fi
    done
  done
done

job_count="$(wc -l < "$JOBS_TSV" | tr -d ' ')"
echo "Queued jobs: $job_count"

if (( job_count > 0 )); then
  if [[ "$PARALLEL" == "1" ]]; then
    if [[ -z "$DEVICES" ]]; then
      if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        DEVICES="$CUDA_VISIBLE_DEVICES"
      else
        DEVICES="0"
      fi
    fi
    IFS=',' read -r -a device_array <<< "$DEVICES"
    echo "Parallel devices: $DEVICES"
    echo "Job granularity: $JOB_GRANULARITY"
    for raw_device in "${device_array[@]}"; do
      device="${raw_device// /}"
      [[ -z "$device" ]] && continue
      worker_loop "$device" &
    done
    wait
  else
    while IFS= read -r line; do
      run_job_line "$line" ""
    done < "$JOBS_TSV"
  fi
fi

failure_count="$(awk -F '\t' 'NR > 1 && $16 != "0" && $16 != "skip" {count++} END {print count + 0}' "$MATRIX_TSV")"
skip_count="$(awk -F '\t' 'NR > 1 && $16 == "skip" {count++} END {print count + 0}' "$MATRIX_TSV")"

echo
echo "Matrix manifest: $MATRIX_TSV"
echo "Output root: $OUTPUT_ROOT"
echo "Failures: $failure_count"
echo "DeepGEMM skipped groups: $skip_count"

RENDER_SCRIPT="$REPO_ROOT/scripts/render_matrix_results.py"
if [[ -f "$RENDER_SCRIPT" ]]; then
  if "$PYTHON_BIN" "$RENDER_SCRIPT" \
    --input "$OUTPUT_ROOT" \
    --output "$OUTPUT_ROOT/results_for_readme.md" \
    --csv "$OUTPUT_ROOT/results_flat.csv"; then
    echo "README results markdown: $OUTPUT_ROOT/results_for_readme.md"
    echo "Flat results CSV: $OUTPUT_ROOT/results_flat.csv"
  else
    echo "warning: failed to render README results markdown" >&2
  fi
fi

if (( failure_count > 0 )); then
  exit 1
fi
exit 0
