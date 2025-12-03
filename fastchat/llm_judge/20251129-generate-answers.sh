#!/usr/bin/env bash
set -euo pipefail

# Models to generate answers for (edit this list as needed).
MODELS=(
  # "shisa-ai/168-llama3.3-70b-v2.1-sft"
  # "shisa-ai/shisa-v2-llama3.3-70b"
  "shisa-ai/170-qwen3-8b-v2.1-dpo-1.2e7"
  "shisa-ai/166-qwen3-30b-a3b-v2.1-dpo-1.2e7"
  "shisa-ai/144-lfm2-1.2b-v2.1-dpo-1e-6"
  "shisa-ai/171-llama3.2-3b-v2.1-153sft-dpo-2e7"
  "shisa-ai/shisa-v2.1-unphi4-14b-152-155-149-nuslerp"
  "LiquidAI/LFM2-1.2B"
  "Qwen/Qwen3-8B"
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

DEFAULT_LOCAL_BASE="http://localhost:8000/v1"
VLLM_ENV_NAME="vllm"
VLLM_MAX_MODEL_LEN=16384
VLLM_PORT="${VLLM_PORT:-$(echo "${DEFAULT_LOCAL_BASE}" | sed -n 's@.*://[^:/]*:\\([0-9]\\+\\).*@\\1@p')}"
VLLM_PORT="${VLLM_PORT:-8000}"
PARALLEL=80
EXPECTED_QUESTIONS=80
GEN_ENV_NAME="${GEN_ENV_NAME:-jamt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTCHAT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

_VLLM_PID=""
_VLLM_MODEL=""
_VLLM_LOG_FILE=""
_VLLM_STARTED=""

sanitize_model() {
  # Replace slashes with double underscores to match existing file naming.
  echo "$1" | sed 's:/:__:g'
}

wait_for_vllm() {
  local url="${1:-${DEFAULT_LOCAL_BASE}/models}"
  local pid="${2:-}"
  local tries=1800
  echo "Waiting for vLLM at ${url} ..."
  for ((i=1; i<=tries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "vLLM is up."
      return 0
    fi
    if [[ -n "${pid}" ]]; then
      if ! kill -0 "${pid}" >/dev/null 2>&1; then
        echo "vLLM process ${pid} exited while waiting for readiness${_VLLM_LOG_FILE:+ (log: ${_VLLM_LOG_FILE})}" >&2
        return 1
      fi
    fi
    sleep 1
  done
  if [[ -n "${pid}" ]]; then
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "Timed out waiting for ${url}; terminating vLLM pid ${pid}" >&2
      kill "${pid}" >/dev/null 2>&1 || true
      wait "${pid}" >/dev/null 2>&1 || true
    fi
  fi
  echo "Timed out waiting for ${url}" >&2
  return 1
}

vllm_has_model() {
  local target_model="$1"
  local url="${DEFAULT_LOCAL_BASE}/models"
  local resp
  if ! resp=$(curl -fsS "${url}" 2>/dev/null); then
    return 2
  fi
  python3 -c 'import json, sys
target = sys.argv[1]
raw = sys.stdin.read().strip()
if not raw:
    sys.exit(2)
try:
    data = json.loads(raw)
except json.JSONDecodeError:
    sys.exit(2)
for m in data.get("data", []):
    if m.get("id") == target:
        sys.exit(0)
sys.exit(1)
' "$target_model" <<<"${resp}"
}

vllm_port_in_use() {
  lsof -iTCP:"${VLLM_PORT}" -sTCP:LISTEN >/dev/null 2>&1
}

cleanup_vllm() {
  if [[ -n "${_VLLM_STARTED:-}" && -n "${_VLLM_PID:-}" ]]; then
    if kill -0 "${_VLLM_PID}" >/dev/null 2>&1; then
      echo "Stopping vLLM (pid ${_VLLM_PID})"
      kill "${_VLLM_PID}" >/dev/null 2>&1 || true
      wait "${_VLLM_PID}" >/dev/null 2>&1 || true
    fi
    _VLLM_PID=""
    _VLLM_MODEL=""
    _VLLM_LOG_FILE=""
    _VLLM_STARTED=""
  fi
}

trap cleanup_vllm EXIT

start_vllm_for_model() {
  local model="$1"

  # Already running with the desired model.
  if [[ -n "${_VLLM_STARTED:-}" && "${_VLLM_MODEL:-}" == "${model}" ]]; then
    if kill -0 "${_VLLM_PID}" >/dev/null 2>&1; then
      return 0
    fi
  fi

  # If an external server is up but missing the model, do not stomp on it.
  if vllm_has_model "${model}"; then
    return 0
  else
    local rc=$?
    if [[ "${rc}" -eq 1 ]]; then
      echo "  ⚠ vLLM at ${DEFAULT_LOCAL_BASE} is up but not serving '${model}'. Please load it there or change DEFAULT_LOCAL_BASE." >&2
      return 1
    fi
  fi

  if vllm_port_in_use; then
    echo "  ⚠ Port ${VLLM_PORT} is in use and ${DEFAULT_LOCAL_BASE} is not responding as vLLM. Free the port or adjust DEFAULT_LOCAL_BASE." >&2
    return 1
  fi

  if [[ -n "${_VLLM_STARTED:-}" && -n "${_VLLM_PID:-}" ]]; then
    cleanup_vllm
  fi

  _VLLM_LOG_FILE="$(mktemp /tmp/vllm-serve-XXXX.log)"
  echo "  Launching vLLM serve for ${model} (log: ${_VLLM_LOG_FILE}) ..."
  mamba run -n "${VLLM_ENV_NAME}" vllm serve "${model}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" >"${_VLLM_LOG_FILE}" 2>&1 &
  _VLLM_PID=$!
  _VLLM_MODEL="${model}"
  _VLLM_STARTED=1

  if ! wait_for_vllm "${DEFAULT_LOCAL_BASE}/models" "${_VLLM_PID}"; then
    echo "  ✗ vLLM failed to become ready for '${model}'. See ${_VLLM_LOG_FILE}" >&2
    cleanup_vllm
    return 1
  fi

  if ! vllm_has_model "${model}"; then
    echo "  ✗ vLLM came up but model '${model}' is not loaded. Check ${_VLLM_LOG_FILE}" >&2
    cleanup_vllm
    return 1
  fi
}

ensure_vllm_for_model() {
  local model="$1"

  if [[ -n "${_VLLM_STARTED:-}" ]]; then
    if [[ "${_VLLM_MODEL:-}" == "${model}" ]]; then
      if kill -0 "${_VLLM_PID}" >/dev/null 2>&1; then
        return 0
      fi
      cleanup_vllm
    else
      cleanup_vllm
    fi
  fi

  if vllm_has_model "${model}"; then
    return 0
  fi

  start_vllm_for_model "${model}"
}

is_complete_answer_file() {
  local file="$1"
  local expected_model="$2"

  # File must exist and be non-empty
  [[ -s "${file}" ]] || return 1

  # Validate it has the expected number of lines (one per question)
  local line_count
  line_count=$(wc -l < "${file}")
  if [[ "${line_count}" -ne "${EXPECTED_QUESTIONS}" ]]; then
    echo "Warning: ${file} has ${line_count} lines, expected ${EXPECTED_QUESTIONS}" >&2
    return 1
  fi

  # Validate each line is valid JSON and has the expected model_id
  local validation_result
  # File must exist and be non-empty
  [[ -s "${file}" ]] || { echo "missing_file"; return 1; }

  validation_result=$(python3 - "$file" "$expected_model" "$EXPECTED_QUESTIONS" <<'PY'
import json, sys
file = sys.argv[1]
expected_model = sys.argv[2]
expected_questions = int(sys.argv[3])
question_ids = []
seen = set()
try:
    with open(file, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"invalid_json@{i}:{e}")
                sys.exit(1)
            mid = data.get("model_id")
            if mid != expected_model:
                print(f"wrong_model@{i}:{mid}")
                sys.exit(1)
            qid = data.get("question_id")
            if qid is None:
                print(f"missing_question_id@{i}")
                sys.exit(1)
            if qid in seen:
                print(f"duplicate_qid:{qid}")
                sys.exit(1)
            seen.add(qid)
            question_ids.append(qid)
except FileNotFoundError:
    print("missing_file")
    sys.exit(1)
expected_set = set(range(1, expected_questions + 1))
found_set = set(question_ids)
missing = expected_set - found_set
extra = found_set - expected_set
if missing:
    print(f"missing_qids:{sorted(missing)}")
    sys.exit(1)
if extra:
    print(f"unexpected_qids:{sorted(extra)}")
    sys.exit(1)
print("complete")
PY
)

  if [[ "${validation_result}" == "complete" ]]; then
    return 0
  else
    echo "${validation_result}"
    return 1
  fi
}

for model in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Model: ${model}"
  safe_model="$(sanitize_model "${model}")"
  out_file="data/ja_mt_bench/model_answer/${safe_model}.jsonl"

  if result=$(is_complete_answer_file "${out_file}" "${model}"); then
    echo "  ✓ Complete answers already exist at ${out_file}"
    continue
  else
    echo "  ↺ Needs generation: ${result}"
    if [[ -f "${out_file}" ]]; then
      echo "  ⚠ Removing existing incomplete file: ${out_file}"
      rm -f "${out_file}"
    fi
  fi

  env_vars=()
  python_cmd=(python)

  if [[ "${model}" == gemini-* ]]; then
    if [[ -z "${GEMINI_API_KEY:-}" ]]; then
      echo "  GEMINI_API_KEY is required for ${model}. Skipping." >&2
      continue
    fi
    env_vars+=(OPENAI_API_KEY="${GEMINI_API_KEY}")
    env_vars+=(OPENAI_API_BASE="https://generativelanguage.googleapis.com/v1beta/openai/")
  elif [[ "${model}" == gpt-* ]]; then
    # Native OpenAI; no base override.
    python_cmd=(mamba run -n "${GEN_ENV_NAME}" python)
  else
    # Assume local vLLM-compatible endpoint.
    if ! ensure_vllm_for_model "${model}"; then
      echo "  ⚠ Unable to prepare vLLM for '${model}'. Skipping." >&2
      continue
    fi
    env_vars+=(OPENAI_API_BASE="${DEFAULT_LOCAL_BASE}")
    python_cmd=(mamba run -n "${GEN_ENV_NAME}" python)
  fi

  echo "  Generating answers -> ${out_file}"
  cmd=("${python_cmd[@]}" gen_api_answer.py \
    --bench-name ja_mt_bench \
    --model "${model}" \
    --parallel "${PARALLEL}")

  if [[ ${#env_vars[@]} -gt 0 ]]; then
    env "${env_vars[@]}" "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
done
