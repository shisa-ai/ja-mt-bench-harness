#!/usr/bin/env bash
set -euo pipefail

# Models to judge (edit this list as needed).
MODELS=(
  "shisa-ai/shisa-v2-llama3.1-405b"
  "shisa-ai/shisa-v2-llama3.3-70b"
  "shisa-ai/shisa-v2-qwen2.5-32b"
  "shisa-ai/shisa-v2-mistral-small-24b"
  "shisa-ai/shisa-v2-unphi4-14b"
  "shisa-ai/shisa-v2-mistral-nemo-12b"
  "shisa-ai/shisa-v2-qwen2.5-7b"
  "shisa-ai/shisa-v2-llama3.1-8b"
  "shisa-ai/llama3.3-70b-merge-nuslerp-168-176-base"
  "shisa-ai/168-llama3.3-70b-v2.1-sft"
  "shisa-ai/144-lfm2-1.2b-v2.1-dpo-1e-6"
  "shisa-ai/171-llama3.2-3b-v2.1-153sft-dpo-2e7"
  "shisa-ai/170-qwen3-8b-v2.1-dpo-1.2e7"
  "shisa-ai/166-qwen3-30b-a3b-v2.1-dpo-1.2e7"
  "shisa-ai/shisa-v2.1-unphi4-14b-152-155-149-nuslerp"
  "LiquidAI/LFM2-1.2B"
  "Qwen/Qwen3-8B"
  "meta-llama/Llama-3.3-70B-Instruct"
  "meta-llama/Llama-3.1-405B-Instruct"
  "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4"
  "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
)

# Judge models to use.
JUDGES=(
  "gpt-4-turbo-2024-04-09"
  "gpt-4o-2024-08-06"
  "gpt-4.1-2025-04-14"
  "gpt-5.1-2025-11-13"
)

PARALLEL=2
BENCH=ja_mt_bench
EXPECTED_QUESTIONS=80

# Check if a judgment file has valid, complete judgments for a given model.
# Returns 0 (success) if the model has complete judgments, 1 otherwise.
judgment_has_complete_model() {
  local jsonl="$1"
  local target_model="$2"

  # If file doesn't exist, clearly not complete
  [[ -f "${jsonl}" ]] || return 1

  # Use Python to robustly check if model exists with all expected questions
  python3 - "$jsonl" "$target_model" "$EXPECTED_QUESTIONS" <<'PY'
import json, sys

try:
    jsonl_path = sys.argv[1]
    target_model = sys.argv[2]
    expected_questions = int(sys.argv[3])

    question_ids_found = set()

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} is not valid JSON: {e}", file=sys.stderr)
                sys.exit(1)

            # Check if this record is for our target model
            model = rec.get("model") or rec.get("model_id")
            if model == target_model:
                question_id = rec.get("question_id")
                if question_id is None:
                    print(f"Error: Record for {target_model} at line {line_num} missing question_id", file=sys.stderr)
                    sys.exit(1)
                question_ids_found.add(question_id)

    # Check if we have all expected questions
    if len(question_ids_found) == expected_questions:
        # Verify we have consecutive IDs from 1 to expected_questions
        expected_ids = set(range(1, expected_questions + 1))
        if question_ids_found == expected_ids:
            print("complete")
            sys.exit(0)
        else:
            missing = expected_ids - question_ids_found
            extra = question_ids_found - expected_ids
            if missing:
                print(f"incomplete: missing question IDs: {sorted(missing)}", file=sys.stderr)
            if extra:
                print(f"incomplete: unexpected question IDs: {sorted(extra)}", file=sys.stderr)
            sys.exit(1)
    elif len(question_ids_found) > 0:
        print(f"incomplete: found {len(question_ids_found)}/{expected_questions} questions", file=sys.stderr)
        sys.exit(1)
    else:
        print("not_found")
        sys.exit(0)

except FileNotFoundError:
    print("not_found")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

for model in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Model: ${model}"
  for judge in "${JUDGES[@]}"; do
    out_file="data/${BENCH}/model_judgment/${judge}_single.jsonl"

    # Wrap call in a conditional to avoid set -e killing the script on expected non-zero
    if check_result=$(judgment_has_complete_model "${out_file}" "${model}" 2>&1); then
      exit_code=0
    else
      exit_code=$?
    fi

    if [[ "${check_result}" == "complete" ]]; then
      echo "  ✓ ${judge}: complete judgments already exist in ${out_file}"
      continue
    elif [[ "${check_result}" == "not_found" ]]; then
      echo "  ↺ ${judge}: not found in ${out_file}, will generate judgments"
    else
      # Incomplete or error – append new judgments (canonicalization will handle duplicates)
      echo "  ↺ ${judge}: incomplete/invalid in ${out_file} (${check_result}), appending new judgments"
    fi

    echo "    Running gen_judgment -> ${out_file}"
    python gen_judgment.py \
      --bench-name "${BENCH}" \
      --model-list "${model}" \
      --judge-model "${judge}" \
      --mode single \
      --parallel "${PARALLEL}" \
      --skip_confirm
  done
done
