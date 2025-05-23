#!/bin/bash

show_help() {
  echo "Usage: ./run.sh [--model MODEL_NAME] [--openai-api-base API_BASE_URL]"
  echo ""
  echo "Options:"
  echo "  --model MODEL_NAME         Specify the model to benchmark"
  echo "  --openai-api-base URL      Specify the OpenAI API base URL (required for custom models)"
  echo ""
  echo "If no arguments are provided, the script will run in interactive mode."
  echo "The script will perform the following steps:"
  echo "  1. Generate answers using the specified model"
  echo "  2. Generate judgments using multiple judge models"
  echo "  3. Analyze and visualize results"
  echo "  4. Compare judge results"
  echo ""
}

# Show help message each time
show_help

# Parse command line arguments
MODEL=""
OPENAI_API_BASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --openai-api-base)
      OPENAI_API_BASE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Interactive mode if no model is specified
if [ -z "${MODEL}" ]; then
  echo "Running in interactive mode..."
  read -p "Enter model name: " MODEL
  
  # Check if model name contains a slash, prompt for API base if needed
  if [[ "${MODEL}" == */* ]]; then
    if [ -z "${OPENAI_API_BASE}" ]; then
      read -p "Enter OpenAI API base URL (e.g., http://localhost:8080/v1): " OPENAI_API_BASE
    fi
  fi
fi

# Set default API base for models with slashes if not provided
if [[ "${MODEL}" == */* && -z "${OPENAI_API_BASE}" ]]; then
  OPENAI_API_BASE="http://localhost:8080/v1"
  echo "Using default API base: ${OPENAI_API_BASE}"
fi

# Build API base parameter if needed
API_BASE_PARAM=""
if [ ! -z "${OPENAI_API_BASE}" ]; then
  API_BASE_PARAM="--openai-api-base ${OPENAI_API_BASE}"
fi

echo "============================="
echo "Running benchmark for model: ${MODEL}"
if [ ! -z "${OPENAI_API_BASE}" ]; then
  echo "Using API base: ${OPENAI_API_BASE}"
fi
echo "============================="
echo ""

# Answer
echo "Step 1: Generating answers..."
echo "Running: python gen_api_answer.py --bench-name ja_mt_bench --model ${MODEL} ${API_BASE_PARAM}"
if [[ "${MODEL}" == */* ]]; then
  CONCURRENCY=80
else
  CONCURRENCY=20
fi
python gen_api_answer.py --bench-name ja_mt_bench --model ${MODEL} ${API_BASE_PARAM} --parallel ${CONCURRENCY}
echo "Done generating answers."
echo ""

# Judge
echo "============================="
echo ""
echo "Step 2: Generating judgments..."

echo "Running judgment with gpt-4-turbo..."
python gen_judgment.py --bench-name ja_mt_bench --model-list ${MODEL} --judge-model gpt-4-turbo --mode single --parallel 20 --skip_confirm

echo "Running judgment with gpt-4o..."
python gen_judgment.py --bench-name ja_mt_bench --model-list ${MODEL} --judge-model gpt-4o --mode single --parallel 20 --skip_confirm

echo "Running judgment with gpt-4.1-2025-04-14..."
python gen_judgment.py --bench-name ja_mt_bench --model-list ${MODEL} --judge-model gpt-4.1-2025-04-14 --mode single --parallel 20 --skip_confirm

echo "Running judgment with gpt-4.1-mini-2025-04-14..."
python gen_judgment.py --bench-name ja_mt_bench --model-list ${MODEL} --judge-model gpt-4.1-mini-2025-04-14 --mode single --parallel 20 --skip_confirm

echo "Done generating judgments."
echo ""

# Analyze Model
echo "============================="
echo ""
echo "Step 3: Analyzing and visualizing results..."
echo "Running: python visualize-results.py --model-list ${MODEL}"
python visualize-results.py --model-list ${MODEL}
echo "Done analyzing results."
echo ""

# Compare judges
echo "============================="
echo ""
echo "Step 4: Comparing judge results..."
echo "Running: python judge-comparison-stats.py"
python judge-comparison-stats.py
echo "Running: python compare-judges.py"
python compare-judges.py
echo "Done comparing judge results."
echo ""

echo "============================="
echo "Benchmark completed successfully!"
echo "============================="
