# ja-mt-bench-harness

This repo is forked 2025-05 specifically to run repeatable "standard" Japanese MT-Bench results that are suitable for reporting/comparison with other models/measurements.
- Designed to use OpenAI-compatible API (eg for vLLM) for answer generation
- Use of the standard OpenAI GPT-4 Turbo (`gpt-4-turbo` AKA `gpt-4-turbo-2024-04-09`) for 1:1 comparisons with older models as well as added support/comparison to the latest OpenAI models (`gpt-4.1-2025-04-14` and `gpt-4.1-mini-2025-04-14`)

As of 2025-05 this is the approximate cost for judging a single run (GPT 4.1 tokenizer is more efficient than the old GPT-4 tokenizer for Japanese):

| Model                                | Input $/M | Output $/M | Input Tokens | Output Tokens | Input Cost | Output Cost | Total Cost |
| ------------------------------------ | --------: | ---------: | -----------: | ------------: | ---------: | ----------: | ---------: | 
| gpt-4-turbo (gpt-4-turbo-2024-04-09) | $10.00    | $30.00     | 250,000      | 50,000        | $2.50      | $1.50       | ~$4.00     |
| gpt-4o (gpt-4o-2024-08-06)           | $2.50     | $10.00     | 200,000      | 50,000        | $0.50      | $0.50       | ~$1.00     |
| gpt-4.1-2025-04-14                   | $2.00     | $8.00      | 200,000      | 50,000        | $0.40      | $0.40       | ~$1.00     |
| gpt-4.1-mini-2025-04-14              | $0.40     | $1.60      | 200,000      | 50,000        | $0.08      | $0.08       | ~$0.20     |

This is a fork of:
- [lightblue-tech/multilingual-mt-bench](https://github.com/lightblue-tech/multilingual-mt-bench)
  - [lm-sys/FastChat](https://github.com/lm-sys/FastChat)


## Install
We use `mamba` for setup. If you want to use this for your own models, you may want to fork a copy to store your model output:

```bash
# Checkout
git clone git@github.com:shisa-ai/ja-mt-bench-harness.git
cd ja-mt-bench-harness

# Base env
mamba create -n ja-mt-bench-harness python=3.12
mamba activate ja-mt-bench-harness

# Install w/ editable
pip install -e ".[model_worker,llm_judge]"

# More dependencies
pip install pandas scipy seaborn tabulate -y
```

## Run
You should run `sglang` or `vllm` (or something else to serve your model via OpenAI-compatible API separately)

```bash
cd fastchat/llm_judge

# There is a helper script - you can run interactively
./run.sh 

# or w/ flags
./run.sh --model $MODEL --openai-api-base $BASE_URL

### or if you want to run manually...

# Generate answers - you probably don't need OPENAI_API_KEY if you're running a local model
# --openai-api-base should be something like "http://localhost:8000/v1"
# You can set parallel as high as you want but it's only 80 requests
[OPENAI_API_KEY=YOUR-KEY] python gen_api_answer.py --bench-name ja_mt_bench --model <YOUR-MODEL> [--openai-api-base YOUR-BASE-URL] --parallel 20

# Judge the answers
[OPENAI_API_KEY=YOUR-KEY] python gen_judgment.py --bench-name ja_mt_bench --model-list <YOUR-MODEL> --judge-model gpt-4-turbo --mode single --parallel 20

# Visualize results 
python visualize-results.py

...
```
