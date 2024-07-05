# What has been added to the multilingual fork?
All of the following datasets have been translated using machine learning before being checked by a native speaker of that language. References answers are added where available. When reference answers are unavailable, math, reasoning and coding questions are removed.
* [German MT-Bench](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman) [source](https://github.com/mayflower/FastEval/blob/main/data/mt-bench/questions_de.json)
* [French MT-Bench](https://huggingface.co/datasets/bofenghuang/mt-bench-french)
* [Indonesian MT-Bench](https://huggingface.co/datasets/lightblue/mt_bench_indonesian)
* [Japanese MT-Bench](https://github.com/Stability-AI/FastChat/tree/jp-stable/fastchat/llm_judge/data/japanese_mt_bench)
* [Polish MT-Bench](https://huggingface.co/datasets/lightblue/mt_bench_polish)
* [Russian MT-Bench](https://huggingface.co/datasets/dim/mt_bench_ru) (no reference answers)
* [Vietnamese MT-Bench](https://huggingface.co/datasets/lightblue/mt_bench_vietnamese)
* [Chinese (simplified) MT-Bench](https://huggingface.co/datasets/GeneZC/MT-Bench-ZH/blob/main/data/mt_bench_zh/question.jsonl)

These can be run by specifying `--bench-name` when running the `gen_*.py` commands. The values for the above languages, respectively, are `de_mt_bench`, `fr_mt_bench`, `id_mt_bench`, `ja_mt_bench`, `pl_mt_bench`, `ru_mt_bench`, `vi_mt_bench`, and `zh_mt_bench`. To run the original English benchmark, use ` mt_bench`.

We have also added the ability to evaluate with gpt-4-turbo and gpt-4o by simply duplicating all the reference answers that existed for gpt-4.

# How to use

```bash
OPENAI_API_KEY=<YOUR_API_KEY> python gen_api_answer.py --bench-name ru_mt_bench --model gpt-3.5-turbo --parallel 4
OPENAI_API_KEY=<YOUR_API_KEY> python gen_judgment.py --bench-name ru_mt_bench --model-list gpt-3.5-turbo --judge-model gpt-4-turbo --mode single --parallel 4
```

Please refer to the original documentation for how to run models locally etc.

# Adding more languages

I would really really really really appreciate if someone would submit a pull request adding other languages. As long as a native speaker has checked the contents of the questions and reference answers, I would be happy to add them to this repository.

### How to contribute

1. Translate (either manually or by using a high quality translation service such as GPT-4 or Claude 3 Opus) both turns of the original questions of the [English MT-Bench dataset](https://github.com/lm-sys/FastChat/blob/f22f2194c9152a25d2987e5118206e3bbb9efd5e/fastchat/llm_judge/data/mt_bench/question.jsonl).
3. Manually check these translations using at least one native speaker of the target language. Fix any strange translations or unnatural wordings.
4. Generate reference answers for the math, reasoning, and coding questions by either manually writing them or by inputting them into a high quality LLM such as GPT-4 or Claude 3 Opus. Note that you **DO NOT NEED** to generate answers for all questions, only the math, reasoning, and coding questions.
5. Manually check these answers, making sure that they sound natural and fix any factual or logical inaccuracies from the answers.
6. Fork this repository.
7. Create a new folder in the `fastchat/llm_judge/data` directory named `xx_mt_bench`, where xx is the language code of your language. For example, an Arabic translation would be called `ar_mt_bench`.
8. In this folder, add a file called `question.jsonl` that has the questions in a JSONL file format in the following style:
```jsonl
{"question_id": 81, "category": "CATEGORY_NAME", "turns": ["FIRST TURN OF CONVERSATION", "SECOND TURN OF CONVERSATION"]}
{"question_id": 82, "category": "CATEGORY_NAME", "turns": ["FIRST TURN OF CONVERSATION", "SECOND TURN OF CONVERSATION"]}
```
N.B. Please keep the `question_id` and `category` consistent with the original English version.

7. In the `fastchat/llm_judge/data/xx_mt_bench` directory, add a new folder called `reference answer`.
8. Add the reference answers under the name `gpt-4.jsonl` in the following format:
```jsonl
{"index":101,"category":"CATEGORY_NAME","choices":[{"index":0,"turns":["REFERENCE ANSWER TO FIRST TURN PROMPT", "REFERENCE ANSWER TO SECOND TURN PROMPT"]}]}
{"index":102,"category":"CATEGORY_NAME","choices":[{"index":0,"turns":["REFERENCE ANSWER TO FIRST TURN PROMPT", "REFERENCE ANSWER TO SECOND TURN PROMPT"]}]}
```
Again keeping the `question_id` and `category` consistent with the original English version.

9. Finally, commit these changes and submit a pull request to merge into this repo.

If you are not familiar with coding or using git, then feel free to just submit an issue (from the issue tab near the top) and if you send me the translations your have made, I would be happy to fix them up into the correct format.

With your co-operation, we can assess more languages and make LLMs more useful together!

# ↓ ORIGINAL REPO DOCUMENTATION ↓

# LLM Judge
| [Paper](https://arxiv.org/abs/2306.05685) | [Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) |

In this package, you can use MT-bench questions and prompts to evaluate your models with LLM-as-a-judge.
MT-bench is a set of challenging multi-turn open-ended questions for evaluating chat assistants.
To automate the evaluation process, we prompt strong LLMs like GPT-4 to act as judges and assess the quality of the models' responses.

## Contents
- [Install](#install)
- [Review Pre-Generated Model Answers and Judgments](#review-pre-generated-model-answers-and-judgments)
- [MT-Bench](#mt-bench)
- [Agreement Computation](#agreement-computation)
- [Datasets](#datasets)
- [Citation](#citation)

## Install
```
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e ".[model_worker,llm_judge]"
```

## Review Pre-Generated Model Answers and Judgments
We provide pre-generated model answers and judgments for some models.
You can view them at this [demo](https://huggingface.co/spaces/lmsys/mt-bench).

To download the pre-generated data, use
```
python3 download_mt_bench_pregenerated.py
```

After downloading the data, you can view them locally by
```
python3 qa_browser.py --share
```
You can use this QA browser to view the answers generated by you later.

## MT-Bench

### Evaluate a model on MT-bench

#### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer.py --model-path [MODEL-PATH] --model-id [MODEL-ID]
```
Arguments:
  - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.
  - `[MODEL-ID]` is a name you give to the model.

e.g.,
```
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
```
The answers will be saved to `data/mt_bench/model_answer/[MODEL-ID].jsonl`.

To make sure FastChat loads the correct prompt template, see the supported models and how to add a new model [here](../../docs/model_support.md#how-to-support-a-new-model).

You can also specify `--num-gpus-per-model` for model parallelism (needed for large 65B models) and `--num-gpus-total` to parallelize answer generation with multiple GPUs.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
For each turn, GPT-4 will give a score on a scale of 10. We then compute the average score on all turns.

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

e.g.,
```
python gen_judgment.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4 --parallel 2
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show the scores for selected models
  ```
  python show_result.py --model-list vicuna-13b-v1.3 alpaca-13b llama-13b claude-v1 gpt-3.5-turbo gpt-4
  ```
- Show all scores
  ```
  python show_result.py
  ```

---

### Other grading options
Besides score-based single-answer grading, we also support two additional grading options based on win rates:
- `pariwise-baseline`: run pairwise comparison against a baseline model.
- `pairwise-all`: run pairwise comparison between all model pairs on all questions.

#### Option 2: pairwise comparison against a baseline (default: gpt-3.5-turbo)

- Generate GPT-4 judgments
```
python gen_judgment.py --mode pairwise-baseline --model-list vicuna-13b-v1.3 alpaca-13b llama-13b --parallel 2
```
The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_pair.jsonl`

- Show results
```
python show_result.py --mode pairwise-baseline
```

#### Option 3: Run GPT-4 judge with all pair comparisons

Another option is to run pairwise comparisons on all possible pairs.
This could be more expensive when #models increases, but it gives you a more comprehensive information.

```
python gen_judgment.py --mode pairwise-all --model-list [LIST-OF-MODEL-ID] --parallel [num-concurrent-api-call]
```

```
python show_result.py --mode pairwise-all
```

### How to get GPT-3.5/GPT-4/Claude's answer?
- `python gen_api_answer.py --model [MODEL-NAME]` to generate GPT-3.5/4 and Claude's answers.


### How to plot the radar figure?

You can use this [colab notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO) to plot the radar figure for MT-bench.

<img src="data/mt_bench/misc/radar.png" width="600" height="450">


### Other backends
We can also use vLLM for answer generation, which can be faster for the models supported by vLLM.

1. Launch a vLLM worker
```
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.vllm_worker --model-path [MODEL-PATH]
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```
  - Arguments:
    - `[MODEL-PATH]` is the path to the weights, which can be a local folder or a Hugging Face repo ID.

2. Generate the answers
```
python gen_api_answer.py --model [MODEL-NAME] --openai-api-base http://localhost:8000/v1 --parallel 50
```
  - Arguments:
    - `[MODEL-NAME]` is the name of the model from Step 1.
    - `--parallel` is the number of concurrent API calls to the vLLM worker.


## Agreement Computation
We released 3.3K human annotations for model responses generated by 6 models in response to 80 MT-bench questions. The dataset is available at [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments).

This Colab [notebook](https://colab.research.google.com/drive/1ctgygDRJhVGUJTQy8-bRZCl1WNcT8De6?usp=sharing) shows how to compute the agreement between humans and GPT-4 judge with the dataset. Our results show that humans and GPT-4 judge achieve over 80\% agreement, the same level of agreement between humans.

## Datasets
- [Chatbot Arena Conversation Dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- [MT-bench Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)


## Citation
Please cite the following paper if you find the code or datasets helpful.
```
@misc{zheng2023judging,
      title={Judging LLM-as-a-judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
