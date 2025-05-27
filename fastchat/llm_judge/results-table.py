import os
import json
import argparse
import numpy as np
from tabulate import tabulate
from typing import List, Dict, Any, Optional
from collections import defaultdict

# MT-Bench categories
CATEGORIES = [
    "coding",
    "extraction",
    "humanities",
    "math",
    "reasoning",
    "roleplay",
    "stem",
    "writing"
]

# Mapping from question_id to category based on the actual MT-bench data
QUESTION_CATEGORY_MAP = {
    # coding: 1-10
    1: "coding", 2: "coding", 3: "coding", 4: "coding", 5: "coding",
    6: "coding", 7: "coding", 8: "coding", 9: "coding", 10: "coding",
    
    # extraction: 11-20
    11: "extraction", 12: "extraction", 13: "extraction", 14: "extraction", 15: "extraction",
    16: "extraction", 17: "extraction", 18: "extraction", 19: "extraction", 20: "extraction",
    
    # humanities: 21-30
    21: "humanities", 22: "humanities", 23: "humanities", 24: "humanities", 25: "humanities",
    26: "humanities", 27: "humanities", 28: "humanities", 29: "humanities", 30: "humanities",
    
    # math: 31-40
    31: "math", 32: "math", 33: "math", 34: "math", 35: "math",
    36: "math", 37: "math", 38: "math", 39: "math", 40: "math",
    
    # reasoning: 41-50
    41: "reasoning", 42: "reasoning", 43: "reasoning", 44: "reasoning", 45: "reasoning",
    46: "reasoning", 47: "reasoning", 48: "reasoning", 49: "reasoning", 50: "reasoning",
    
    # roleplay: 51-60
    51: "roleplay", 52: "roleplay", 53: "roleplay", 54: "roleplay", 55: "roleplay",
    56: "roleplay", 57: "roleplay", 58: "roleplay", 59: "roleplay", 60: "roleplay",
    
    # stem: 61-70
    61: "stem", 62: "stem", 63: "stem", 64: "stem", 65: "stem",
    66: "stem", 67: "stem", 68: "stem", 69: "stem", 70: "stem",
    
    # writing: 71-80
    71: "writing", 72: "writing", 73: "writing", 74: "writing", 75: "writing",
    76: "writing", 77: "writing", 78: "writing", 79: "writing", 80: "writing"
}

def analyze_mt_bench_scores(
    bench_name: str,
    model_list: List[str],
    judge_name_filter: Optional[str] = None, 
) -> None:
    """Analyze MT-Bench scores and generate tables using tabulate.

    Args:
        bench_name: Name of the benchmark.
        model_list: List of models to analyze.
        judge_name_filter: Specific judge to use (if None, uses all available judges).
    """
    judgment_dir = f"data/{bench_name}/model_judgment"
    if not os.path.exists(judgment_dir):
        print(f"Judgment directory not found: {judgment_dir}")
        return

    judgment_files = []
    available_judge_names = []

    if judge_name_filter:
        judgment_file_path = os.path.join(judgment_dir, f"{judge_name_filter}_single.jsonl")
        if os.path.exists(judgment_file_path):
            judgment_files.append(judgment_file_path)
            available_judge_names.append(judge_name_filter)
        else:
            print(f"Judgment file not found for specified judge: {judge_name_filter}")
            return
    else:
        for filename in os.listdir(judgment_dir):
            if filename.endswith("_single.jsonl"):
                judgment_files.append(os.path.join(judgment_dir, filename))
                judge_name_from_file = filename.replace("_single.jsonl", "")
                available_judge_names.append(judge_name_from_file)

    if not judgment_files:
        print(f"No judgment files found in {judgment_dir}")
        return

    print(f"Found {len(judgment_files)} judgment file(s) for judge(s): {sorted(list(set(available_judge_names)))}")

    all_judgments_by_judge = {}
    for judgment_file, current_judge_name in zip(judgment_files, available_judge_names):
        try:
            with open(judgment_file, "r") as f:
                judgments = [json.loads(line) for line in f]
                all_judgments_by_judge[current_judge_name] = judgments
                print(f"Loaded {len(judgments)} judgments from {judgment_file} for judge '{current_judge_name}'")
        except Exception as e:
            print(f"Error loading {judgment_file}: {e}")
            continue
    
    if not all_judgments_by_judge:
        print("No judgments were successfully loaded.")
        return

    question_file = f"data/{bench_name}/question.jsonl"
    question_id_to_category_map = {}
    try:
        with open(question_file, "r") as f:
            questions_data = [json.loads(line) for line in f]
        for q_data in questions_data:
            question_id_to_category_map[q_data["question_id"]] = q_data.get("category", "unknown")
        print(f"Successfully loaded category mapping from {question_file}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {question_file} ({e}). Falling back to predefined QUESTION_CATEGORY_MAP.")
        # Populate from QUESTION_CATEGORY_MAP as a fallback
        # This ensures that if the file is missing, we can still try to map known IDs
        question_id_to_category_map = {k: v for k, v in QUESTION_CATEGORY_MAP.items()} 

    scores_by_judge_model_cat_turn = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    all_models_found_in_judgments = set()

    for current_judge_name, judgments in all_judgments_by_judge.items():
        if not judgments: continue
        # Determine key for model name ('model' or 'model_id') from the first judgment entry
        model_key = "model" if "model" in judgments[0] else "model_id"
        
        for judgment_item in judgments:
            if "score" not in judgment_item:
                continue
            model_name = judgment_item[model_key]
            all_models_found_in_judgments.add(model_name)

            # Only process models that are in the requested model_list
            if model_name not in model_list:
                continue

            question_id = judgment_item["question_id"]
            turn = judgment_item.get("turn", 1) # Default to turn 1 if not specified
            score = judgment_item["score"]
            
            category = question_id_to_category_map.get(question_id, QUESTION_CATEGORY_MAP.get(question_id, "unknown"))
            if category == "unknown":
                print(f"Warning: Unknown category for question_id {question_id} for model {model_name}, judge {current_judge_name}")

            scores_by_judge_model_cat_turn[current_judge_name][model_name][category][turn].append(score)

    # Filter model_list to those actually found and requested
    models_to_process = sorted([m for m in model_list if m in all_models_found_in_judgments])
    if not models_to_process:
        print(f"None of the models specified in --model-list ({model_list}) were found in the judgment files from judge(s): {available_judge_names}.")
        return
    print(f"Processing scores for models: {models_to_process}")

    # Calculate average scores (combining turns 1 and 2) for each category
    avg_scores_judge_model_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    judges_with_scores_for_processed_models = set()

    for current_judge_name, model_data in scores_by_judge_model_cat_turn.items():
        for model_name, cat_data in model_data.items():
            if model_name not in models_to_process:
                continue
            for cat, turn_data in cat_data.items():
                all_turn_scores = []
                # Ensure turns are integers if they come from JSON keys
                if 1 in turn_data: all_turn_scores.extend(turn_data[1])
                if 2 in turn_data: all_turn_scores.extend(turn_data[2])
                
                if all_turn_scores:
                    avg_scores_judge_model_cat[current_judge_name][model_name][cat] = np.mean(all_turn_scores)
                    judges_with_scores_for_processed_models.add(current_judge_name)
                else:
                    # If a category was expected but had no scores (e.g. only turn 3 which we ignore)
                    avg_scores_judge_model_cat[current_judge_name][model_name][cat] = float('nan') 

    final_judge_columns = sorted(list(judges_with_scores_for_processed_models))
    if not final_judge_columns:
        print(f"No scores found from any judge for the specified models: {models_to_process}")
        return

    for model_name in models_to_process:
        print(f"\n--- Scores for Model: {model_name} ---")
        table_data = []
        headers = ["Category"] + final_judge_columns

        # category_scores_for_model = defaultdict(list) # To calculate overall later; not strictly needed for current table output

        for cat in CATEGORIES:
            row = [cat]
            for judge_col_name in final_judge_columns:
                score = avg_scores_judge_model_cat[judge_col_name][model_name].get(cat, float('nan'))
                row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                # if not np.isnan(score):
                #     category_scores_for_model[judge_col_name].append(score) # Not strictly needed for current table output
            table_data.append(row)

        # Add Overall summary row
        overall_row = ["Overall"]
        # overall_scores_by_judge = defaultdict(list) # Not strictly needed for current table output
        for judge_col_name in final_judge_columns:
            scores = [avg_scores_judge_model_cat[judge_col_name][model_name].get(c, np.nan) for c in CATEGORIES]
            valid_scores = [s for s in scores if not np.isnan(s)]
            avg = np.mean(valid_scores) if valid_scores else float('nan')
            overall_row.append(f"{avg:.2f}" if not np.isnan(avg) else "N/A")
            # if not np.isnan(avg):
            #      overall_scores_by_judge[judge_col_name].append(avg)
        table_data.append(overall_row)

        # Add Overall xCM (excluding Coding and Math) summary row
        overall_xcm_row = ["Overall xCM"]
        categories_for_xcm = [c for c in CATEGORIES if c not in ["coding", "math"]]
        # overall_xcm_scores_by_judge = defaultdict(list) # Not strictly needed for current table output

        for judge_col_name in final_judge_columns:
            scores_xcm = [avg_scores_judge_model_cat[judge_col_name][model_name].get(c, np.nan) for c in categories_for_xcm]
            valid_scores_xcm = [s for s in scores_xcm if not np.isnan(s)]
            avg_xcm = np.mean(valid_scores_xcm) if valid_scores_xcm else float('nan')
            overall_xcm_row.append(f"{avg_xcm:.2f}" if not np.isnan(avg_xcm) else "N/A")
            # if not np.isnan(avg_xcm):
            #     overall_xcm_scores_by_judge[judge_col_name].append(avg_xcm)
        table_data.append(overall_xcm_row)
        
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description="Analyze MT-Bench scores and generate result tables.")

    parser.add_argument("--bench-name", type=str, default="ja_mt_bench",
                        help="Benchmark name (default: ja_mt_bench)")
    parser.add_argument("--model-list", type=str, nargs="+", required=True,
                        help="List of model IDs to analyze (e.g., model1 model2)")
    parser.add_argument("--judge", type=str, default=None,
                        help="Specific judge to filter by (e.g., gpt-4). If not provided, uses all available judges.")

    args = parser.parse_args()
    
    analyze_mt_bench_scores(
        bench_name=args.bench_name,
        model_list=args.model_list,
        judge_name_filter=args.judge
    )

if __name__ == "__main__":
    main()
