import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import os
import argparse

# File paths for the judgment files - modify if needed
judgment_files = [
    "data/ja_mt_bench/model_judgment/gpt-4-turbo_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-4o_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-4.1-2025-04-14_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-4.1-mini-2025-04-14_single.jsonl"
]

judge_names = [
    "GPT-4-Turbo",
    "GPT-4o",
    "GPT-4.1",
    "GPT-4.1-mini"
]

# Function to load judgments
def load_judgments(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Load questions to get categories
def load_questions(file_path="data/ja_mt_bench/question.jsonl"):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

questions = load_questions()
question_categories = {q["question_id"]: q.get("category", "Unknown") for q in questions}

# Get all unique categories
categories = sorted(set(question_categories.values()))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare judge scores for MT-Bench')
parser.add_argument('--include-incomplete', action='store_true', 
                    help='Include models that don\'t have scores from all judges')
args = parser.parse_args()

# Process judgments for each judge
all_results = {}
model_results = {}
models_with_scores = {}

for judge_idx, (judgment_file, judge_name) in enumerate(zip(judgment_files, judge_names)):
    judgments = load_judgments(judgment_file)

    # Check if using 'model' or 'model_id' key
    key_name = 'model' if 'model' in judgments[0] else 'model_id'

    # Get all unique models
    all_models = set(j.get(key_name, 'unknown') for j in judgments if key_name in j)
    all_models = sorted(all_models)
    print(f"Found models: {all_models} in {judge_name}")
    
    # Keep track of which models have scores from which judges
    for model in all_models:
        if model not in models_with_scores:
            models_with_scores[model] = set()
        models_with_scores[model].add(judge_name)

    # First pass: discover all turns that exist
    all_turns = set()
    for j in judgments:
        if 'score' in j:
            turn = j.get('turn', 0)
            all_turns.add(turn)

    all_turns = sorted(all_turns)
    print(f"Found turns: {all_turns} in {judge_name}")

    # Collect all scores by model, category and turn
    # For the summary (all models combined)
    category_scores = {}
    overall_scores = {}

    for turn in all_turns:
        category_scores[turn] = {cat: [] for cat in categories}
        overall_scores[turn] = []

    # For per-model results
    model_category_scores = {model: {} for model in all_models}
    model_overall_scores = {model: {} for model in all_models}

    for model in all_models:
        for turn in all_turns:
            model_category_scores[model][turn] = {cat: [] for cat in categories}
            model_overall_scores[model][turn] = []

    # Process judgments
    for j in judgments:
        if 'score' not in j:
            continue

        turn = j.get('turn', 0)
        question_id = j['question_id']
        score = j['score']
        category = question_categories.get(question_id, "Unknown")
        model = j.get(key_name, 'unknown')

        # Add to overall scores (for summary)
        overall_scores[turn].append(score)

        # Add to category scores (for summary)
        if category in categories:
            category_scores[turn][category].append(score)

        # Add to model-specific scores
        if model in all_models:
            # Add to model overall scores
            model_overall_scores[model][turn].append(score)

            # Add to model category scores
            if category in categories:
                model_category_scores[model][turn][category].append(score)

    # Calculate averages for summary (all models combined)
    results = {}

    # Overall averages by turn
    for turn in all_turns:
        if overall_scores[turn]:
            results[f'Turn {turn}'] = np.mean(overall_scores[turn])
        else:
            results[f'Turn {turn}'] = np.nan

    # Overall average across all turns
    all_scores = []
    for turn_scores in overall_scores.values():
        all_scores.extend(turn_scores)
    if all_scores:
        results['Overall Average'] = np.mean(all_scores)
    else:
        results['Overall Average'] = np.nan

    # Category averages by turn
    for turn in all_turns:
        for category in categories:
            if category_scores[turn][category]:
                key = f'{category} (Turn {turn})'
                results[key] = np.mean(category_scores[turn][category])
            else:
                key = f'{category} (Turn {turn})'
                results[key] = np.nan

    # Category averages across all turns
    for category in categories:
        cat_scores = []
        for turn in all_turns:
            cat_scores.extend(category_scores[turn][category])
        if cat_scores:
            results[f'{category} (Average)'] = np.mean(cat_scores)
        else:
            results[f'{category} (Average)'] = np.nan

    all_results[judge_name] = results

    # Calculate per-model results
    model_judge_results = {}

    for model in all_models:
        model_results_dict = {}

        # Overall averages by turn
        for turn in all_turns:
            if model_overall_scores[model][turn]:
                model_results_dict[f'Turn {turn}'] = np.mean(model_overall_scores[model][turn])
            else:
                model_results_dict[f'Turn {turn}'] = np.nan

        # Overall average across all turns
        all_model_scores = []
        for turn_scores in model_overall_scores[model].values():
            all_model_scores.extend(turn_scores)
        if all_model_scores:
            model_results_dict['Overall Average'] = np.mean(all_model_scores)
        else:
            model_results_dict['Overall Average'] = np.nan

        # Category averages by turn
        for turn in all_turns:
            for category in categories:
                if model_category_scores[model][turn][category]:
                    key = f'{category} (Turn {turn})'
                    model_results_dict[key] = np.mean(model_category_scores[model][turn][category])
                else:
                    key = f'{category} (Turn {turn})'
                    model_results_dict[key] = np.nan

        # Category averages across all turns
        for category in categories:
            cat_scores = []
            for turn in all_turns:
                cat_scores.extend(model_category_scores[model][turn][category])
            if cat_scores:
                model_results_dict[f'{category} (Average)'] = np.mean(cat_scores)
            else:
                model_results_dict[f'{category} (Average)'] = np.nan

        model_judge_results[model] = model_results_dict

    if judge_name not in model_results:
        model_results[judge_name] = {}
    model_results[judge_name] = model_judge_results

# Create the main comparison table for a specific model
def create_comparison_table(model_name=None, require_all_judges=True):
    if model_name is None:
        # This is the original behavior - all models blended
        rows = []

        # Get all categories and turns for proper ordering
        sample_results = list(all_results.values())[0]
        all_turns = sorted([int(k.split()[1]) for k in sample_results.keys() if k.startswith('Turn ')])

        # Overall scores first
        overall_row = ['Overall Average']
        for judge_name in judge_names:
            if judge_name in all_results:
                score = all_results[judge_name]['Overall Average']
                overall_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
            else:
                overall_row.append("N/A")
        rows.append(overall_row)

        # Turn-specific overall scores
        for turn in all_turns:
            turn_row = [f'Overall (Turn {turn})']
            for judge_name in judge_names:
                if judge_name in all_results:
                    score = all_results[judge_name].get(f'Turn {turn}', np.nan)
                    turn_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                else:
                    turn_row.append("N/A")
            rows.append(turn_row)

        # Add separator
        separator_row = ['-' * 20] + ['-' * 10 for _ in judge_names]
        rows.append(separator_row)

        # Category averages (across all turns)
        for category in categories:
            cat_row = [f'{category} (Average)']
            for judge_name in judge_names:
                if judge_name in all_results:
                    score = all_results[judge_name].get(f'{category} (Average)', np.nan)
                    cat_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                else:
                    cat_row.append("N/A")
            rows.append(cat_row)

        # Category scores by turn
        for turn in all_turns:
            # Add turn separator
            turn_separator = [f'--- Turn {turn} ---'] + ['-' * 10 for _ in judge_names]
            rows.append(turn_separator)

            for category in categories:
                cat_turn_row = [f'{category}']
                for judge_name in judge_names:
                    if judge_name in all_results:
                        score = all_results[judge_name].get(f'{category} (Turn {turn})', np.nan)
                        cat_turn_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                    else:
                        cat_turn_row.append("N/A")
                rows.append(cat_turn_row)

        # Create headers
        headers = ["Category"] + judge_names

        # Format the table using tabulate
        table = tabulate(rows, headers=headers, tablefmt="simple")

        return "MT-Bench Scores by Category and Judge (All Models)\n" + table
    else:
        # This is for a specific model
        rows = []

        # Use the first judge that has this model to determine turns
        sample_judge = None
        for judge_name in judge_names:
            if judge_name in model_results and model_name in model_results[judge_name]:
                sample_judge = judge_name
                break

        if not sample_judge:
            return f"No data found for model: {model_name}"

        sample_results = model_results[sample_judge][model_name]
        all_turns = sorted([int(k.split()[1]) for k in sample_results.keys() if k.startswith('Turn ')])

        # Overall scores first
        overall_row = ['Overall Average']
        for judge_name in judge_names:
            if judge_name in model_results and model_name in model_results[judge_name]:
                score = model_results[judge_name][model_name]['Overall Average']
                overall_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
            else:
                overall_row.append("N/A")
        rows.append(overall_row)

        # Turn-specific overall scores
        for turn in all_turns:
            turn_row = [f'Overall (Turn {turn})']
            for judge_name in judge_names:
                if judge_name in model_results and model_name in model_results[judge_name]:
                    score = model_results[judge_name][model_name].get(f'Turn {turn}', np.nan)
                    turn_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                else:
                    turn_row.append("N/A")
            rows.append(turn_row)

        # Add separator
        separator_row = ['-' * 20] + ['-' * 10 for _ in judge_names]
        rows.append(separator_row)

        # Category averages (across all turns)
        for category in categories:
            cat_row = [f'{category} (Average)']
            for judge_name in judge_names:
                if judge_name in model_results and model_name in model_results[judge_name]:
                    score = model_results[judge_name][model_name].get(f'{category} (Average)', np.nan)
                    cat_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                else:
                    cat_row.append("N/A")
            rows.append(cat_row)

        # Category scores by turn
        for turn in all_turns:
            # Add turn separator
            turn_separator = [f'--- Turn {turn} ---'] + ['-' * 10 for _ in judge_names]
            rows.append(turn_separator)

            for category in categories:
                cat_turn_row = [f'{category}']
                for judge_name in judge_names:
                    if judge_name in model_results and model_name in model_results[judge_name]:
                        score = model_results[judge_name][model_name].get(f'{category} (Turn {turn})', np.nan)
                        cat_turn_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
                    else:
                        cat_turn_row.append("N/A")
                rows.append(cat_turn_row)

        # Create headers
        headers = ["Category"] + judge_names

        # Format the table using tabulate
        table = tabulate(rows, headers=headers, tablefmt="simple")

        return f"{model_name}\n" + table

# Create a function to get all unique models across all judges
def get_all_unique_models(require_all_judges=True):
    """Get all unique models, optionally filtering to only those with all judges.
    
    Args:
        require_all_judges: If True, only return models that have scores from all judges
        
    Returns:
        List of model names sorted alphabetically
    """
    unique_models = set()
    
    if require_all_judges:
        # Only include models that have scores from all judges
        for model, judges in models_with_scores.items():
            if len(judges) == len(judge_names):
                unique_models.add(model)
    else:
        # Include all models regardless of judge coverage
        for judge_name in model_results:
            for model_name in model_results[judge_name]:
                unique_models.add(model_name)
                
    return sorted(unique_models)

# Create a simpler summary table (just averages)
def create_summary_table(require_all_judges=True):
    rows = []

    # Overall average
    overall_row = ['Overall Average']
    for judge_name in judge_names:
        if judge_name in all_results:
            score = all_results[judge_name]['Overall Average']
            overall_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
        else:
            overall_row.append("N/A")
    rows.append(overall_row)

    # Category averages
    for category in categories:
        cat_row = [category]
        for judge_name in judge_names:
            if judge_name in all_results:
                score = all_results[judge_name].get(f'{category} (Average)', np.nan)
                cat_row.append(f"{score:.2f}" if not np.isnan(score) else "N/A")
            else:
                cat_row.append("N/A")
        rows.append(cat_row)

    # Create headers
    headers = ["Category"] + judge_names

    # Format the table using tabulate
    table = tabulate(rows, headers=headers, tablefmt="simple")

    return "MT-Bench Summary: Average Scores by Category and Judge\n" + table

# Get all unique models
unique_models = get_all_unique_models(not args.include_incomplete)

# Print the summary table
print("\n" + "="*80)
print(create_summary_table(not args.include_incomplete))

# Print the all-models comparison table
print("\n" + "="*80)
print(create_comparison_table(require_all_judges=not args.include_incomplete))

# Print detailed tables for each model
for model in unique_models:
    print("\n" + "="*80)
    print(create_comparison_table(model, require_all_judges=not args.include_incomplete))

# Save the tables to a file
output_filename = 'judge_comparison_table.txt'
if args.include_incomplete:
    output_filename = 'judge_comparison_table_all_models.txt'
    
with open(output_filename, 'w') as f:
    f.write("="*80 + "\n")
    if args.include_incomplete:
        f.write("NOTE: Including models that don't have scores from all judges\n")
    else:
        f.write("NOTE: Only showing models that have scores from all judges\n")
    f.write("="*80 + "\n")
    f.write(create_summary_table(not args.include_incomplete) + "\n")
    f.write("\n" + "="*80 + "\n")
    f.write(create_comparison_table(require_all_judges=not args.include_incomplete) + "\n")
    
    # Add per-model tables
    for model in unique_models:
        f.write("\n" + "="*80 + "\n")
        f.write(create_comparison_table(model, require_all_judges=not args.include_incomplete) + "\n")

print(f"\nTables saved to '{output_filename}'")
