import json
import numpy as np
import pandas as pd
from tabulate import tabulate
import os

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

# Process judgments for each judge
all_results = {}

for judge_idx, (judgment_file, judge_name) in enumerate(zip(judgment_files, judge_names)):
    judgments = load_judgments(judgment_file)

    # Check if using 'model' or 'model_id' key
    key_name = 'model' if 'model' in judgments[0] else 'model_id'

    # First pass: discover all turns that exist
    all_turns = set()
    for j in judgments:
        if 'score' in j:
            turn = j.get('turn', 0)
            all_turns.add(turn)

    all_turns = sorted(all_turns)
    print(f"Found turns: {all_turns} in {judge_name}")

    # Collect all scores by category and turn
    category_scores = {}
    overall_scores = {}

    for turn in all_turns:
        category_scores[turn] = {cat: [] for cat in categories}
        overall_scores[turn] = []

    # Process judgments
    for j in judgments:
        if 'score' not in j:
            continue

        turn = j.get('turn', 0)
        question_id = j['question_id']
        score = j['score']
        category = question_categories.get(question_id, "Unknown")

        # Add to overall scores
        overall_scores[turn].append(score)

        # Add to category scores
        if category in categories:
            category_scores[turn][category].append(score)

    # Calculate averages
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

# Create the main comparison table
def create_comparison_table():
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

    return "MT-Bench Scores by Category and Judge\n" + table

# Create a simpler summary table (just averages)
def create_summary_table():
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

# Print both tables
print("\n" + "="*80)
print(create_summary_table())

print("\n" + "="*80)
print(create_comparison_table())

# Save the tables to a file
with open('judge_comparison_table.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write(create_summary_table() + "\n")
    f.write("\n" + "="*80 + "\n")
    f.write(create_comparison_table() + "\n")

print(f"\nTables saved to 'judge_comparison_table.txt'")
