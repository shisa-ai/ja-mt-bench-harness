import argparse
import json
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from canonicalize_scores import canonicalize_judgments
from scipy.stats import pearsonr, spearmanr, kendalltau

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Compare judge scores across different models')
parser.add_argument('--filter', type=str, default='',
                   help='Comma-delimited substrings to include only matching models (e.g., "shisa,tokyotech")')
parser.add_argument('--exclude', type=str, default='',
                   help='Comma-delimited substrings to exclude from models (e.g., "LiquidAI,Qwen")')
args = parser.parse_args()

# Process filter and exclude lists
filter_list = [f.strip() for f in args.filter.split(',') if f.strip()]
exclude_list = [e.strip() for e in args.exclude.split(',') if e.strip()]

# File paths for the judgment files - modify if needed
judgment_files = [
    "data/ja_mt_bench/model_judgment/gpt-4-turbo-2024-04-09_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-4o-2024-08-06_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-4.1-2025-04-14_single.jsonl",
    "data/ja_mt_bench/model_judgment/gpt-5.1-2025-11-13_single.jsonl",
    # "data/ja_mt_bench/model_judgment/gpt-4.1-mini-2025-04-14_single.jsonl"
]

judge_names = [
    "GPT-4-Turbo",
    "GPT-4o",
    "GPT-4.1",
    "GPT-5.1",
    # "GPT-4.1-mini",
]


# Function to load judgments (canonicalized to avoid duplicate overweighting)
def load_judgments(file_path):
    with open(file_path, 'r') as f:
        raw = [json.loads(line) for line in f]
    return canonicalize_judgments(raw, file_path)

# Load judgments from the three files
judgments = [load_judgments(file) for file in judgment_files]

# Create dictionaries for easy comparison
# Note: Using 'model' key instead of 'model_id' based on your file format
score_dicts = []
for judge_judgments in judgments:
    # Check if using 'model' or 'model_id' key
    key_name = 'model' if 'model' in judge_judgments[0] else 'model_id'
    # Create dictionary with scores keyed by (question_id, model, turn)
    score_dict = {(j['question_id'], j[key_name], j.get('turn', 0)): j['score']
                  for j in judge_judgments if 'score' in j}
    score_dicts.append(score_dict)

# Get common keys across all three dictionaries
common_keys = set.intersection(*[set(d.keys()) for d in score_dicts])

# Convert to a pandas DataFrame for easier analysis
data = []
for key in common_keys:
    question_id, model, turn = key
    row = {
        'question_id': question_id,
        'model': model,
        'turn': turn
    }
    # Add scores from each judge
    for i, (judge_name, score_dict) in enumerate(zip(judge_names, score_dicts)):
        row[judge_name] = score_dict[key]
    data.append(row)

df = pd.DataFrame(data)

# Apply filtering based on command-line arguments
if filter_list:
    print(f"\n=== Applying filter: including models matching {filter_list} ===")
    df = df[df['model'].apply(lambda m: any(f in m for f in filter_list))]

if exclude_list:
    print(f"\n=== Applying exclusion: excluding models matching {exclude_list} ===")
    df = df[~df['model'].apply(lambda m: any(e in m for e in exclude_list))]

if filter_list or exclude_list:
    print(f"Filtered to {len(df['model'].unique())} unique models: {sorted(df['model'].unique())}\n")

# Calculate overall statistics
print("=== Overall Statistics ===")
for i, judge1 in enumerate(judge_names):
    for j, judge2 in enumerate(judge_names):
        if i < j:  # Compare each pair once
            corr_pearson, p_pearson = pearsonr(df[judge1], df[judge2])
            corr_spearman, p_spearman = spearmanr(df[judge1], df[judge2])
            mean_diff = np.mean(df[judge2] - df[judge1])
            mean_abs_diff = np.mean(np.abs(df[judge2] - df[judge1]))

            print(f"\n{judge1} vs {judge2}:")
            print(f"  Pearson correlation: {corr_pearson:.4f} (p={p_pearson:.4f})")
            print(f"  Spearman correlation: {corr_spearman:.4f} (p={p_spearman:.4f})")
            print(f"  Mean difference ({judge2} - {judge1}): {mean_diff:.4f}")
            print(f"  Mean absolute difference: {mean_abs_diff:.4f}")

            # Also calculate by turn if available
            if 'turn' in df.columns:
                for turn in df['turn'].unique():
                    turn_df = df[df['turn'] == turn]
                    corr_pearson, p_pearson = pearsonr(turn_df[judge1], turn_df[judge2])
                    mean_diff = np.mean(turn_df[judge2] - turn_df[judge1])
                    print(f"  Turn {turn}:")
                    print(f"    Pearson correlation: {corr_pearson:.4f} (p={p_pearson:.4f})")
                    print(f"    Mean difference: {mean_diff:.4f}")

# Create visualizations
plt.figure(figsize=(18, 6))

# Create scatter plot grid for all judge pairs
n_judges = len(judge_names)
plt.figure(figsize=(15, 15))
plot_idx = 1

# Create scatter plots for each pair of judges
for i in range(n_judges):
    for j in range(i+1, n_judges):
        judge1 = judge_names[i]
        judge2 = judge_names[j]
        plt.subplot(n_judges-1, n_judges-1, plot_idx)
        plt.scatter(df[judge1], df[judge2], alpha=0.5)
        plt.plot([0, 10], [0, 10], 'r--')  # Perfect agreement line
        
        # Set consistent axis ranges
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        
        plt.xlabel(f'{judge1} Scores')
        plt.ylabel(f'{judge2} Scores')
        plt.title(f'{judge1} vs {judge2}')
        plt.grid(True, alpha=0.3)
        plot_idx += 1

plt.tight_layout()
plt.savefig('judge_comparison_scatter.png')
print("\nScatter plot saved as 'judge_comparison_scatter.png'")

# Box plots of scores by judge
plt.figure(figsize=(10, 6))
judge_scores = pd.melt(df, id_vars=['question_id', 'model', 'turn'],
                     value_vars=judge_names,
                     var_name='Judge', value_name='Score')
sns.boxplot(x='Judge', y='Score', data=judge_scores)
plt.title('Distribution of Scores by Judge')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('judge_scores_boxplot.png')
print("Box plot saved as 'judge_scores_boxplot.png'")

# Histograms of differences
n_pairs = (n_judges * (n_judges - 1)) // 2
rows = (n_pairs + 2) // 3  # Calculate needed rows (rounded up)
plt.figure(figsize=(15, 5 * rows))
plot_idx = 1

# Generate histograms for all judge pairs
for i in range(n_judges):
    for j in range(i+1, n_judges):
        judge1 = judge_names[i]
        judge2 = judge_names[j]
        plt.subplot(rows, 3, plot_idx)
        plt.hist(df[judge2] - df[judge1], bins=20, alpha=0.7)
        plt.xlabel(f'Score Difference ({judge2} - {judge1})')
        plt.ylabel('Frequency')
        plt.title(f'{judge2} vs {judge1}')
        plt.grid(True, alpha=0.3)
        
        # Set consistent x-axis limits for all histograms
        all_diffs = [df[judge_names[j]] - df[judge_names[i]] for i in range(n_judges) for j in range(i+1, n_judges)]
        max_diff = max([diff.abs().max() for diff in all_diffs])
        plt.xlim(-max_diff, max_diff)
        
        plot_idx += 1
# Deleted content (replaced by the loop above)

plt.tight_layout()
plt.savefig('judge_differences_hist.png')
print("Histogram of differences saved as 'judge_differences_hist.png'")

# Create correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[judge_names].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.4f')
plt.title('Pearson Correlation Heatmap Between Judges')
plt.tight_layout()
plt.savefig('judge_correlation_heatmap.png')
print("Correlation heatmap saved as 'judge_correlation_heatmap.png'")

# Create summary table with average scores by model
model_scores = df.groupby('model')[judge_names].mean().reset_index()
model_scores_idx = model_scores.set_index('model')
baseline_judge = "GPT-5.1" if "GPT-5.1" in judge_names else judge_names[0]
model_scores_sorted = model_scores.sort_values(by=baseline_judge, ascending=False)
print("\n=== Average Scores by Model ===")
print(model_scores_sorted.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Line plot of judged models across judges (each line is a model, x-axis is judge)
plt.figure(figsize=(12, 6))
ordered_models = model_scores_sorted['model'].tolist()  # keep legend order consistent
judge_order = judge_names  # keep x-axis order consistent
for model in ordered_models:
    scores = model_scores_idx.loc[model, judge_order]
    plt.plot(judge_order, scores, marker='o', label=model, linewidth=2)
plt.title('Average Model Scores Across Judges')
plt.xlabel('Judge')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 10)
plt.grid(True, alpha=0.3)
plt.legend(title='Model', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
plt.tight_layout(rect=(0, 0, 0.82, 1))
plt.savefig('judge_scores_by_model.png')
print("Line plot saved as 'judge_scores_by_model.png'")

# Calculate overall average scores for each judge
avg_scores = {judge: df[judge].mean() for judge in judge_names}
print("\n=== Overall Average Scores ===")
for judge, score in avg_scores.items():
    print(f"{judge}: {score:.2f}")

# Ranking stability versus baseline judge
if len(model_scores) > 1:
    print(f"\n=== Ranking Stability (vs {baseline_judge}) ===")
    baseline_ranks = model_scores_idx[baseline_judge].rank(ascending=False, method='dense')
    baseline_ranks = baseline_ranks.astype(int)
    baseline_order = model_scores_idx[baseline_judge].sort_values(ascending=False)
    print(f"Baseline order ({baseline_judge}): {', '.join(baseline_order.index)}")

    for judge in judge_names:
        if judge == baseline_judge:
            continue

        judge_ranks = model_scores_idx[judge].rank(ascending=False, method='dense').astype(int)
        tau, p_tau = kendalltau(baseline_ranks, judge_ranks)
        rank_deltas = (judge_ranks - baseline_ranks).astype(int)
        movers = [
            f"{model} {'up' if delta < 0 else 'down'} {abs(delta)}"
            for model, delta in rank_deltas.items() if delta != 0
        ]

        pair_flips = []
        for a, b in combinations(model_scores_idx.index, 2):
            base_cmp = model_scores_idx.at[a, baseline_judge] - model_scores_idx.at[b, baseline_judge]
            judge_cmp = model_scores_idx.at[a, judge] - model_scores_idx.at[b, judge]
            if base_cmp == 0 or judge_cmp == 0:
                continue  # skip ties
            if base_cmp * judge_cmp < 0:
                pair_flips.append(f"{a} vs {b}")

        movers_text = ", ".join(movers) if movers else "no rank changes vs baseline"
        flips_preview = ", ".join(pair_flips[:8]) if pair_flips else "none"
        if len(pair_flips) > 8:
            flips_preview += f", ... (+{len(pair_flips) - 8} more)"

        print(f"{judge}: Kendall tau={tau:.3f} (p={p_tau:.3f}); {movers_text}; pairwise flips ({len(pair_flips)}): {flips_preview}")

# Save the detailed DataFrame to CSV for further analysis
df.to_csv('judge_comparison_details.csv', index=False)
print("\nDetailed comparison saved to 'judge_comparison_details.csv'")

# If you have turn data, create a visualization of scores by turn
if 'turn' in df.columns and len(df['turn'].unique()) > 1:
    plt.figure(figsize=(12, 6))
    turn_data = judge_scores.copy()
    sns.boxplot(x='turn', y='Score', hue='Judge', data=turn_data)
    plt.title('Scores by Turn and Judge')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scores_by_turn.png')
    print("Scores by turn visualization saved as 'scores_by_turn.png'")

    # Also output average scores by turn
    turn_avg = df.groupby('turn')[judge_names].mean().reset_index()
    print("\n=== Average Scores by Turn ===")
    print(turn_avg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
