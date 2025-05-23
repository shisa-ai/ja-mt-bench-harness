import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# File paths for the three judgment files
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
print("\n=== Average Scores by Model ===")
print(model_scores.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Calculate overall average scores for each judge
avg_scores = {judge: df[judge].mean() for judge in judge_names}
print("\n=== Overall Average Scores ===")
for judge, score in avg_scores.items():
    print(f"{judge}: {score:.2f}")

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
