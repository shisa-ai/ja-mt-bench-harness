import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
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

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine_type = 'circle'
                verts = unit_poly_verts(num_vars)
                # close off polygon by repeating first vertex
                verts.append(verts[0])
                path = Path(verts)
                spine = Spine(self, spine_type, path)
                # The spine is now at the origin. Move it to the center of the
                # plot.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                     + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def unit_poly_verts(num_vars):
    """Return vertices of polygon for radar chart"""
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # Rotate theta such that the first axis is at the top
    theta += np.pi/2
    verts = np.column_stack((np.cos(theta), np.sin(theta)))
    return verts.tolist()

def analyze_mt_bench_scores(
    bench_name: str,
    model_list: List[str],
    output_dir: Optional[str] = None,
    judge_name: Optional[str] = None,
    no_display: bool = False,
    show_only_summary: bool = True
) -> None:
    """Analyze MT-Bench scores and generate spider plots.
    
    Args:
        bench_name: Name of the benchmark
        model_list: List of models to analyze
        output_dir: Directory to save output figures
        judge_name: Specific judge to use (if None, will use all available judges)
    """
    # Find all available judgment files
    judgment_dir = f"data/{bench_name}/model_judgment"
    if not os.path.exists(judgment_dir):
        print(f"Judgment directory not found: {judgment_dir}")
        return
        
    # Find all judgment files
    judgment_files = []
    judge_names = []
    
    if judge_name:
        # Use only the specified judge
        judgment_file = f"{judgment_dir}/{judge_name}_single.jsonl"
        if os.path.exists(judgment_file):
            judgment_files.append(judgment_file)
            judge_names.append(judge_name)
        else:
            print(f"Judgment file not found for judge: {judge_name}")
            return
    else:
        # Find all available judgment files
        for filename in os.listdir(judgment_dir):
            if filename.endswith("_single.jsonl"):
                judgment_files.append(os.path.join(judgment_dir, filename))
                # Extract judge name from filename (remove _single.jsonl)
                judge_name = filename.replace("_single.jsonl", "")
                judge_names.append(judge_name)
                
    if not judgment_files:
        print(f"No judgment files found in {judgment_dir}")
        return
        
    print(f"Found {len(judgment_files)} judgment files: {judge_names}")

    # Load all judgments
    all_judgments = {}
    
    for idx, (judgment_file, judge_name) in enumerate(zip(judgment_files, judge_names)):
        try:
            with open(judgment_file, "r") as f:
                judgments = [json.loads(line) for line in f]
                all_judgments[judge_name] = judgments
                print(f"Loaded {len(judgments)} judgments from {judgment_file}")
        except Exception as e:
            print(f"Error loading {judgment_file}: {e}")
            continue

    # Load questions to get categories from the original source
    question_file = f"data/{bench_name}/question.jsonl"
    question_categories = {}
    
    try:
        with open(question_file, "r") as f:
            questions = [json.loads(line) for line in f]
            
        # Create a mapping from question_id to category from the actual questions
        for q in questions:
            question_id = q["question_id"]
            category = q.get("category", "unknown")
            question_categories[question_id] = category
    except (FileNotFoundError, json.JSONDecodeError):
        # Fall back to our predefined mapping if file doesn't exist or has issues
        print(f"Warning: Could not load {question_file}, using predefined category mapping")
        # Use judgments from the first judge to build the mapping
        first_judge = list(all_judgments.keys())[0]
        for judgment in all_judgments[first_judge]:
            question_id = judgment["question_id"]
            question_categories[question_id] = QUESTION_CATEGORY_MAP.get(question_id, "unknown")

    # Organize scores by judge, model, turn, and category
    scores_by_judge_model_turn = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scores_by_judge_model_turn_category = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    avg_scores_by_judge_model_turn = defaultdict(lambda: defaultdict(dict))
    avg_scores_by_judge_model_turn_category = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # Also keep track of which models are scored by each judge
    models_by_judge = defaultdict(set)
    all_available_models = set()
    
    # Process each judgment for each judge
    for judge_name, judgments in all_judgments.items():
        # Filter judgments for the requested models
        filtered_judgments = [j for j in judgments if j.get("model", j.get("model_id", "")) in model_list]
        
        # Use the appropriate key (model or model_id)
        key_name = "model" if "model" in judgments[0] else "model_id"
        
        for j in filtered_judgments:
            if "score" not in j:
                continue

            model = j[key_name]
            question_id = j["question_id"]
            # Use the actual turn from the data (usually 1 or 2)
            turn = j.get("turn", 1)  # Default to turn 1 if not specified
            score = j["score"]
            category = question_categories.get(question_id, "unknown")

            # Add score to appropriate collections
            scores_by_judge_model_turn[judge_name][model][turn].append(score)
            scores_by_judge_model_turn_category[judge_name][model][turn][category].append(score)
            
            # Track which models this judge has scored
            models_by_judge[judge_name].add(model)
            all_available_models.add(model)

    # Calculate average scores
    avg_scores_by_model_turn = {}
    avg_scores_by_model_turn_category = {}
    # Calculate averages for each judge
    for judge_name in all_judgments:
        for model in models_by_judge[judge_name]:
            for turn in scores_by_judge_model_turn[judge_name][model]:
                # Overall average for this turn
                turn_scores = scores_by_judge_model_turn[judge_name][model][turn]
                avg_scores_by_judge_model_turn[judge_name][model][turn] = np.mean(turn_scores) if turn_scores else 0

                # Category averages for this turn
                for cat in CATEGORIES:
                    scores = scores_by_judge_model_turn_category[judge_name][model][turn].get(cat, [])
                    avg_scores_by_judge_model_turn_category[judge_name][model][turn][cat] = np.mean(scores) if scores else 0

    # Print results
    print("====== MT-Bench Scores by Judge ======")
    
    # For each model
    for model in all_available_models:
        if model not in model_list:
            continue
            
        print(f"\n{'='*40}\nModel: {model}\n{'='*40}")
        
        # For each judge that scored this model
        for judge_name in judge_names:
            if model not in models_by_judge[judge_name]:
                print(f"\n{judge_name}: No scores available")
                continue
                
            print(f"\n{judge_name}:")
            
            # Safely get turn averages with default of 0 if turn doesn't exist
            turn1_avg = avg_scores_by_judge_model_turn[judge_name][model].get(1, 0)
            turn2_avg = avg_scores_by_judge_model_turn[judge_name][model].get(2, 0)
            overall_avg = (turn1_avg + turn2_avg) / 2 if turn2_avg > 0 else turn1_avg

            print(f"  Turn 1 Average: {turn1_avg:.2f}")
            if turn2_avg > 0:
                print(f"  Turn 2 Average: {turn2_avg:.2f}")
                print(f"  Overall Average: {overall_avg:.2f}")
            else:
                print("  Turn 2: Not evaluated")

            # Only print category scores if the turn exists
            if 1 in avg_scores_by_judge_model_turn_category[judge_name][model]:
                print("\n  Category Scores (Turn 1):")
                for cat in CATEGORIES:
                    cat_score = avg_scores_by_judge_model_turn_category[judge_name][model][1].get(cat, float('nan'))
                    if np.isnan(cat_score):
                        print(f"    {cat}: N/A")
                    else:
                        print(f"    {cat}: {cat_score:.2f}")

            if turn2_avg > 0 and 2 in avg_scores_by_judge_model_turn_category[judge_name][model]:
                print("\n  Category Scores (Turn 2):")
                for cat in CATEGORIES:
                    cat_score = avg_scores_by_judge_model_turn_category[judge_name][model][2].get(cat, float('nan'))
                    if np.isnan(cat_score):
                        print(f"    {cat}: N/A")
                    else:
                        print(f"    {cat}: {cat_score:.2f}")

    # Create average scores across turns for each judge
    avg_scores_by_judge_model_category = defaultdict(lambda: defaultdict(dict))
    
    # Keep track of figures to display
    figures_to_display = []
    
    # For summary plot
    all_model_category_scores = {}
    all_model_overall_scores = {}
    
    # Calculate average scores across turns
    for judge_name in judge_names:
        for model in models_by_judge[judge_name]:
            for cat in CATEGORIES:
                scores = []
                # Gather scores from all turns
                for turn in [1, 2]:
                    if turn in avg_scores_by_judge_model_turn_category[judge_name][model]:
                        score = avg_scores_by_judge_model_turn_category[judge_name][model][turn].get(cat, float('nan'))
                        if not np.isnan(score):
                            scores.append(score)
                
                # Calculate the average if we have scores
                if scores:
                    score = np.mean(scores)
                    avg_scores_by_judge_model_category[judge_name][model][cat] = score
                    
                    # Store for summary plot
                    if model not in all_model_category_scores:
                        all_model_category_scores[model] = {}
                    if cat not in all_model_category_scores[model]:
                        all_model_category_scores[model][cat] = []
                    all_model_category_scores[model][cat].append(score)
                else:
                    avg_scores_by_judge_model_category[judge_name][model][cat] = float('nan')

    # Create radar charts for each judge (turns combined)
    for judge_name in judge_names:
        # Get models that have scores from this judge
        models_with_scores = [m for m in model_list if m in models_by_judge[judge_name]]
        
        if not models_with_scores:
            continue

        # Set up the radar chart
        N = len(CATEGORIES)
        theta = radar_factory(N, frame='polygon')

        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='radar'))

        # Plot each model
        for i, model in enumerate(models_with_scores):
            color = plt.cm.tab10(i % 10)
            cat_scores = []

            for cat in CATEGORIES:
                score = avg_scores_by_judge_model_category[judge_name][model].get(cat, float('nan'))
                cat_scores.append(score if not np.isnan(score) else 0)

            ax.plot(theta, cat_scores, color=color, label=model)
            ax.fill(theta, cat_scores, facecolor=color, alpha=0.25)

        # Set the labels and title
        ax.set_varlabels(CATEGORIES)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_title(f"{judge_name} - MT-Bench Average Category Scores")

        # Add a legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # Save the figure if output_dir is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"{judge_name}_radar_average.png"), dpi=300, bbox_inches='tight')

        plt.tight_layout()
        
        # Only store this figure if we want to show all plots
        if not no_display and not show_only_summary:
            figures_to_display.append(fig)
        else:
            plt.close()
            
        # Create a comparison chart across judges for each model (using averages)
        for model in model_list:
            if model not in all_available_models:
                continue
                
            # Get all judges that scored this model
            judges_with_model = [j for j in judge_names if model in models_by_judge[j]]
            
            if len(judges_with_model) <= 1:
                continue  # Skip if only one judge scored this model
                
            # Set up the radar chart
            N = len(CATEGORIES)
            theta = radar_factory(N, frame='polygon')

            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='radar'))

            # Plot each judge's scores
            for i, judge in enumerate(judges_with_model):
                color = plt.cm.Set2(i % 8)
                cat_scores = []

                for cat in CATEGORIES:
                    score = avg_scores_by_judge_model_category[judge][model].get(cat, float('nan'))
                    cat_scores.append(score if not np.isnan(score) else 0)

                ax.plot(theta, cat_scores, color=color, label=judge)
                ax.fill(theta, cat_scores, facecolor=color, alpha=0.25)

            # Set the labels and title
            ax.set_varlabels(CATEGORIES)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_title(f"{model} - Judge Comparison (Average Scores)")

            # Add a legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            # Save the figure if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                # Create a safe filename by replacing slashes with underscores
                safe_model_name = model.replace('/', '_')
                plt.savefig(os.path.join(output_dir, f"{safe_model_name}_judge_comparison_average.png"), 
                            dpi=300, bbox_inches='tight')

            plt.tight_layout()
            
            # Store the figure if we want to display it later, otherwise close it
            if no_display or show_only_summary:
                plt.close()
            else:
                figures_to_display.append(fig)

    # Create visualizations based on the number of judges
    if not no_display:
        # Create a list of models that have scores
        models_to_plot = [m for m in model_list if m in all_model_category_scores]
        
        if not models_to_plot:
            print("No models with scores to plot")
            return
            
        # Create radar factory
        N = len(CATEGORIES)
        theta = radar_factory(N, frame='polygon')
        
        # Check if we have a single judge or multiple judges
        if len(judge_names) == 1 and judge_name is not None:
            # SINGLE JUDGE MODE: Show all models on one plot
            single_judge = judge_names[0]
            
            # Create a single plot
            plt.figure(figsize=(12, 12))
            ax = plt.subplot(111, projection='radar')
            
            # Plot each model
            for i, model in enumerate(models_to_plot):
                if model not in models_by_judge[single_judge]:
                    continue
                    
                # Use a nice color palette
                color = plt.cm.tab10(i % 10)
                cat_scores = []
                
                for cat in CATEGORIES:
                    score = avg_scores_by_judge_model_category[single_judge][model].get(cat, float('nan'))
                    cat_scores.append(score if not np.isnan(score) else 0)
                
                ax.plot(theta, cat_scores, color=color, label=model, linewidth=2)
                ax.fill(theta, cat_scores, facecolor=color, alpha=0.25)
            
            # Set the labels and title
            ax.set_varlabels(CATEGORIES)
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_title(f"{single_judge} Judge - Model Comparison")
            
            # Add a legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Save the figure if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"{single_judge}_model_comparison.png"),
                          dpi=300, bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            
        else:
            # MULTIPLE JUDGES MODE: Show each model with all its judges
            # Determine grid layout for subplots - make it square
            num_models = len(models_to_plot)
            cols = int(np.ceil(np.sqrt(num_models)))
            rows = int(np.ceil(num_models / cols))
            
            # Create figure with subplots - square form factor
            fig_size = max(10, 5 * cols)  # Base size on number of plots, but with a minimum
            fig, axes = plt.subplots(rows, cols, figsize=(fig_size, fig_size), 
                                   subplot_kw=dict(projection='radar'))
            
            # Convert to iterable if only one subplot
            if num_models == 1:
                axes = np.array([axes])
            
            # Flatten axes array for easier indexing
            axes = np.array(axes).flatten()
            
            # Plot each model with all its judges
            for i, model in enumerate(models_to_plot):
                ax = axes[i]
                
                # Find all judges that scored this model
                judges_with_model = [j for j in judge_names if model in models_by_judge[j]]
                
                # Plot each judge's scores for this model with improved color scheme
                for j, judge in enumerate(judges_with_model):
                    # Use a better color scheme - Set2 has nicer colors than tab10
                    color = plt.cm.Set2(j % 8)
                    cat_scores = []
                    
                    for cat in CATEGORIES:
                        score = avg_scores_by_judge_model_category[judge][model].get(cat, float('nan'))
                        cat_scores.append(score if not np.isnan(score) else 0)
                    
                    ax.plot(theta, cat_scores, color=color, label=judge, linewidth=2)
                    ax.fill(theta, cat_scores, facecolor=color, alpha=0.25)  # Increased opacity
                
                # Set the labels and title for this subplot
                ax.set_varlabels(CATEGORIES)
                ax.set_ylim(0, 10)
                ax.set_yticks([2, 4, 6, 8, 10])
                ax.set_title(f"{model} - Judge Comparison")
                
                # Add a legend - only for plots with data
                if judges_with_model:
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize='small')
            
            # Hide any unused subplots
            for j in range(num_models, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout(pad=3.0)
            
            # Save the figure if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"all_models_judge_comparison.png"), 
                          dpi=300, bbox_inches='tight')
            
            # Show the plot
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze MT-Bench scores and generate spider plots")

    parser.add_argument("--bench-name", type=str, default="ja_mt_bench",
                        help="Benchmark name (default: ja_mt_bench)")
    parser.add_argument("--model-list", type=str, nargs="+", required=True,
                        help="List of model IDs to analyze")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save output figures (optional)")
    parser.add_argument("--judge", type=str, default=None,
                        help="Specific judge to use (default: use all available judges)")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't display any plots interactively")
    parser.add_argument("--show-only-summary", action="store_true", default=True,
                        help="Only show the summary plot (default: True)")

    args = parser.parse_args()
    
    analyze_mt_bench_scores(
        bench_name=args.bench_name,
        model_list=args.model_list,
        output_dir=args.output_dir,
        judge_name=args.judge,
        no_display=args.no_display,
        show_only_summary=args.show_only_summary
    )

if __name__ == "__main__":
    main()
