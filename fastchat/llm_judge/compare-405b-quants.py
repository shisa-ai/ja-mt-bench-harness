import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Step 1: Load Data from TXT into DataFrame ---
def parse_mtbench_table(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    models = []
    current_model = None
    current_scores = {}
    parsing_scores = False
    score_regex = re.compile(r'^\s*([A-Za-z0-9 .\-_/()]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$')
    for line in lines:
        line = line.rstrip()
        model_header = re.match(r"^([a-zA-Z0-9\-_/\.]+)$", line)
        if model_header and not line.startswith("Category"):
            if current_model and current_scores:
                models.append({"Model": current_model, **current_scores})
            current_model = line
            current_scores = {}
            parsing_scores = False
            continue
        if line.strip().startswith("Category") and "GPT-4.1" in line:
            parsing_scores = True
            continue
        if line.strip() == "" or line.startswith("="):
            parsing_scores = False
            continue
        if parsing_scores:
            match = score_regex.match(line)
            if match:
                field = match.group(1).strip()
                gpt_41 = match.group(4)
                current_scores[field] = float(gpt_41)
    if current_model and current_scores:
        models.append({"Model": current_model, **current_scores})
    return pd.DataFrame(models)

# --- Step 2: Define categories and model selections ---
categories = [
    "writing (Average)", "roleplay (Average)", "reasoning (Average)", "math (Average)",
    "coding (Average)", "extraction (Average)", "stem (Average)", "humanities (Average)"
]
category_labels = [
    "Writing", "Roleplay", "Reasoning", "Math", "Coding", "Extraction", "STEM", "Humanities"
]

main_models_labels = [
    ("shisa-v2-llama3.1-405b", "Full FP16"),
    ("shisa-v2-llama3.1-405b-W8A8-INT8", "W8A8-INT8"),
    ("shisa-v2-llama3.1-405b-FP8-Dynamic", "FP8-Dynamic"),
    ("shisa-v2-llama3.1-405b-Q8_0", "Q8_0"),
    ("shisa-v2-llama3.1-405b-Q4_K_M", "Q4_K_M"),
    ("shisa-v2-llama3.1-405b-IQ4_XS", "IQ4_XS"),
    ("shisa-v2-llama3.1-405b-IQ3_M", "IQ3_M"),
    ("shisa-v2-llama3.1-405b-IQ3_XS", "IQ3_XS"),
    ("shisa-v2-llama3.1-405b-IQ2_XXS", "IQ2_XXS"),
]
legend_with_size = {
    "Full FP16": "Full FP16 (810 GB)",
    "FP8-Dynamic": "FP8-Dynamic (405 GB)",
    "W8A8-INT8": "W8A8-INT8 (405 GB)",
    "Q8_0": "Q8_0 (405 GB)",
    "Q4_K_M": "Q4_K_M (227 GB)",
    "IQ4_XS": "IQ4_XS (202 GB)",
    "IQ3_M": "IQ3_M (170 GB)",
    "IQ3_XS": "IQ3_XS (155 GB)",
    "IQ2_XXS": "IQ2_XXS (100 GB)",
    "70B FP16": "70B FP16 (140 GB)",
}
custom_colors = {
    "Full FP16": "#ff4bb2",         # bright pink
    "W8A8-INT8": "#8e24aa",         # purple
    "FP8-Dynamic": "#2196f3",       # blue
    "Q8_0": "#007f00",              # dark green
    "Q4_K_M": "#00e5e5",            # cyan
    "IQ4_XS": "#aeea00",            # yellow-green
    "IQ3_M": "#ff9800",             # orange
    "IQ3_XS": "#ffc300",            # deep gold
    "IQ2_XXS": "#ff1744",           # bright red
}
baseline_70b_model = "shisa-ai/shisa-v2-llama3.3-70b"
baseline_70b_label = legend_with_size["70B FP16"]
baseline_70b_color = "lightgrey"

# --- Step 3: Plot full 405B + baseline graph ---
def plot_full_family(df, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))  # width=14 inches, height=8 inches

    handles_labels = []
    # Draw all but Full FP16 and 70B baseline
    for model, label in main_models_labels:
        if label == "Full FP16":
            continue
        color = custom_colors[label]
        display_label = legend_with_size[label]
        row = df[df["Model"].str.replace("shisa-ai/", "") == model]
        if row.empty:
            row = df[df["Model"] == model]
        if not row.empty:
            scores = row.iloc[0][categories].values
            line, = plt.plot(category_labels, scores, marker="o", label=display_label, linewidth=1.0, color=color, zorder=2)
            handles_labels.append((line, display_label))
    # Draw 70B baseline above quants, but before FP16
    baseline_row = df[df["Model"] == baseline_70b_model]
    if not baseline_row.empty:
        baseline_scores = baseline_row.iloc[0][categories].values
        line, = plt.plot(category_labels, baseline_scores, label=baseline_70b_label, color=baseline_70b_color, linestyle="--", linewidth=2.5, zorder=3)
        handles_labels.append((line, baseline_70b_label))
    # Full FP16 last, thickest, on top
    model_fp16 = "shisa-v2-llama3.1-405b"
    row_fp16 = df[df["Model"].str.replace("shisa-ai/", "") == model_fp16]
    if row_fp16.empty:
        row_fp16 = df[df["Model"] == model_fp16]
    if not row_fp16.empty:
        scores_fp16 = row_fp16.iloc[0][categories].values
        line_fp16, = plt.plot(category_labels, scores_fp16, marker="o", label=legend_with_size["Full FP16"], linewidth=2.5, color=custom_colors["Full FP16"], zorder=4)
        handles_labels = [(line_fp16, legend_with_size["Full FP16"])] + handles_labels
    handles, labels = zip(*handles_labels)

    leg = plt.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.01), fontsize="medium")
    leg.get_frame().set_visible(False)

    ax.yaxis.grid(True, which='major', color='#e5e5e5', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_yticks(range(0, 11, 1))  # Show grid at every 1.0

    plt.ylabel("Score")
    plt.title("Shisa V2 405B JA MT-Bench (Judge: GPT-4.1)")
    plt.ylim(5, 10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# --- Step 4: Plot FP16 + 8-bit quants only ---
def plot_fp16_8bit(df, output_path):
    models_8bit = [
        ("shisa-v2-llama3.1-405b", "Full FP16"),
        ("shisa-v2-llama3.1-405b-FP8-Dynamic", "FP8-Dynamic"),
        ("shisa-v2-llama3.1-405b-Q8_0", "Q8_0"),
        ("shisa-v2-llama3.1-405b-W8A8-INT8", "W8A8-INT8")
    ]
    plot_colors = [custom_colors[label] for _, label in models_8bit]
    plot_labels = [legend_with_size[label] for _, label in models_8bit]
    for idx, ((model, label), color, display_label) in enumerate(zip(models_8bit, plot_colors, plot_labels)):
        row = df[df["Model"] == model]
        if row.empty:
            row = df[df["Model"].str.replace("shisa-ai/", "") == model]
        if not row.empty:
            scores = row.iloc[0][categories].values
            plt.plot(category_labels, scores, marker="o", label=display_label, linewidth=2.2 if label=="Full FP16" else 1.3, color=color)
    plt.ylabel("Score")
    plt.title("Shisa V2 405B FP16 & 8-bit Quants\nJA MT-Bench (Judge: GPT-4.1)")
    plt.ylim(0, 10)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="medium", title="Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Edit filename as needed
    txt_file = "judge_comparison_table.txt"
    df = parse_mtbench_table(txt_file)
    df.to_csv("shisa_405b_mtbench_scores.csv", index=False)
    plot_full_family(df, "shisa_405b_mtbench_family.png")
    plot_fp16_8bit(df, "shisa_405b_fp16_8bit.png")
