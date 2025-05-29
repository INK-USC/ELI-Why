import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nltk

from pandarallel import pandarallel
from tqdm import tqdm
from helper import *

# -- 0. Initialization --

# Download necessary nltk data
nltk.download('wordnet')
nltk.download('punkt_tab')

tqdm.pandas()
pandarallel.initialize()

# Read in the full dataset from a JSONL file
# data_full = pd.read_json("../ELI_Why_with_rationales.jsonl", lines=True)
data_full = pd.read_json("integrated_ELI_Why_with_rationales.jsonl", lines=True)

# Create indices for STEM/non-STEM filtering
full_idx = data_full.index
stem_idx = data_full[data_full['Domain'] == 'STEM'].index
non_stem_idx = data_full[data_full['Domain'] != 'STEM'].index

# -------------------------
# Main function starts here
# -------------------------
def main(model_name: str, set_type: str, flesch_only: bool = False):
    # -- 1. Load target word list and lemmatize--
    # Use the full dataset
    data = data_full.copy()

    # Dynamically update roles based on model name.
    # The roles correspond to:
    #   [Graduate School, High School, Elementary School, Default, Web Retrieved]
    roles = [f"{model_name} Graduate School",
             f"{model_name} High School",
             f"{model_name} Elementary School",
             f"{model_name} Default",
             "Web Retrieved"]

    # Select filtered data and run_name based on set type.
    if set_type == "full":
        filtered_data = data[roles]
        run_name = f"{model_name}_full_set"
    elif set_type == "stem":
        filtered_data = data.loc[stem_idx, roles]
        run_name = f"{model_name}_stem_set"
    elif set_type == "nonstem":
        filtered_data = data.loc[non_stem_idx, roles]
        run_name = f"{model_name}_nonstem_set"
    else:
        raise ValueError("Set must be one of 'full', 'stem', or 'nonstem'.")

    print('#' * 100)
    print('Current run:', run_name)
    print('#' * 100)

    # Define the three key roles for comparisons
    grad_role = roles[0]     # Graduate School (formerly PhD)
    high_role = roles[1]     # High School
    elem_role = roles[2]     # Elementary School

    # -- 2. Compute readability metrics --
    flesch_scores = filtered_data.apply(lambda col: col.apply(lambda x: compute_readability(x, flesch_reading_ease)))
    linsear_scores = filtered_data.apply(lambda col: col.apply(lambda x: compute_readability(x, linsear_write_formula)))
    dale_chall_scores = filtered_data.apply(lambda col: col.apply(lambda x: compute_readability(x, dale_chall_readability_score)))

    sentences_df = filtered_data.apply(lambda col: col.apply(count_sentences))
    words_df = filtered_data.apply(lambda col: col.apply(count_words))

    avg_words_per_sentence_df = pd.DataFrame()
    reading_time_df = pd.DataFrame()
    te_score_df = pd.DataFrame()

    for role in roles:
        avg_words_per_sentence_df[role] = words_df[role] / sentences_df[role]
        reading_time_df[role] = filtered_data[role].apply(avg_reading_time)
        te_score_df[role] = filtered_data[role].parallel_apply(te_score)

    # Filter out instances based on quality criteria
    filter_condition = (sentences_df > 20) | (avg_words_per_sentence_df > 40) | (reading_time_df > 35) | \
                       (te_score_df > 1) | (flesch_scores < -40) | (linsear_scores > 30) | (dale_chall_scores > 15)

    filtered_data = filtered_data.mask(filter_condition)
    flesch_scores = flesch_scores.mask(filter_condition)
    linsear_scores = linsear_scores.mask(filter_condition)
    dale_chall_scores = dale_chall_scores.mask(filter_condition)
    sentences_df = sentences_df.mask(filter_condition)
    words_df = words_df.mask(filter_condition)
    avg_words_per_sentence_df = avg_words_per_sentence_df.mask(filter_condition)
    reading_time_df = reading_time_df.mask(filter_condition)
    te_score_df = te_score_df.mask(filter_condition)

    # If the shortcut flag is set, only plot Flesch Reading Ease and exit.
    if flesch_only:
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["tab:orange", "tab:green", "tab:blue", "#505050", "goldenrod"]
        for role, c in zip(roles, colors):
            subset = flesch_scores[role].dropna()
            sns.kdeplot(subset, ax=ax, fill=True, color=c,
                        label=role, bw_adjust=1.5, alpha=0.1, linewidth=1.5)
        ax.set_title("Flesch Reading Ease")
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(loc='upper right')
        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plot_path = os.path.join("plots", "Qwen2.5_flesch_reading_ease_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        return

    # --- 3. Classify explanations using Llama --

    # --- 3a. Generate classification for each row & each role ---
    classification_csv = f"{model_name}_mech_teleo_classification.csv"
    no_class_plot = False
    if not os.path.exists(classification_csv):
        no_class_plot = True
    else:
        classification_df = pd.read_csv(classification_csv)

    if not no_class_plot:
        classification_df = classification_df.mask(filter_condition)
        if set_type == "nonstem":
            classification_df = classification_df.loc[non_stem_idx]
        elif set_type == "stem":
            classification_df = classification_df.loc[stem_idx]

        # Calculate mechanistic and teleological proportions for the three key roles
        mech_count = {}
        tele_count = {}
        total_count = {}
        for role in [grad_role, high_role, elem_role]:
            role_values = classification_df[role].dropna()
            mech_count[role] = (role_values == "Mechanistic").sum()
            tele_count[role] = (role_values == "Teleological").sum()
            total_count[role] = len(role_values)

        print("\n=== Overall Classification Percentages ===")
        for role in [grad_role, high_role, elem_role]:
            if total_count[role] == 0:
                print(f"{role}: No data")
            else:
                mech_pct = mech_count[role] / total_count[role] * 100
                tele_pct = tele_count[role] / total_count[role] * 100
                print(f"{role}: Mechanistic = {mech_pct:.2f}%, Teleological = {tele_pct:.2f}%")

        # -- 4. Statistical test: Pairwise one-tailed proportion tests --
        print("\n=== Pairwise One-Tailed Proportion Tests (Mechanistic) ===")
        p_grad_high = one_tailed_proportion_test(
            mech_count[grad_role], total_count[grad_role],
            mech_count[high_role], total_count[high_role],
            alternative='larger'
        )
        print(f"{grad_role} vs {high_role}: p={p_grad_high:.4f}")

        p_high_elem = one_tailed_proportion_test(
            mech_count[high_role], total_count[high_role],
            mech_count[elem_role], total_count[elem_role],
            alternative='larger'
        )
        print(f"{high_role} vs {elem_role}: p={p_high_elem:.4f}")

        p_grad_elem = one_tailed_proportion_test(
            mech_count[grad_role], total_count[grad_role],
            mech_count[elem_role], total_count[elem_role],
            alternative='larger'
        )
        print(f"{grad_role} vs {elem_role}: p={p_grad_elem:.4f}")

    # -- 5. Print and store averages and standard deviations for each metric/role --
    metrics_dataframes = {
        "# Sentences": sentences_df,
        "Avg Words/Sentence": avg_words_per_sentence_df,
        "Avg Reading Time (s)": reading_time_df,
        "TE Score": te_score_df,
        "Flesch Reading Ease": flesch_scores,
        "Linsear Write Formula": linsear_scores,
        "Dale-Chall Readability": dale_chall_scores
    }

    analysis_report = []
    analysis_report.append("=== Averages and Standard Deviations ===")
    for metric_name, df in metrics_dataframes.items():
        analysis_report.append(f"\n--- {metric_name} ---")
        print(f"\n--- {metric_name} ---")
        for role in roles:
            mean_val = df[role].mean()
            std_val = df[role].std()
            line = f"{role}: mean={mean_val:.2f}, std={std_val:.2f}"
            analysis_report.append(line)
            print(line)

    # Write the analysis report to a file in the "results" directory
    os.makedirs("results", exist_ok=True)
    analysis_file_path = os.path.join("results", f"{run_name}_analysis.txt")
    with open(analysis_file_path, "w") as f:
        f.write("\n".join(analysis_report))
    print(f"\nAnalysis report saved to {analysis_file_path}")

    # -- 6. Plot 4×2 subplots --
    fig, axs = plt.subplots(4, 2, figsize=(18, 15))
    axs = axs.flatten()

    plots_info = [
        (sentences_df, "# Sentences"),
        (avg_words_per_sentence_df, "Average Words/Sentence"),
        (reading_time_df, "Average Reading Time (s)"),
        (te_score_df, "TE Score"),
        (flesch_scores, "Flesch Reading Ease"),
        (linsear_scores, "Linsear Write Formula"),
        (dale_chall_scores, "Dale-Chall Readability")
    ]

    colors = ["tab:orange", "tab:green", "tab:blue", "#505050", "goldenrod"]

    for i, (df, title) in enumerate(plots_info):
        if i == 7 and no_class_plot:
            continue
        for role, c in zip(roles, colors):
            subset = df[role].dropna()
            sns.kdeplot(subset, ax=axs[i], fill=True, color=c,
                        label=role, bw_adjust=1.5, alpha=0.1, linewidth=1.5)
        axs[i].set_title(title)
        axs[i].set_xlabel("Score")
        axs[i].set_ylabel("Density" if i % 2 == 0 else "")
        axs[i].legend(loc='upper right')

    if not no_class_plot:
        mech_ratio, mech_low, mech_up = compute_prop_and_ci(classification_df, [grad_role, high_role, elem_role], "Mechanistic")
        mech_err = np.array([
            [mech_ratio[i] - mech_low[i] for i in range(len(mech_ratio))],
            [mech_up[i] - mech_ratio[i] for i in range(len(mech_ratio))]
        ])
        axs[7].bar([grad_role, high_role, elem_role], mech_ratio, yerr=mech_err, color=colors[:3], capsize=5)
        axs[7].set_title("% Mechanistic per Role")
        axs[7].set_ylabel("% Mechanistic")
        axs[7].set_xticks([0, 1, 2])
        axs[7].set_xticklabels([grad_role, high_role, elem_role], rotation=20)

    plt.tight_layout()
    # Save the figure before showing
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{run_name}_readability_plots.png")
    print(f"Plots saved to plots/{run_name}_readability_plots.png")

    # -- 7. Run KS tests for all metrics --
    for metric_name, df in metrics_dataframes.items():
        run_ks_tests(df, metric_name, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis with model name and set type")
    parser.add_argument("--model", type=str, choices=["GPT4", "Llama3.2", "Qwen2.5", "Gemma3", "R1_Distilled_Llama"],
                        help="Choose the model: GPT4 or Llama3.2 or Qwen2.5 or Gemma3 or R1_Distilled_Llama")
    parser.add_argument("--set", type=str, choices=["full", "stem", "nonstem"],
                        help="Choose the set: full, stem, or nonstem")
    parser.add_argument("--all", action="store_true",
                        help="Run analysis for all combinations of model and set types")
    parser.add_argument("--flesch_only", action="store_true",
                        help="Shortcut: run only Qwen2.5's Flesch Reading Ease plot")
    args = parser.parse_args()
    
    if args.flesch_only:
        if args.all or args.model or args.set:
            print("Warning: --flesch_only ignores --all, --model and --set arguments. Running only Qwen2.5 full set Flesch Reading Ease plot.")
        main("Qwen2.5", "full", flesch_only=True)
    elif args.all:
        models = ["GPT4", "Llama3.2", "Qwen2.5", "Gemma3", "R1_Distilled_Llama"]
        set_types = ["full", "stem", "nonstem"]
        for model in models:
            for set_type in set_types:
                print(f"\nRunning analysis for model: {model}, set: {set_type}")
                main(model, set_type)
    else:
        if not args.model or not args.set:
            parser.error("Either specify --all or provide both --model and --set arguments")
        main(args.model, args.set)
