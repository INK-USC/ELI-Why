#!/usr/bin/env python
import argparse
import os
import pandas as pd

from helper import compute_tesdiff_heatmap, plot_heatmap

def mode_compute(model: str):
    """
    Compute TESDiff heatmaps for various dataset splits using data read from a JSONL file.
    TESDiff heatmaps are computed on the specified role columns.
    """
    # Read in the full dataset from a JSONL file.
    data_full = pd.read_json("../ELI_Why_with_rationales.jsonl", lines=True)
    
    # Create subsets based on the "Domain" column.
    full_df = data_full.copy()
    stem_df = full_df[full_df['Domain'] == 'STEM'].copy()
    non_stem_df = full_df[full_df['Domain'] != 'STEM'].copy()
    
    # Define the roles/columns to consider for the TESDiff calculation.
    roles = [f"{model} Graduate School",
             f"{model} High School",
             f"{model} Elementary School",
             f"{model} Default",
             "Web Retrieved"]
    
    # Ensure all expected columns exist in the dataset.
    for col in roles:
        if col not in full_df.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")
    
    # Create the output directory if it doesn't exist.
    os.makedirs("similarity_scores", exist_ok=True)
    
    # Process each subset: full, STEM, and non-STEM.
    datasets = {"full_df": full_df, "stem_df": stem_df, "non_stem_df": non_stem_df}
    for df_name, df in datasets.items():
        print(f"Computing TESDiff heatmap for {df_name}...")
        tesdiff_score_df = compute_tesdiff_heatmap(df, roles)
        save_path = f"similarity_scores/{model}_{df_name}_tesdiff_score.png"
        plot_heatmap(tesdiff_score_df, f"TESDiff Score Heatmap for {df_name}", 
                     save_path=save_path, show=False, cmap="cividis")
        tesdiff_score_df.to_csv(save_path[:-4] + ".csv", index=True)
        print(f"Saved TESDiff heatmap for {df_name} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Compute TESDiff heatmaps")
    parser.add_argument("--model", type=str,
                        help="Choose the model: GPT4o or Llama3.2")
    parser.add_argument("--all", action="store_true",
                        help="Compute TESDiff heatmaps for all models")
    
    args = parser.parse_args()
    
    if args.all:
        if args.model is not None:
            raise ValueError("Cannot specify both --all and --model.")
        for m in ["GPT4o", "Llama3.2"]:
            print(f"\nComputing TESDiff heatmaps for {m}...")
            mode_compute(m)
    else:
        if args.model not in ["GPT4o", "Llama3.2"]:
            raise ValueError("Invalid model specified. Choose either 'GPT4o' or 'Llama3.2'.")
        mode_compute(args.model)

if __name__ == "__main__":
    main()
