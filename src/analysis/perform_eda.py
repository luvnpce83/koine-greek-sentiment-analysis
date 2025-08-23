import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_nt_valence(row):
    """
    Calculates the valence score for the NT dataset based on annotator votes.
    Formula: (Positive_counts - Negative_counts) / Total_valid_votes
    """
    label_cols = [f'Label {i}' for i in range(1, 9)]
    votes = row[label_cols]

    # It's crucial to handle non-string values that might appear
    positive_votes = sum(1 for vote in votes if isinstance(vote, str) and 'Positive' in vote)
    negative_votes = sum(1 for vote in votes if isinstance(vote, str) and 'Negative' in vote)

    # The denominator should be the total number of annotators who provided a valid vote.
    # We assume all 8 labels are present for each row.
    total_votes = 8

    if total_votes == 0:
        return 0

    return (positive_votes - negative_votes) / total_votes

def analyze_datasets():
    """
    Performs EDA on NT and Homeric datasets, calculates valence scores,
    and visualizes their distributions.
    """
    print("--- Starting Exploratory Data Analysis (EDA) ---")

    # Create output directory for plots
    output_dir = "eda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Process NT Dataset ---
    print("\n--- Processing New Testament (NT) Dataset ---")
    try:
        nt_df = pd.read_csv('data/NT_annotation_raw.csv')

        # Drop rows where 'Text' is NaN, as they are not usable
        nt_df.dropna(subset=['Text'], inplace=True)

        nt_df['valence_score'] = nt_df.apply(calculate_nt_valence, axis=1)

        # Rename column for consistency and save the processed NT data
        nt_df.rename(columns={'Text': 'text'}, inplace=True)
        nt_output_path = 'data/processed_nt_valence.csv'
        nt_df[['text', 'valence_score']].to_csv(nt_output_path, index=False)
        print(f"Processed NT data saved to {nt_output_path}")

        # Print statistics for NT scores
        print("\nNT Valence Score Statistics:")
        print(nt_df['valence_score'].describe())

        # Plot and save histogram for NT scores
        plt.figure(figsize=(10, 6))
        nt_df['valence_score'].hist(bins=20, alpha=0.7, label='NT Valence Scores')
        plt.title('Distribution of Valence Scores (New Testament)')
        plt.xlabel('Valence Score')
        plt.ylabel('Frequency')
        plt.legend()
        nt_hist_path = os.path.join(output_dir, 'nt_valence_distribution.png')
        plt.savefig(nt_hist_path)
        plt.close()
        print(f"NT distribution plot saved to {nt_hist_path}")

    except FileNotFoundError:
        print("❌ Error: 'data/NT_annotation_raw.csv' not found.")
        return

    # --- 2. Process Homeric Dataset ---
    print("\n--- Processing Homeric Dataset ---")
    try:
        homeros_df = pd.read_csv('data/Homeros_annotation.csv')

        # Assuming columns are 'text', 'positive_probability', 'negative_probability'
        # Let's verify column names first
        print("Homeric data columns:", homeros_df.columns.tolist())

        # The actual column names might be different, let's be robust
        # Common patterns are 'positive', 'pos_prob', etc.
        pos_col = next((col for col in homeros_df.columns if 'pos' in col.lower()), None)
        neg_col = next((col for col in homeros_df.columns if 'neg' in col.lower()), None)
        text_col = next((col for col in homeros_df.columns if 'text' in col.lower()), None)

        if not all([pos_col, neg_col, text_col]):
             print("❌ Error: Could not find required columns (text, positive, negative) in Homeric data.")
             return

        homeros_df['valence_score'] = homeros_df[pos_col] - homeros_df[neg_col]

        # Save the processed Homeric data
        homeros_output_path = 'data/processed_homeros_valence.csv'
        homeros_df[[text_col, 'valence_score']].to_csv(homeros_output_path, index=False)
        print(f"Processed Homeric data saved to {homeros_output_path}")

        # Print statistics for Homeric scores
        print("\nHomeric Valence Score Statistics:")
        print(homeros_df['valence_score'].describe())

        # Plot and save histogram for Homeric scores
        plt.figure(figsize=(10, 6))
        homeros_df['valence_score'].hist(bins=20, alpha=0.7, color='orange', label='Homeric Valence Scores')
        plt.title('Distribution of Valence Scores (Homeric)')
        plt.xlabel('Valence Score')
        plt.ylabel('Frequency')
        plt.legend()
        homeros_hist_path = os.path.join(output_dir, 'homeros_valence_distribution.png')
        plt.savefig(homeros_hist_path)
        plt.close()
        print(f"Homeric distribution plot saved to {homeros_hist_path}")

    except FileNotFoundError:
        print("❌ Error: 'data/Homeros_annotation.csv' not found.")
        return

    # --- 3. Compare Distributions ---
    print("\n--- Comparing Distributions ---")
    plt.figure(figsize=(12, 7))
    nt_df['valence_score'].hist(bins=20, alpha=0.6, label='NT Scores', density=True)
    homeros_df['valence_score'].hist(bins=20, alpha=0.6, label='Homeric Scores', density=True)
    plt.title('Comparison of Valence Score Distributions (Normalized)')
    plt.xlabel('Valence Score')
    plt.ylabel('Density')
    plt.legend()
    comparison_hist_path = os.path.join(output_dir, 'comparison_valence_distribution.png')
    plt.savefig(comparison_hist_path)
    plt.close()
    print(f"Comparison plot saved to {comparison_hist_path}")
    print("\n--- EDA script finished successfully. ---")


if __name__ == '__main__':
    analyze_datasets()
