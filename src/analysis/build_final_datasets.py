import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_final_datasets():
    """
    Splits the NT data and merges with Homeric data to create the final
    training and testing datasets for the valence regression task.
    """
    print("--- Building Final Datasets for Valence Regression ---")

    try:
        # Load the processed datasets from the EDA step
        nt_df = pd.read_csv('data/processed_nt_valence.csv')
        homeros_df = pd.read_csv('data/processed_homeros_valence.csv')

        # Ensure column names are consistent ('text', 'valence_score')
        # The EDA script saved the Homeric text column as 'text' if it found it.
        # Let's rename it just in case for consistency before merging.
        homeros_df.rename(columns={col: 'text' for col in homeros_df.columns if 'text' in col.lower()}, inplace=True)

    except FileNotFoundError as e:
        print(f"❌ Error: Missing processed data file. Please run the EDA script first. Details: {e}")
        return

    # --- 1. Split the NT dataset ---
    # We'll use 20% of the NT data as a held-out test set.
    # The random_state ensures the split is the same every time.
    nt_train_pool, nt_test_set = train_test_split(
        nt_df,
        test_size=0.2,
        random_state=42  # The answer to life, the universe, and everything.
    )

    print(f"NT data split: {len(nt_train_pool)} for training pool, {len(nt_test_set)} for final test set.")

    # --- 2. Combine training pool ---
    # The main training pool will consist of 80% of NT data and all of the Homeric data.
    train_pool_df = pd.concat([nt_train_pool, homeros_df], ignore_index=True)

    # Shuffle the combined dataset to mix NT and Homeric samples
    train_pool_df = train_pool_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Combined training pool created with {len(train_pool_df)} total samples.")

    # --- 3. Save the final datasets ---
    train_pool_path = 'data/valence_train_pool.csv'
    test_set_path = 'data/valence_test_set.csv'

    train_pool_df.to_csv(train_pool_path, index=False)
    print(f"Full training pool saved to {train_pool_path}")

    nt_test_set.to_csv(test_set_path, index=False)
    print(f"Final held-out test set saved to {test_set_path}")

    print("\n--- Final dataset creation complete. ---")


if __name__ == '__main__':
    create_final_datasets()
