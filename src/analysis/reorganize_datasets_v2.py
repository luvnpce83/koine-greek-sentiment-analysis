import pandas as pd
from sklearn.model_selection import train_test_split

def reorganize_datasets_v2():
    """
    Implements the advanced data splitting strategy:
    - Splits NT data into a primary test set (20) and training seeds (62).
    - Splits Homeric data into a secondary test set (30) and a training pool (581).
    """
    print("--- Reorganizing Datasets with V2 Strategy ---")

    try:
        # Load the processed datasets from the EDA step
        nt_df = pd.read_csv('data/processed_nt_valence.csv')
        homeros_df = pd.read_csv('data/processed_homeros_valence.csv')
    except FileNotFoundError as e:
        print(f"❌ Error: Missing processed data file. Please run the EDA script first. Details: {e}")
        return

    # --- 1. Split the NT dataset ---
    # We need exactly 20 samples for the primary test set.
    nt_train_seeds_df, nt_primary_test_set_df = train_test_split(
        nt_df,
        test_size=20,
        random_state=42 # Reproducibility is key
    )

    print(f"NT data split: {len(nt_train_seeds_df)} for training seeds, {len(nt_primary_test_set_df)} for primary test set.")

    # --- 2. Split the Homeric dataset ---
    # We need 30 samples for the secondary (generalization) test set.
    homeros_train_pool_df, homeros_secondary_test_set_df = train_test_split(
        homeros_df,
        test_size=30,
        random_state=42
    )

    print(f"Homeric data split: {len(homeros_train_pool_df)} for training pool, {len(homeros_secondary_test_set_df)} for secondary test set.")

    # --- 3. Save the new datasets ---
    nt_train_seeds_df.to_csv('data/nt_train_seeds.csv', index=False)
    nt_primary_test_set_df.to_csv('data/nt_primary_test_set.csv', index=False)
    homeros_train_pool_df.to_csv('data/homeros_train_pool.csv', index=False)
    homeros_secondary_test_set_df.to_csv('data/homeros_secondary_test_set.csv', index=False)

    print("\nNew datasets saved successfully:")
    print("- data/nt_train_seeds.csv")
    print("- data/nt_primary_test_set.csv")
    print("- data/homeros_train_pool.csv")
    print("- data/homeros_secondary_test_set.csv")

    print("\n--- Dataset reorganization complete. ---")


if __name__ == '__main__':
    reorganize_datasets_v2()
