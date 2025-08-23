import pandas as pd

def build_final_pool():
    """
    Combines the augmented NT data with the Homeric training data to create
    the final, balanced training pool for the regression model.
    """
    print("--- Building Final Training Pool ---")

    try:
        # Load the two components of our training data
        augmented_nt_df = pd.read_csv('data/augmented_nt_train.csv')
        homeros_pool_df = pd.read_csv('data/homeros_train_pool.csv')
    except FileNotFoundError as e:
        print(f"❌ Error: Missing a required data file. Details: {e}")
        print("Please ensure 'augmented_nt_train.csv' and 'homeros_train_pool.csv' exist.")
        return

    print(f"Loaded {len(augmented_nt_df)} augmented NT samples.")
    print(f"Loaded {len(homeros_pool_df)} Homeric training samples.")

    # --- 1. Combine and Shuffle ---
    final_pool_df = pd.concat([augmented_nt_df, homeros_pool_df], ignore_index=True)

    # Shuffle the combined dataset to ensure random distribution during training
    final_pool_df = final_pool_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Combined and shuffled. Total samples in final pool: {len(final_pool_df)}")

    # --- 2. Save the final dataset ---
    final_pool_path = 'data/final_valence_train_pool.csv'
    final_pool_df.to_csv(final_pool_path, index=False)

    print(f"\n✅ Final training pool successfully saved to: {final_pool_path}")


if __name__ == '__main__':
    build_final_pool()
