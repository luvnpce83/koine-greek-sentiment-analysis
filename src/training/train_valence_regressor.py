# train_valence_regressor.py (v4 - W&B Enabled)

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import os
import argparse
import wandb

# --- 1. Custom Trainer for Regression ---
class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

# --- 2. Custom Metrics Function for Regression ---
def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()

    if len(np.unique(labels)) < 2 or len(np.unique(preds)) < 2:
        pearson_corr, spearman_corr = 0.0, 0.0
    else:
        pearson_corr, _ = pearsonr(labels, preds)
        spearman_corr, _ = spearmanr(labels, preds)

    mse = mean_squared_error(labels, preds)

    return {
        "mse": mse,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
    }

def set_seed(seed):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(args.seed)

    # --- 3. Setup Paths and Directories ---
    train_pool_path = './data/final_valence_train_pool.csv'
    primary_test_path = './data/nt_primary_test_set.csv'
    secondary_test_path = './data/homeros_secondary_test_set.csv'

    output_dir_name = f"valence-regressor-augmented_epochs{args.epochs}_lr{args.learning_rate}_seed{args.seed}"
    local_output_dir = os.path.join('./local_model_output', output_dir_name)

    os.makedirs(local_output_dir, exist_ok=True)

    # --- 4. Prepare Dataset ---
    print("--- Loading and Preparing Augmented Dataset for Regression ---")

    train_pool_df = pd.read_csv(train_pool_path).dropna(subset=['text'])
    train_df, eval_df = train_test_split(train_pool_df, test_size=0.15, random_state=args.seed)

    primary_test_df = pd.read_csv(primary_test_path).dropna(subset=['text'])
    secondary_test_df = pd.read_csv(secondary_test_path).dropna(subset=['text'])

    print(f"Data Split: Train={len(train_df)}, Eval={len(eval_df)}")
    print(f"Test Sets: Primary (NT)={len(primary_test_df)}, Secondary (Homeric)={len(secondary_test_df)}")

    for df in [train_df, eval_df, primary_test_df, secondary_test_df]:
        df.rename(columns={'valence_score': 'labels'}, inplace=True)

    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'labels']])
    primary_test_dataset = Dataset.from_pandas(primary_test_df[['text', 'labels']])
    secondary_test_dataset = Dataset.from_pandas(secondary_test_df[['text', 'labels']])

    # --- 5. Tokenize Data ---
    print("--- Tokenizing Data ---")
    tokenizer = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    primary_test_dataset = primary_test_dataset.map(tokenize_function, batched=True)
    secondary_test_dataset = secondary_test_dataset.map(tokenize_function, batched=True)

    # --- 6. Load Model for Regression ---
    print("--- Loading Model for Regression (num_labels=1) ---")
    model = AutoModelForSequenceClassification.from_pretrained("pranaydeeps/Ancient-Greek-BERT", num_labels=1)

    # --- 7. Define Training Arguments ---
    print("--- Defining Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{local_output_dir}/logs",
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson_correlation",
        save_total_limit=2,
        greater_is_better=True,
        report_to="wandb",
        run_name=output_dir_name,
    )

    # --- 8. Initialize Trainer ---
    print("--- Initializing Trainer ---")
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_regression,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # --- 9. Train the Model ---
    print("\n--- Starting Model Training on Augmented Data ---")
    trainer.train()

    # --- 10. Final Multi-Faceted Evaluation ---
    print("\n--- Evaluating on Primary Test Set (In-Domain NT) ---")
    primary_test_results = trainer.predict(primary_test_dataset)
    print("Primary Test Set Metrics:")
    print(primary_test_results.metrics)

    print("\n--- Evaluating on Secondary Test Set (Out-of-Domain Homeric) ---")
    secondary_test_results = trainer.predict(secondary_test_dataset)
    print("Secondary Test Set Metrics:")
    print(secondary_test_results.metrics)

    # Log final metrics to wandb if it's active
    if wandb.run:
        wandb.log({
            "primary_test_metrics": primary_test_results.metrics,
            "secondary_test_metrics": secondary_test_results.metrics
        })
        wandb.finish()

    # --- 11. Save Final Model ---
    print("\n--- Training complete. Saving final model. ---")
    trainer.save_model(local_output_dir)
    print(f"🎉 Model and logs successfully saved to {local_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Ancient Greek BERT for Valence Regression (Augmented).")

    parser.add_argument("--epochs", type=int, default=10, help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop training if pearson_correlation doesn't improve for this many epochs.")

    args = parser.parse_args()

    # The user is responsible for logging in via the CLI (`wandb login`)
    # The script will report to wandb if it can, otherwise the Trainer handles it gracefully.
    main(args)
