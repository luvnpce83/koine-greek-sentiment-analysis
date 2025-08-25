# train_valence_regressor.py (final version)

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
        # For regression, logits are the predicted values. Squeeze to match label shape.
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss

# --- 2. Custom Metrics Function for Regression ---
def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()

    mse = mean_squared_error(labels, preds)
    pearson_corr, _ = pearsonr(labels, preds)
    spearman_corr, _ = spearmanr(labels, preds)

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
    train_pool_path = './data/valence_data_augmented_train_pool.csv'
    test_set_path = './data/valence_test_set_nt.csv'
    output_dir_name = f"valence-regressor_epochs{args.epochs}_lr{args.learning_rate}_seed{args.seed}"
    local_output_dir = os.path.join('./local_model_output', output_dir_name)

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    # --- 4. Prepare Dataset ---
    print("--- Loading and Preparing Dataset for Regression ---")

    train_pool_df = pd.read_csv(train_pool_path)
    train_df, eval_df = train_test_split(
        train_pool_df,
        test_size=0.15,
        random_state=args.seed
    )
    test_df = pd.read_csv(test_set_path)

    print(f"Data Split: Train={len(train_df)}, Eval={len(eval_df)}, Final Test={len(test_df)}")

    train_df = train_df.rename(columns={'valence_score': 'labels'})
    eval_df = eval_df.rename(columns={'valence_score': 'labels'})
    test_df = test_df.rename(columns={'valence_score': 'labels'})

    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'labels']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

    # --- 5. Tokenize Data ---
    print("--- Tokenizing Data ---")
    tokenizer = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")

    def tokenize_function(examples):
        # 🟢 CHANGE: Use the max_length from args
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_length)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # --- 6. Load Model for Regression ---
    print("--- Loading Model for Regression (num_labels=1) ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        "pranaydeeps/Ancient-Greek-BERT",
        num_labels=1 # This is the key change for regression
    )

    # --- 7. Define Training Arguments ---
    print("--- Defining Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=f"{local_output_dir}/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="pearson_correlation",
        save_total_limit=2,
        greater_is_better=True,
        report_to="wandb",
        run_name=output_dir_name
    )

    # --- 8. Initialize Trainer ---
    print("--- Initializing Trainer ---")
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_regression,
        tokenizer=tokenizer, # 🟢 ADDITION: Pass tokenizer to Trainer
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    # --- 9. Train the Model ---
    print("--- Starting Model Training ---")
    trainer.train()

    # --- 10. Evaluate on the Final Held-Out Test Set ---
    print("\n--- Evaluating on Final Held-Out Test Set ---")
    test_results = trainer.predict(test_dataset)
    print("Final Test Set Metrics:")
    print(test_results.metrics)

    wandb.log({"final_test_metrics": test_results.metrics})

    # --- 11. Save Final Model and Tokenizer ---
    print("\n--- Training complete. Saving final model and tokenizer. ---")
    # 🟢 CHANGE: This function now saves BOTH the model and the tokenizer
    trainer.save_model(local_output_dir)
    wandb.finish()
    print(f"🎉 Model, tokenizer, logs, and test results successfully saved to {local_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Ancient Greek BERT for Valence Regression.")

    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Stop training if pearson_correlation doesn't improve for this many epochs.")
    # 🟢 ADDITION: New argument for tokenizer max length
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for the tokenizer.")

    args = parser.parse_args()

    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log in to wandb: {e}")

    main(args)
