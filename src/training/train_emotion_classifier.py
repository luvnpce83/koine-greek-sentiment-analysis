# train_local.py (Corrected for RuntimeError)

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, ClassLabel
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse
import wandb

# --- 1. Custom Trainer for Weighted Loss ---
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        # 🎯 FIX: Corrected --1 to -1 for tensor reshaping
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 2. Custom Metrics Function (No Changes) ---
def compute_metrics_custom(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
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
    local_data_path = './data/augmented_emotion_data.csv'
    output_dir_name = f"full-finetune_epochs{args.epochs}_lr{args.learning_rate}_seed{args.seed}"
    local_output_dir = os.path.join('./local_model_output', output_dir_name)

    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    # --- 4. Prepare Dataset (Leakage Corrected) ---
    print("--- Loading and Preparing Dataset (Leakage Corrected) ---")
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    # Load original data to create a clean, un-augmented validation set
    primary_df = pd.read_csv('./data/primary_emotion_data.csv')
    augmented_df = pd.read_csv(local_data_path)

    # Split the ORIGINAL data to get a clean, un-augmented validation set
    primary_train_df, eval_df = sklearn_train_test_split(
        primary_df,
        test_size=0.2,
        stratify=primary_df['label'],
        random_state=args.seed
    )
    
    # The training set is the entire augmented dataset, MINUS the sentences reserved for evaluation
    eval_texts = set(eval_df['text'])
    train_df = augmented_df[~augmented_df['text'].isin(eval_texts)].copy()

    print(f"Clean Split Complete: Train size={len(train_df)}, Eval size={len(eval_df)}")

    primary_labels = sorted(augmented_df['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(primary_labels)}
    id2label = {i: label for i, label in enumerate(primary_labels)}
    
    train_df['labels'] = train_df['label'].map(label2id)
    eval_df['labels'] = eval_df['label'].map(label2id)

    train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
    eval_dataset = Dataset.from_pandas(eval_df[['text', 'labels']])

    # --- 5. Tokenize Data ---
    print("--- Tokenizing Data ---")
    tokenizer = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    train_dataset = train_dataset.cast_column("labels", ClassLabel(names=primary_labels))
    eval_dataset = eval_dataset.cast_column("labels", ClassLabel(names=primary_labels))

    # --- 6. Calculate Class Weights (No Changes) ---
    print("--- Calculating Class Weights ---")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_dataset['labels']),
        y=np.array(train_dataset['labels'])
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Calculated Class Weights: {class_weights_tensor}")

    # --- 7. Configure LoRA and Load Model (No Changes) ---
    # print(f"--- Configuring LoRA with r={args.lora_r} ---")
    # peft_config = LoraConfig(
    #     lora_alpha=args.lora_r * 2,
    #     lora_dropout=0.1,
    #     r=args.lora_r,
    #     bias="none",
    #     task_type=TaskType.SEQ_CLS,
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
        "pranaydeeps/Ancient-Greek-BERT",
        num_labels=len(primary_labels),
        label2id=label2id,
        id2label=id2label
    )
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    # --- 8. Define Training Arguments ---
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
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        greater_is_better=True,
        use_mps_device=torch.backends.mps.is_available(),
        report_to="wandb",
        run_name=output_dir_name
    )

    # --- 9. Initialize Trainer with Early Stopping ---
    print("--- Initializing Trainer with Early Stopping ---")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_custom,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
    )

    trainer.train()

    # --- 10. Save Final Model ---
    print("--- Training complete. Saving final model. ---")
    trainer.save_model(local_output_dir)
    wandb.finish()
    print(f"🎉 Model and logs successfully saved to {local_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Ancient Greek BERT with LoRA.")
    
    parser.add_argument("--epochs", type=int, default=32, help="Maximum number of training epochs.")
    parser.add_argument("--lora_r", type=int, default=16, help="The 'r' parameter for LoRA.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Stop training if f1_macro doesn't improve for this many epochs.")
    
    args = parser.parse_args()
    
    try:
        wandb.login()
    except Exception as e:
        print(f"Could not log in to wandb: {e}")

    main(args)