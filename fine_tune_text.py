# fine_tune_text.py
"""
Fine-tune BERT for 5-class emotion classification on a custom text dataset.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

class TextEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def fine_tune_bert(train_texts, train_labels, val_texts, val_labels, num_labels=5, output_dir="./bert_finetuned"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextEmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextEmotionDataset(val_texts, val_labels, tokenizer)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    # Early stopping and model checkpointing
    from transformers import EarlyStoppingCallback
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Try more epochs for better learning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )
    # Define compute_metrics for better evaluation
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average='weighted')
        }
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # Hyperparameter tuning stub (suggest using optuna or Ray Tune)
    # def hp_space(trial):
    #     return {
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
    #         "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 6),
    #         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
    #     }
    # trainer.hyperparameter_search(direction="maximize", hp_space=hp_space)

    # For transfer learning, try other pre-trained models from HuggingFace
    # For ensembling, combine predictions from multiple models

# Example usage (replace with your data):
# train_texts, train_labels = [...], [...]
# val_texts, val_labels = [...], [...]
# fine_tune_bert(train_texts, train_labels, val_texts, val_labels)
