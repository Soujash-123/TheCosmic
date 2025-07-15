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
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Example usage (replace with your data):
# train_texts, train_labels = [...], [...]
# val_texts, val_labels = [...], [...]
# fine_tune_bert(train_texts, train_labels, val_texts, val_labels)
