# fine_tune_audio.py
"""
Fine-tune Wav2Vec2 for 5-class emotion classification on a custom audio dataset.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Trainer, TrainingArguments

class AudioEmotionDataset(Dataset):
    def __init__(self, audio_arrays, labels, processor, sampling_rate=16000):
        self.audio_arrays = audio_arrays
        self.labels = labels
        self.processor = processor
        self.sampling_rate = sampling_rate
    def __len__(self):
        return len(self.audio_arrays)
    def __getitem__(self, idx):
        inputs = self.processor(self.audio_arrays[idx], sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def fine_tune_wav2vec2(train_audio, train_labels, val_audio, val_labels, num_labels=5, output_dir="./wav2vec2_finetuned"):
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    train_dataset = AudioEmotionDataset(train_audio, train_labels, processor)
    val_dataset = AudioEmotionDataset(val_audio, val_labels, processor)
    model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-base-960h', num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

# Example usage (replace with your data):
# fine_tune_wav2vec2(train_audio, train_labels, val_audio, val_labels)
