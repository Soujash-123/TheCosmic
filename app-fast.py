from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from fine_tune_text import fine_tune_bert

app = FastAPI()

MODEL_DIR = "./bert_finetuned"
model = None
tokenizer = None

class TrainRequest(BaseModel):
    train_texts: list
    train_labels: list
    val_texts: list
    val_labels: list
    output_dir: str = MODEL_DIR

class PredictRequest(BaseModel):
    texts: list

@app.post('/train')
def train(req: TrainRequest):
    fine_tune_bert(req.train_texts, req.train_labels, req.val_texts, req.val_labels, output_dir=req.output_dir)
    global model, tokenizer
    model = BertForSequenceClassification.from_pretrained(req.output_dir)
    tokenizer = BertTokenizer.from_pretrained(req.output_dir)
    return {"message": f"Model trained and saved to {req.output_dir}"}

@app.post('/predict')
def predict(req: PredictRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        if not os.path.exists(MODEL_DIR):
            return {"error": "Model not trained yet."}
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    inputs = tokenizer(req.texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
    return {"predictions": preds}
