import os
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from PIL import Image
import torchaudio
import moviepy.editor as mp
from torchvision import transforms
import tempfile

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Config ----------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LOCAL_MODEL_DIR = "./sentiment_model"
label_map = ["V. Bad", "Bad", "Normal", "Good", "V. Good"]

# ---------- Load Text Model ----------
def load_text_model():
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        model.save_pretrained(LOCAL_MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    return pipeline("sentiment-analysis", model=model.to(device), tokenizer=tokenizer, return_all_scores=True)

text_pipeline = load_text_model()

def convert_score(score):
    positive_score = score[1]['score']
    if positive_score > 0.9:
        return "V. Good"
    elif positive_score > 0.75:
        return "Good"
    elif positive_score > 0.4:
        return "Normal"
    elif positive_score > 0.2:
        return "Bad"
    else:
        return "V. Bad"

# ---------- Routes ----------

@app.route("/predict/text", methods=["POST"])
def predict_text():
    data = request.json
    text = data.get("text", "")
    score = text_pipeline(text)[0]
    sentiment = convert_score(score)
    return jsonify({"sentiment": sentiment})

@app.route("/predict/chat", methods=["POST"])
def predict_chat():
    data = request.json
    chat = data.get("chat", "")
    lines = chat.strip().split("\n")
    scores = []
    for line in lines:
        if line.strip():
            s = text_pipeline(line)[0]
            scores.append(s[1]['score'])
    avg = sum(scores)/len(scores) if scores else 0.5
    sentiment = convert_score([{ 'label': 'NEGATIVE', 'score': 1 - avg }, { 'label': 'POSITIVE', 'score': avg }])
    return jsonify({"sentiment": sentiment})

@app.route("/predict/image", methods=["POST"])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    image = Image.open(request.files['image']).convert("RGB")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensor = transform(image)
    brightness = tensor.mean().item()
    if brightness > 0.8:
        sentiment = "V. Good"
    elif brightness > 0.6:
        sentiment = "Good"
    elif brightness > 0.4:
        sentiment = "Normal"
    elif brightness > 0.2:
        sentiment = "Bad"
    else:
        sentiment = "V. Bad"
    return jsonify({"sentiment": sentiment})

@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        request.files['audio'].save(temp_audio.name)
        waveform, _ = torchaudio.load(temp_audio.name)
        energy = waveform.abs().mean().item()
        os.remove(temp_audio.name)
    if energy > 0.3:
        sentiment = "V. Good"
    elif energy > 0.2:
        sentiment = "Good"
    elif energy > 0.1:
        sentiment = "Normal"
    elif energy > 0.05:
        sentiment = "Bad"
    else:
        sentiment = "V. Bad"
    return jsonify({"sentiment": sentiment})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online"}), 200


@app.route("/predict/video", methods=["POST"])
def predict_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        request.files['video'].save(temp_video.name)
        clip = mp.VideoFileClip(temp_video.name)
        audio_path = temp_video.name + ".wav"
        clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
        waveform, _ = torchaudio.load(audio_path)
        energy = waveform.abs().mean().item()
        os.remove(temp_video.name)
        os.remove(audio_path)
    if energy > 0.3:
        sentiment = "V. Good"
    elif energy > 0.2:
        sentiment = "Good"
    elif energy > 0.1:
        sentiment = "Normal"
    elif energy > 0.05:
        sentiment = "Bad"
    else:
        sentiment = "V. Bad"
    return jsonify({"sentiment": sentiment})

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
