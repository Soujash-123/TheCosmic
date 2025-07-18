# 📄 Sentiment Analysis API — Deployment & Integration Guide

## 🚀 Overview

This guide explains how to **deploy**, **host**, and **integrate** a multimodal sentiment analysis API built using Flask and transformer models. The API supports:

* ✅ **Text** input
* 🖼️ **Images** (e.g., emotion from photo brightness — placeholder logic)
* 🔊 **Audio** (speech energy — placeholder logic)
* 📹 **Video** (converted to audio for sentiment)
* 💬 **Chat History** (multi-line text, analyzed line by line)

It classifies sentiment into five categories:

> **Very Good**, **Good**, **Normal**, **Bad**, **Very Bad**

---

## 🛠️ Project Structure

```
project-root/
├── app.py                  # Main Flask app
├── sentiment_model/       # Auto-downloaded HuggingFace model (compact)
├── requirements.txt       # Dependencies
└── render.yaml            # (Optional) Render.com deploy file
```

---

## 📦 1. Install Requirements (Local / Colab)

To install all dependencies:

```bash
pip install -r requirements.txt
```

> 💡 Use `python -m venv myenv` and `source myenv/bin/activate` before installing to isolate dependencies.

If working in Colab:

```python
!pip install flask transformers torch torchvision torchaudio opencv-python moviepy
```

---

## 🧪 2. Run Locally

Start the Flask server:

```bash
python app.py
```

Test the API is live:

```bash
curl http://localhost:5000/ping
```

Expected:

```json
{"status":"online"}
```

---

## ☁️ 3. Deploy to Render (Cloud Hosting)

### ✅ Option A: Auto Deploy from GitHub

1. Push your project to GitHub
2. Visit [https://render.com](https://render.com)
3. Choose **Web Service → Connect your GitHub repo**
4. Configure:

   * **Build Command**: `pip install -r requirements.txt`
   * **Start Command**: `gunicorn app:app`
   * **Environment Variable**: `PYTHON_VERSION = 3.10`

### ✅ Option B: Use `render.yaml`

Include this in the root of your repo:

```yaml
services:
  - type: web
    name: sentiment-api
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
```

Then push to GitHub and deploy on Render.

---

## 🔌 4. REST API Endpoints

| Endpoint         | Method | Input Type | Example cURL                                                                                                               |
| ---------------- | ------ | ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| `/ping`          | GET    | -          | `curl http://localhost:5000/ping`                                                                                          |
| `/predict/text`  | POST   | JSON       | `curl -X POST -H "Content-Type: application/json" -d '{"text": "I love it"}' http://localhost:5000/predict/text`           |
| `/predict/chat`  | POST   | JSON       | `curl -X POST -H "Content-Type: application/json" -d '{"chat": "Hi\nI feel terrible"}' http://localhost:5000/predict/chat` |
| `/predict/image` | POST   | Multipart  | `curl -X POST -F image=@"path.jpg" http://localhost:5000/predict/image`                                                    |
| `/predict/audio` | POST   | Multipart  | `curl -X POST -F audio=@"audio.wav" http://localhost:5000/predict/audio`                                                   |
| `/predict/video` | POST   | Multipart  | `curl -X POST -F video=@"clip.mp4" http://localhost:5000/predict/video`                                                    |

> 🔒 You can later add token-based auth or API keys for securing endpoints.

---

## 📁 5. GitHub-Safe Model Folder

The app downloads and saves only minimal required files from Hugging Face:

```
sentiment_model/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json
├── vocab.txt
```

✅ Size is \~60–90MB (safe to commit to GitHub)

---

## ⚙️ 6. Integration Options

### Web Frontend

You can use JS/AJAX or Python frontend to consume endpoints:

```js
fetch("/predict/text", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "This is amazing!" })
})
.then(res => res.json())
.then(data => console.log(data.sentiment));
```

### Python Client

```python
import requests
response = requests.post("http://localhost:5000/predict/text", json={"text": "Great work"})
print(response.json())
```

### Frontend CORS (Optional)

Enable CORS by installing Flask-CORS:

```bash
pip install flask-cors
```

Then add to `app.py`:

```python
from flask_cors import CORS
CORS(app)
```

---

## 💡 7. Advanced Enhancements

| Feature                 | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| 🔐 Auth Middleware      | Add API keys / JWT-based auth                         |
| 🧠 Replace Placeholders | Use ViT for images and Wav2Vec for audio              |
| 🔄 Batch Processing     | Accept multiple inputs in one POST                    |
| 📊 Logging & Metrics    | Add request logging, latency, counters                |
| 🚢 Docker Deploy        | Use Dockerfile + `docker-compose.yml` for portability |

---

## 🧼 8. Clean GitHub Commit

Before pushing:

```bash
git add .
git commit -m "Production-ready sentiment API"
git push origin main
```

---

## 🧠 Need Help?

* Ping the repo maintainer
* Or visit [https://huggingface.co/models](https://huggingface.co/models) for alternate transformer models

---

