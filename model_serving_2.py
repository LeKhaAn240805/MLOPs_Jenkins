import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import gradio as gr
import socket
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====== CONFIG ======
MAX_FEATURES = 10000
MAX_LEN = 200
LABELS = ["Negative", "Positive"]

# ====== LOAD MODEL ======
model = tf.keras.models.load_model("sentiment_lstm.h5")

# Load IMDb word index
word_index = imdb.get_word_index()

# ====== FASTAPI ======
app = FastAPI(title="Sentiment Analysis API + Gradio UI")

# Encode text input
def encode_text(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    encoded = []
    for w in tokens:
        idx = word_index.get(w, 2)  # 2 = unknown
        if idx < MAX_FEATURES:
            encoded.append(idx + 3)
        else:
            encoded.append(2)
    return pad_sequences([encoded], maxlen=MAX_LEN)

# Predict function
def predict_sentiment(text: str):
    encoded = encode_text(text)
    proba = float(model.predict(encoded)[0][0])
    label = LABELS[1] if proba > 0.5 else LABELS[0]
    return f"Sentiment: {label}\nProbability positive: {proba:.4f}"

# ====== FASTAPI ENDPOINTS ======
class Request(BaseModel):
    text: str

@app.post("/predict")
def predict_api(request: Request):
    encoded = encode_text(request.text)
    proba = float(model.predict(encoded)[0][0])
    label = LABELS[1] if proba > 0.5 else LABELS[0]

    return {"sentiment": label, "probability_positive": proba}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# ====== GRADIO UI ======
def create_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## Sentiment Analysis Demo (LSTM)")
        textbox = gr.Textbox(label="Nhập câu để phân tích cảm xúc")
        output = gr.TextArea(label="Kết quả")
        btn = gr.Button("Predict")
        btn.click(fn=predict_sentiment, inputs=textbox, outputs=output)
    return demo

# Mount Gradio vào FastAPI
gradio_app = gr.mount_gradio_app(app, create_gradio(), path="/ui")

# ====== HELPER ======
def get_local_ip():
    """Lấy IP LAN máy chạy server"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # kết nối tới IP public tạm thời
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# ====== RUN SERVER ======
if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"Server is running on LAN: http://{local_ip}:9000")
    print(f"Gradio UI available at: http://{local_ip}:9000/ui")
    uvicorn.run(app, host="0.0.0.0", port=9000)
