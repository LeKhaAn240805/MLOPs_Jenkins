import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import gradio as gr

MAX_FEATURES = 10000
MAX_LEN = 200

# Load model
model = tf.keras.models.load_model("sentiment_lstm.h5")

# Load IMDb word index
word_index = imdb.get_word_index()

app = FastAPI()

LABELS = ["Negative", "Positive"]

def encode_text(text):
    tokens = text.lower().split()
    encoded = []

    for w in tokens:
        if w in word_index and word_index[w] < MAX_FEATURES:
            encoded.append(word_index[w] + 3)  # IMDb offset
        else:
            encoded.append(2)  # unknown token

    return pad_sequences([encoded], maxlen=MAX_LEN)


def predict_sentiment(text: str):
    encoded = encode_text(text)
    proba = float(model.predict(encoded)[0][0])
    label = LABELS[1] if proba > 0.5 else LABELS[0]
    return f"Sentiment: {label}\nProbability positive: {proba:.4f}"


# FastAPI endpoint (nếu vẫn muốn gọi API)
class Request(BaseModel):
    text: str

@app.post("/predict")
def predict_api(request: Request):
    encoded = encode_text(request.text)
    proba = float(model.predict(encoded)[0][0])
    label = LABELS[1] if proba > 0.5 else LABELS[0]

    return {
        "sentiment": label,
        "probability_positive": proba
    }

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
