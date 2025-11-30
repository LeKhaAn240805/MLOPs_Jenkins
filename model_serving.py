# model_serving.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

MAX_FEATURES = 10000
MAX_LEN = 200

# Load model
model = tf.keras.models.load_model("sentiment_lstm.h5")

# Load word index từ IMDb
word_index = imdb.get_word_index()

app = FastAPI()

class Request(BaseModel):
    text: str

LABELS = ["Negative", "Positive"]

def encode_text(text):
    tokens = text.lower().split()

    encoded = []
    for w in tokens:
        if w in word_index and word_index[w] < MAX_FEATURES:
            encoded.append(word_index[w] + 3)  # IMDb offset
        else:
            encoded.append(2)  # unknown token id = 2

    return pad_sequences([encoded], maxlen=MAX_LEN)

@app.post("/predict")
def predict(request: Request):
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
