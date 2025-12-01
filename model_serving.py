# model_serving.py
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

LABELS = ["Negative", "Positive"]

class Request(BaseModel):
    text: str

def load_model(file_path):
    data = joblib.load(file_path)
    return data["model"], data["pipeline"]

model, pipeline = load_model("model.pkl")

@app.post("/predict")
def predict(input_data: Request):
    transformed = pipeline.transform([input_data.text])
    pred = model.predict(transformed)[0]
    
    return {"sentiment": LABELS[pred]}

@app.get("/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
