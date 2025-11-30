# model_training.py
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

MAX_FEATURES = 10000
MAX_LEN = 200

def load_preprocessed_data(file_path):
    data = joblib.load(file_path)
    return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

def build_lstm_model():
    model = Sequential([
        Embedding(MAX_FEATURES, 128, input_length=MAX_LEN),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_preprocessed_data("preprocessed_data.pkl")

    model = build_lstm_model()

    print("Training LSTM model...")
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1
    )

    print("Saving model...")
    model.save("sentiment_lstm.h5")

    print("Training completed!")
