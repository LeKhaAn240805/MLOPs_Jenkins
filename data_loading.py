# data_loading.py
import joblib
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_FEATURES = 10000   # số lượng từ giữ lại
MAX_LEN = 200          # độ dài chuỗi đầu vào

def load_data():
    print("Loading IMDb sentiment dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

    print("Example of raw data sample:", X_train[0][:20])
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    print("Padding sequences...")

    X_train_padded = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test_padded  = pad_sequences(X_test,  maxlen=MAX_LEN)

    return X_train_padded, X_test_padded

def save_preprocessed_data(X_train, X_test, y_train, y_test, file_path):
    joblib.dump({
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }, file_path)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    X_train_padded, X_test_padded = preprocess_data(X_train, X_test)

    save_preprocessed_data(X_train_padded, X_test_padded, y_train, y_test,
                           "preprocessed_data.pkl")

    print("Preprocessing done")
