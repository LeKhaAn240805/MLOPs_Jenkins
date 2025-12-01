# model_evaluation.py
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_preprocessed_data(file_path):
    data = joblib.load(file_path)
    return data["X_test"], data["y_test"]

def evaluate():
    model = tf.keras.models.load_model("sentiment_lstm.h5")

    X_test, y_test = load_preprocessed_data("preprocessed_data.pkl")

    preds = (model.predict(X_test) > 0.5).astype("int32")

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")      
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    evaluate()
