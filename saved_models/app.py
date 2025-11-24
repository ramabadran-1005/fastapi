from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------- Load Model and Scaler ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "base_cnn_lstm_fixed.h5")
SCALER_PATH = os.path.join(BASE_DIR, "global_scaler.joblib")

print("Loading artifacts...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
print("✅ Model & Scaler Loaded Successfully!")

# ---------- Classification Logic ----------
def classify_risk(score):
    if score < 0.3:
        return "Healthy"
    elif score < 0.6:
        return "Moderate Risk"
    elif score < 0.8:
        return "High Risk"
    else:
        return "Critical"

# ---------- Prediction API ----------
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        if not data or "rows" not in data:
            return jsonify({"error": "Missing 'rows'"}), 400

        df = pd.DataFrame(data["rows"])
        feature_cols = [c for c in df.columns if c not in ["_id", "timestamp", "nodeId"]]
        X_scaled = scaler.transform(df[feature_cols])

        preds = model.predict(X_scaled)
        risk_scores = np.clip(preds.flatten(), 0, 1)
        labels = [classify_risk(s) for s in risk_scores]

        df["risk_score"] = risk_scores
        df["status"] = labels
        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
