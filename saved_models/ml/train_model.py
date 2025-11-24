# ml/train_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import joblib
import os

# =============================
# CONFIG
# =============================
SEQ_LEN = 50
EPOCHS = 50
BATCH_SIZE = 32
DATA_PATH = "data/training_dataset.csv"  # path to your CSV

FEATURES = [
    'TGS2620','TGS2602','TGS2600',
    'Drift2620','Var2620','Flat2620',
    'Drift2600','Var2600','Flat2600',
    'Drift2602','Var2602','Flat2602',
    'Uptime_sec','Jitter_ms','RSSI_dBm','CPU_Temp_C','FreeHeap_bytes'
]

SAVE_DIR = "ml/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)[FEATURES].dropna()

# Create pseudo risk score if missing
df["risk_score"] = (
    (df["TGS2620"] + df["TGS2602"] + df["TGS2600"]) / 1000 +
    df["Drift2600"] + df["Drift2602"] + df["Drift2620"]
) / 10
df["risk_score"] = df["risk_score"].clip(0, 1)

# =============================
# WINDOWING
# =============================
X, y = [], []
data = df[FEATURES].values
labels = df["risk_score"].values

for i in range(len(data) - SEQ_LEN):
    X.append(data[i:i+SEQ_LEN])
    y.append(labels[i+SEQ_LEN])

X, y = np.array(X, np.float32), np.array(y, np.float32)

# =============================
# SCALER
# =============================
flat = X.reshape(-1, X.shape[-1])
scaler = StandardScaler().fit(flat)
joblib.dump(scaler, f"{SAVE_DIR}/global_scaler.joblib")

X = scaler.transform(flat).reshape(X.shape)

# =============================
# MODEL
# =============================
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=(SEQ_LEN, X.shape[-1])),
    layers.Conv1D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling1D(2),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model.save(f"{SAVE_DIR}/base_cnn_lstm.h5")

print(f"\n✅ Model saved to {SAVE_DIR}/base_cnn_lstm.h5")
