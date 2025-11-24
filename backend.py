# backend.py — FastAPI ML prediction service (no Mongo dependency)
# Place this in your ml/ folder and run:
#   uvicorn backend:app --host 0.0.0.0 --port 5000
#
# Behavior:
# - Loads saved models if available:
#     MODEL_PATH default: ./saved_models
#     base_cnn_lstm.h5
#     meta_extra_trees.joblib
#     global_scaler.joblib
# - Falls back to heuristic if models missing
# - Accepts POST /predict with:
#     {"sequence":[[tgs2620,tgs2602,tgs2600],...], "nodeId": 2101}
# - Returns rich output:
#     riskScore, binary, ternary, four_class, health_state, ops_mode
# - No Mongo usage. Safe, portable.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os
import numpy as np
import joblib
import time
from datetime import datetime

app = FastAPI(title="NWarehouse ML (predict)", version="1.0")

# ----------------------------
# CONFIG + MODEL PATHS
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/ml/saved_models")
BASE_MODEL_FILE = os.path.join(MODEL_PATH, "base_cnn_lstm.h5")
META_MODEL_FILE = os.path.join(MODEL_PATH, "meta_extra_trees.joblib")
SCALER_FILE = os.path.join(MODEL_PATH, "global_scaler.joblib")

base_model = None
meta_model = None
scaler = None
tensorflow_available = False

# ----------------------------
# LOAD OPTIONAL TENSORFLOW
# ----------------------------
try:
    import tensorflow as tf
    tensorflow_available = True
except Exception:
    tensorflow_available = False

# ----------------------------
# LOAD BASE MODEL
# ----------------------------
if tensorflow_available and os.path.exists(BASE_MODEL_FILE):
    try:
        base_model = tf.keras.models.load_model(BASE_MODEL_FILE)
        print("Loaded base model:", BASE_MODEL_FILE)
    except Exception as e:
        print("Failed to load base model:", e)
        base_model = None
else:
    if os.path.exists(BASE_MODEL_FILE):
        print("TensorFlow missing: base model skipped.")
    else:
        print("Base model file not found.")

# ----------------------------
# LOAD META MODEL
# ----------------------------
if os.path.exists(META_MODEL_FILE):
    try:
        meta_model = joblib.load(META_MODEL_FILE)
        print("Loaded meta model:", META_MODEL_FILE)
    except Exception as e:
        print("Failed to load meta model:", e)
        meta_model = None
else:
    print("Meta model file not found.")

# ----------------------------
# LOAD SCALER
# ----------------------------
if os.path.exists(SCALER_FILE):
    try:
        scaler = joblib.load(SCALER_FILE)
        print("Loaded scaler:", SCALER_FILE)
    except Exception as e:
        print("Failed to load scaler:", e)
        scaler = None
else:
    print("Scaler file not found.")


# ----------------------------
# REQUEST PAYLOAD
# ----------------------------
class PredictPayload(BaseModel):
    sequence: List[List[float]]
    nodeId: Optional[Any] = None


# ----------------------------
# NORMALIZE SCORE 0..100
# ----------------------------
def normalize_score(raw: float) -> float:
    try:
        r = float(raw)
    except Exception:
        return 0.0

    if 0.0 <= r <= 1.0:
        return r * 100.0

    if r < 0:
        return 0.0
    if r > 100:
        return min(r, 100.0)
    return r


# ----------------------------
# CLASSIFICATION LOGIC
# ----------------------------
def classify(score: float) -> Dict[str, str]:
    # binary
    binary = "Healthy" if score <= 50 else "Faulty"

    # ternary
    if score <= 20:
        ternary = "Good"
    elif score <= 50:
        ternary = "Warning"
    else:
        ternary = "Critical"

    # multi-class
    if score <= 10:
        four_class = "Normal"
        health_state = "Healthy"
        ops_mode = "Normal Operation"
    elif score <= 40:
        four_class = "Alert"
        health_state = "Degraded"
        ops_mode = "Monitor"
    elif score <= 75:
        four_class = "Failure Likely"
        health_state = "Unstable"
        ops_mode = "Prepare Maintenance"
    else:
        four_class = "Failure Imminent"
        health_state = "Failed"
        ops_mode = "Shutdown / Replace Node"

    return {
        "binary": binary,
        "ternary": ternary,
        "four_class": four_class,
        "health_state": health_state,
        "ops_mode": ops_mode,
    }


# ----------------------------
# BASE MODEL PREDICTION
# ----------------------------
def model_predict_from_base(sequence: np.ndarray) -> float:
    if base_model is None:
        raise RuntimeError("base_model not loaded")
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim == 2:
        x = x.reshape((1,) + x.shape)
    p = base_model.predict(x, verbose=0)
    return float(np.mean(p))


# ----------------------------
# META MODEL PREDICTION
# ----------------------------
def meta_predict_from_base_pred(base_pred: float) -> float:
    if meta_model is None:
        raise RuntimeError("meta_model not loaded")
    arr = np.asarray([[base_pred]])
    out = meta_model.predict(arr)
    return float(out[0])


# ----------------------------
# FALLBACK HEURISTIC
# ----------------------------
def heuristic_score(sequence: List[List[float]]) -> float:
    arr = np.array(sequence, dtype=float)
    if arr.size == 0:
        return 0.0

    if scaler is not None:
        try:
            feats_mean = arr.mean(axis=0)
            scaled = scaler.transform([feats_mean])[0]
            return float(np.mean(scaled)) * 100.0
        except Exception:
            pass

    if arr.shape[-1] == 3:
        latest = arr[-1]
        w = np.array([0.4, 0.3, 0.3])
        denom = np.max(latest) if np.max(latest) > 0 else 1.0
        raw = np.dot(w, latest / denom)
        return normalize_score(raw)

    avg = float(np.mean(arr))
    if avg <= 1.0:
        return avg * 100
    return min(100.0, (avg / (avg + 100)) * 100)


# ----------------------------
# PREDICT ROUTE
# ----------------------------
@app.post("/predict")
async def predict(payload: PredictPayload):
    start_ts = time.time()
    seq = payload.sequence

    if not seq or not isinstance(seq, list):
        raise HTTPException(status_code=400, detail="sequence missing")

    try:
        seq_arr = np.array(seq, dtype=float)
    except:
        raise HTTPException(status_code=400, detail="sequence must be numeric")

    model_used = "heuristic"
    final_score = 0.0

    try:
        # base + scaler
        if base_model is not None and scaler is not None:
            try:
                flat = seq_arr.reshape(-1, seq_arr.shape[-1])
                scaled = scaler.transform(flat)
                x = scaled.reshape((1, scaled.shape[0], scaled.shape[1]))
                bp = model_predict_from_base(x)
                model_used = "base"

                if meta_model:
                    mp = meta_predict_from_base_pred(bp)
                    final_score = normalize_score(mp)
                    model_used = "meta"
                else:
                    final_score = normalize_score(bp)
            except:
                final_score = heuristic_score(seq)
                model_used = "heuristic"

        # base only
        elif base_model is not None:
            try:
                bp = model_predict_from_base(
                    seq_arr if seq_arr.ndim == 3 else seq_arr.reshape((1, seq_arr.shape[0], seq_arr.shape[1]))
                )
                model_used = "base"

                if meta_model:
                    mp = meta_predict_from_base_pred(bp)
                    final_score = normalize_score(mp)
                    model_used = "meta"
                else:
                    final_score = normalize_score(bp)

            except:
                final_score = heuristic_score(seq)
                model_used = "heuristic"

        # no base model
        else:
            if meta_model:
                try:
                    guess = heuristic_score(seq) / 100.0
                    mp = meta_model.predict([[guess]])[0]
                    final_score = normalize_score(mp)
                    model_used = "meta"
                except:
                    final_score = heuristic_score(seq)
            else:
                final_score = heuristic_score(seq)

    except Exception as e:
        print("Predict fallback error:", e)
        final_score = heuristic_score(seq)
        model_used = "heuristic"

    final_score = float(max(0.0, min(100.0, final_score)))

    cls = classify(final_score)

    resp = {
        "nodeId": payload.nodeId,
        "riskScore": round(final_score, 4),
        "binary": cls["binary"],
        "ternary": cls["ternary"],
        "four_class": cls["four_class"],
        "health_state": cls["health_state"],
        "ops_mode": cls["ops_mode"],
        "model_used": model_used,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "_timing_ms": round((time.time() - start_ts) * 1000, 2),
    }

    return resp


@app.get("/")
def hello():
    return {
        "ok": True,
        "models": {
            "base": bool(base_model),
            "meta": bool(meta_model),
            "scaler": bool(scaler),
        },
    }
