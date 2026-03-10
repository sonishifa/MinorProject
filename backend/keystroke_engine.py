"""
Keystroke Emotion Engine — detects emotion from HOW the user types.

Loads the trained Gradient Boosting model from Stage 2 (Kaggle).
Accepts raw keystroke timing events from the frontend (keydown/keyup timestamps),
computes the same aggregate features used during training, and predicts emotion.
"""
import numpy as np
import joblib
from config import (
    KEYSTROKE_MODEL_PATH, KEYSTROKE_SCALER_PATH,
    KEYSTROKE_IMPUTER_PATH, KEYSTROKE_LABEL_ENCODER_5_PATH,
    KEYSTROKE_FEATURE_COLS_PATH, EMOTION_VA_MAP,
)

# ── Load model artifacts ──
try:
    ks_model = joblib.load(KEYSTROKE_MODEL_PATH)
    ks_scaler = joblib.load(KEYSTROKE_SCALER_PATH)
    ks_imputer = joblib.load(KEYSTROKE_IMPUTER_PATH)
    ks_label_encoder = joblib.load(KEYSTROKE_LABEL_ENCODER_5_PATH)
    ks_feature_cols = joblib.load(KEYSTROKE_FEATURE_COLS_PATH)
    KS_AVAILABLE = True
    print(f"[KeystrokeEngine] Loaded model with {len(ks_feature_cols)} features, "
          f"classes: {list(ks_label_encoder.classes_)}")
except Exception as e:
    print(f"[KeystrokeEngine] Model not available: {e}")
    KS_AVAILABLE = False


# ── The timing feature names we compute (must match training) ──
TIMING_NAMES = ["D1U1", "D1U2", "D1D2", "U1D2", "U1U2", "D1U3", "D1D3"]


def compute_keystroke_features(events: list[dict]) -> dict | None:
    """
    Compute aggregate features from raw keystroke events.

    Args:
        events: list of dicts, each with keys:
            - keyCode: int
            - keyDown: float (timestamp in ms)
            - keyUp: float (timestamp in ms)

    Returns:
        dict of features matching training format, or None if insufficient data.
    """
    if len(events) < 5:
        return None

    # ── Compute timing features from raw events ──
    # D1U1 = key hold duration (keyUp - keyDown for same key)
    # D1D2 = time between consecutive keyDowns
    # U1D2 = time between keyUp of key1 and keyDown of key2 (flight time)
    # D1U2 = time between keyDown of key1 and keyUp of key2
    # U1U2 = time between consecutive keyUps
    # D1U3, D1D3 = trigraph timings (span 3 keys)

    timings = {name: [] for name in TIMING_NAMES}

    for i, evt in enumerate(events):
        # D1U1: dwell time (how long key is held)
        if "keyDown" in evt and "keyUp" in evt:
            timings["D1U1"].append(evt["keyUp"] - evt["keyDown"])

        # Digraph features (between consecutive keys)
        if i > 0:
            prev = events[i - 1]
            timings["D1D2"].append(evt["keyDown"] - prev["keyDown"])
            timings["D1U2"].append(evt["keyUp"] - prev["keyDown"])
            timings["U1D2"].append(evt["keyDown"] - prev["keyUp"])
            timings["U1U2"].append(evt["keyUp"] - prev["keyUp"])

        # Trigraph features (span 3 keys)
        if i > 1:
            prev2 = events[i - 2]
            timings["D1D3"].append(evt["keyDown"] - prev2["keyDown"])
            timings["D1U3"].append(evt["keyUp"] - prev2["keyDown"])

    # ── Aggregate into statistics (matching training) ──
    features = {}
    for name in TIMING_NAMES:
        data = np.array(timings[name], dtype=float)
        data = data[np.isfinite(data)]  # remove any Inf/NaN

        if len(data) < 3:
            for stat in ["mean", "std", "median", "q25", "q75", "iqr"]:
                features[f"{name}_{stat}"] = np.nan
        else:
            features[f"{name}_mean"] = float(np.mean(data))
            features[f"{name}_std"] = float(np.std(data))
            features[f"{name}_median"] = float(np.median(data))
            features[f"{name}_q25"] = float(np.percentile(data, 25))
            features[f"{name}_q75"] = float(np.percentile(data, 75))
            features[f"{name}_iqr"] = float(np.percentile(data, 75) - np.percentile(data, 25))

    # ── Derived features ──
    d1d2 = np.array(timings["D1D2"], dtype=float)
    d1d2 = d1d2[np.isfinite(d1d2)]
    d1u1 = np.array(timings["D1U1"], dtype=float)
    d1u1 = d1u1[np.isfinite(d1u1)]
    u1d2 = np.array(timings["U1D2"], dtype=float)
    u1d2 = u1d2[np.isfinite(u1d2)]

    # Typing speed
    features["typing_speed"] = float(1.0 / np.mean(d1d2)) if len(d1d2) > 0 and np.mean(d1d2) > 0 else np.nan

    # Rhythm regularity (CV of inter-key intervals)
    features["rhythm_cv"] = float(np.std(d1d2) / np.mean(d1d2)) if len(d1d2) > 2 and np.mean(d1d2) > 0 else np.nan

    # Dwell-to-flight ratio
    if len(d1u1) > 0 and len(u1d2) > 0 and abs(np.mean(u1d2)) > 1e-8:
        features["dwell_flight_ratio"] = float(np.mean(d1u1) / (abs(np.mean(u1d2)) + 1e-8))
    else:
        features["dwell_flight_ratio"] = np.nan

    # Keystroke count
    features["n_keystrokes"] = float(len(events))

    # Error-related (count delete/backspace keys: keyCodes 8 and 46)
    delete_count = sum(1 for e in events if e.get("keyCode") in [8, 46])
    features["DelFreq"] = float(delete_count / len(events)) if len(events) > 0 else 0.0
    features["LeftFreq"] = 0.0  # Backspace is already counted in DelFreq
    features["TotTime"] = float(events[-1].get("keyUp", 0) - events[0].get("keyDown", 0)) if len(events) > 1 else 0.0
    features["error_ratio"] = features["DelFreq"] + features["LeftFreq"]

    return features


def predict_keystroke_emotion(events: list[dict]) -> dict:
    """
    Predict emotion from keystroke events.

    Args:
        events: list of keystroke event dicts from the frontend

    Returns:
        dict with keys: emotion, valence, arousal, confidence, source
    """
    if not KS_AVAILABLE:
        return _neutral_result("keystroke_unavailable")

    features = compute_keystroke_features(events)
    if features is None:
        return _neutral_result("keystroke_insufficient_data")

    # ── Build feature vector in the same order as training ──
    feature_vector = []
    for col in ks_feature_cols:
        feature_vector.append(features.get(col, np.nan))

    X = np.array([feature_vector], dtype=np.float64)

    # Impute missing values (same as training)
    X = ks_imputer.transform(X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    X = ks_scaler.transform(X)

    # Predict
    prediction = ks_model.predict(X)[0]
    emotion = ks_label_encoder.inverse_transform([prediction])[0]

    # Get confidence from probability estimates if available
    if hasattr(ks_model, "predict_proba"):
        probas = ks_model.predict_proba(X)[0]
        confidence = float(np.max(probas))
    else:
        confidence = 0.5

    # Map to valence/arousal
    va = EMOTION_VA_MAP.get(emotion, {"valence": 0.0, "arousal": 0.0})

    return {
        "emotion": emotion,
        "valence": va["valence"],
        "arousal": va["arousal"],
        "confidence": confidence,
        "source": "keystroke",
        # Data quality metadata for dynamic confidence
        "n_keystrokes": len(events),
        "error_ratio": features.get("error_ratio", 0.0),
    }


def _neutral_result(source: str = "keystroke") -> dict:
    return {
        "emotion": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "confidence": 0.2,
        "source": source,
        "n_keystrokes": 0,
        "error_ratio": 0.0,
    }
