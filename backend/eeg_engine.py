"""
EEG Emotion Engine — classifies emotion from EEG feature vectors.

Loads the trained Random Forest model from Stage 1 (Kaggle).
Accepts a pre-computed feature vector (from the EEG simulator or live data)
and returns an emotion prediction with confidence.
"""
import numpy as np
import joblib
from config import (
    EEG_MODEL_PATH, EEG_SCALER_PATH,
    EEG_LABEL_ENCODER_7_PATH, EEG_LABEL_ENCODER_3_PATH,
    EMOTION_VA_MAP,
)

# ── Load model artifacts ──
try:
    eeg_model = joblib.load(EEG_MODEL_PATH)
    eeg_scaler = joblib.load(EEG_SCALER_PATH)
    eeg_le_7 = joblib.load(EEG_LABEL_ENCODER_7_PATH)
    eeg_le_3 = joblib.load(EEG_LABEL_ENCODER_3_PATH)
    EEG_AVAILABLE = True
    print(f"[EEGEngine] Loaded model — 7-class labels: {list(eeg_le_7.classes_)}")
except Exception as e:
    print(f"[EEGEngine] Model not available: {e}")
    EEG_AVAILABLE = False


def predict_eeg_emotion(features: np.ndarray) -> dict:
    """
    Predict emotion from an EEG feature vector.

    Args:
        features: 1D numpy array of EEG features (must match training dimensions)

    Returns:
        dict with keys: emotion, valence, arousal, confidence, source, probabilities
    """
    if not EEG_AVAILABLE:
        return _neutral_result("eeg_unavailable")

    try:
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = eeg_scaler.transform(X)

        prediction = eeg_model.predict(X)[0]
        emotion = eeg_le_7.inverse_transform([prediction])[0]

        # Confidence from probability estimates
        if hasattr(eeg_model, "predict_proba"):
            probas = eeg_model.predict_proba(X)[0]
            confidence = float(np.max(probas))
            # Build full probability dict
            prob_dict = {eeg_le_7.inverse_transform([i])[0]: float(p)
                         for i, p in enumerate(probas)}
        else:
            confidence = 0.4
            prob_dict = {}

        va = EMOTION_VA_MAP.get(emotion, {"valence": 0.0, "arousal": 0.0})

        return {
            "emotion": emotion,
            "valence": va["valence"],
            "arousal": va["arousal"],
            "confidence": confidence,
            "source": "eeg",
            "probabilities": prob_dict,
            "is_simulated": True,  # Flag for dynamic confidence (capped at 0.5)
        }

    except Exception as e:
        print(f"[EEGEngine] Prediction error: {e}")
        return _neutral_result("eeg_error")


def _neutral_result(source: str = "eeg") -> dict:
    return {
        "emotion": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "confidence": 0.2,
        "source": source,
        "probabilities": {},
        "is_simulated": True,
    }
