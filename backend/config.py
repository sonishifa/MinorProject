"""
Configuration for the Brain-First Model Tuning Toolkit.
All constants, paths, and adaptation parameters live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# ── API Keys ──
# config.py (add these lines near other API keys)

# ── Gemini API (replaces OpenAI) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ── Model file paths ──
EEG_MODEL_PATH = MODELS_DIR / "eeg_emotion_model.pkl"
EEG_SCALER_PATH = MODELS_DIR / "eeg_scaler.pkl"
EEG_LABEL_ENCODER_7_PATH = MODELS_DIR / "eeg_label_encoder_7.pkl"
EEG_LABEL_ENCODER_3_PATH = MODELS_DIR / "eeg_label_encoder_3.pkl"

KEYSTROKE_MODEL_PATH = MODELS_DIR / "keystroke_emotion_model.pkl"
KEYSTROKE_SCALER_PATH = MODELS_DIR / "keystroke_scaler.pkl"
KEYSTROKE_IMPUTER_PATH = MODELS_DIR / "keystroke_imputer.pkl"
KEYSTROKE_LABEL_ENCODER_5_PATH = MODELS_DIR / "keystroke_label_encoder_5.pkl"
KEYSTROKE_LABEL_ENCODER_3_PATH = MODELS_DIR / "keystroke_label_encoder_3.pkl"
KEYSTROKE_FEATURE_COLS_PATH = MODELS_DIR / "keystroke_feature_cols.pkl"

# ── Emotion → Valence/Arousal mapping ──
# Used by the fusion engine to convert discrete emotions to continuous space.
# Valence: -1 (negative) to +1 (positive)
# Arousal: -1 (calm) to +1 (excited/agitated)

EMOTION_VA_MAP = {
    # EEG 7-class emotions
    "joy":         {"valence":  0.8, "arousal":  0.7},
    "inspiration": {"valence":  0.9, "arousal":  0.8},
    "tenderness":  {"valence":  0.5, "arousal": -0.3},
    "neutral":     {"valence":  0.0, "arousal":  0.0},
    "sadness":     {"valence": -0.7, "arousal": -0.5},
    "fear":        {"valence": -0.5, "arousal":  0.6},
    "disgust":     {"valence": -0.8, "arousal":  0.3},
    # Keystroke 5-class emotions
    "happy":       {"valence":  0.8, "arousal":  0.7},
    "calm":        {"valence":  0.4, "arousal": -0.5},
    "angry":       {"valence": -0.6, "arousal":  0.9},
    "sad":         {"valence": -0.7, "arousal": -0.5},
    # Text-derived (used by text_analyzer)
    "frustration": {"valence": -0.6, "arousal":  0.8},
    "anxiety":     {"valence": -0.5, "arousal":  0.6},
    "curiosity":   {"valence":  0.3, "arousal":  0.4},
    "excitement":  {"valence":  0.9, "arousal":  0.9},
}

# ── LLM Adaptation profiles ──
# Keyed by emotional "zone" derived from valence/arousal quadrants.

ADAPTATION_PROFILES = {
    "positive_high": {
        "tone": "enthusiastic, encouraging, and expansive",
        "temperature": 0.9,
        "memory_depth": 12,
        "latency_ms": 0,
    },
    "positive_low": {
        "tone": "warm, gentle, and conversational",
        "temperature": 0.7,
        "memory_depth": 8,
        "latency_ms": 100,
    },
    "neutral": {
        "tone": "balanced, informative, and professional",
        "temperature": 0.7,
        "memory_depth": 7,
        "latency_ms": 0,
    },
    "negative_high": {
        "tone": "calm, direct, and solution-focused. Avoid jargon. Acknowledge difficulty briefly then provide a clear fix",
        "temperature": 0.3,
        "memory_depth": 3,
        "latency_ms": 300,
    },
    "negative_low": {
        "tone": "supportive, gentle, and kind. Use shorter sentences. Be present without being overly cheerful",
        "temperature": 0.5,
        "memory_depth": 4,
        "latency_ms": 400,
    },
}

# ── Fusion: Dynamic Confidence ──
# No fixed per-source weights — each source computes its own
# reliability score per-message based on data quality.
# Effective weight = dynamic_confidence / sum(all_confidences)

# ── Fusion: Dual-Scale Temporal Smoothing ──
# Two EMA timescales to balance reactivity vs stability:
#   Message-scale (fast): tracks moment-to-moment reactions
#   Session-scale (slow): tracks overall conversation mood
MESSAGE_ALPHA = 0.4    # Fast EMA — reacts to individual messages
SESSION_ALPHA = 0.1    # Slow EMA — captures session-level mood
SCALE_BLEND = 0.6      # 60% message-scale + 40% session-scale

# Signals below this confidence are discarded entirely
CONFIDENCE_THRESHOLD = 0.3

