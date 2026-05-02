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

# ── Gemini API ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# ── Keystroke DL model paths (CNN-LSTM, 4-class) ──
KEYSTROKE_DEPLOY_MODEL_PATH   = MODELS_DIR / "keystroke_deploy_model.pt"
KEYSTROKE_SEQ_MEAN_PATH       = MODELS_DIR / "keystroke_seq_mean.npy"
KEYSTROKE_SEQ_STD_PATH        = MODELS_DIR / "keystroke_seq_std.npy"
KEYSTROKE_STAT_MEAN_PATH      = MODELS_DIR / "keystroke_stat_mean.npy"
KEYSTROKE_STAT_STD_PATH       = MODELS_DIR / "keystroke_stat_std.npy"
KEYSTROKE_LABEL_ENCODER_PATH  = MODELS_DIR / "keystroke_label_encoder.pkl"
KEYSTROKE_USER_BASELINES_PATH = MODELS_DIR / "keystroke_user_baselines.pkl"

# ── Keystroke model constants (must match training) ──
MAX_SEQ_LEN  = 100
SEQ_FEAT_DIM = 12     # 7 timing + 4 key_type_onehot + 1 textType_flag
STAT_DIM     = 21     # mean + std + median for 7 timing cols
N_CLASSES    = 4

# ── Emotion → Valence/Arousal mapping ──
# Used by the fusion engine to convert discrete emotions to continuous space.
# Valence: -1 (negative) to +1 (positive)
# Arousal: -1 (calm) to +1 (excited/agitated)

EMOTION_VA_MAP = {
    # Keystroke 4-class emotions (CNN-LSTM model)
    "positive":    {"valence":  0.7, "arousal":  0.6},
    "angry":       {"valence": -0.6, "arousal":  0.9},
    "sad":         {"valence": -0.7, "arousal": -0.5},
    "neutral":     {"valence":  0.0, "arousal":  0.0},
    # Facial expression emotions (DeepFace)
    "happy":       {"valence":  0.8, "arousal":  0.7},
    # Text-derived (used by text_analyzer / Gemini classifier)
    "frustration": {"valence": -0.6, "arousal":  0.8},
    "anxiety":     {"valence": -0.5, "arousal":  0.6},
    "curiosity":   {"valence":  0.3, "arousal":  0.4},
    "excitement":  {"valence":  0.9, "arousal":  0.9},
    "joy":         {"valence":  0.8, "arousal":  0.7},
    "sadness":     {"valence": -0.7, "arousal": -0.5},
    "fear":        {"valence": -0.5, "arousal":  0.6},
    "disgust":     {"valence": -0.8, "arousal":  0.3},
    "anger":       {"valence": -0.6, "arousal":  0.9},
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
