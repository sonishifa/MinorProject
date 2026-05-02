"""
Keystroke Emotion Engine — CNN-LSTM Deep Learning Model (4-class)

Loads the trained CNN-LSTM model from Kaggle and runs inference on
raw keystroke events from the frontend.

4 classes: positive | angry | sad | neutral

Input: list of keystroke events with keyCode, keyDown, keyUp (seconds)
Output: dict with emotion, valence, arousal, confidence, source
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
from config import (
    KEYSTROKE_DEPLOY_MODEL_PATH,
    KEYSTROKE_SEQ_MEAN_PATH, KEYSTROKE_SEQ_STD_PATH,
    KEYSTROKE_STAT_MEAN_PATH, KEYSTROKE_STAT_STD_PATH,
    KEYSTROKE_LABEL_ENCODER_PATH,
    KEYSTROKE_USER_BASELINES_PATH,
    EMOTION_VA_MAP,
    MAX_SEQ_LEN, SEQ_FEAT_DIM, STAT_DIM, N_CLASSES,
)

TIMING_COLS = ['D1U1', 'D1U2', 'D1D2', 'U1D2', 'U1U2', 'D1U3', 'D1D3']


# ══════════════════════════════════════════════════════════════════════════
# CNN-LSTM Architecture — must exactly match training notebook
# ══════════════════════════════════════════════════════════════════════════

class CNN_LSTM(nn.Module):
    def __init__(self, feat_dim=SEQ_FEAT_DIM, stat_dim=STAT_DIM,
                 n_classes=N_CLASSES, drop=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(feat_dim, 64, 3, padding=1), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(drop * 0.5),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2), nn.Dropout(drop * 0.5),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.lstm1 = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.drop  = nn.Dropout(drop)
        self.lstm2 = nn.LSTM(256, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn  = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head  = nn.Sequential(
            nn.Linear(256 + stat_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(drop), nn.Linear(128, n_classes)
        )

    def forward(self, x, mask=None, stat=None):
        out = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm1(out)
        out = self.drop(out)
        out, _ = self.lstm2(out)
        scores = self.attn(out)
        if mask is not None:
            pool_mask = mask[:, ::2].unsqueeze(-1)
            scores = scores.masked_fill(~pool_mask, float('-inf'))
        ctx = (torch.softmax(scores, dim=1) * out).sum(1)
        if stat is not None:
            ctx = torch.cat([ctx, stat], dim=1)
        return self.head(ctx)


# ══════════════════════════════════════════════════════════════════════════
# Key-type one-hot encoding (must match training)
# ══════════════════════════════════════════════════════════════════════════

def keycode_to_type(keycode):
    try:
        kc = int(float(keycode))
    except (ValueError, TypeError):
        return 3
    if (65 <= kc <= 90) or (97 <= kc <= 122): return 0  # alpha
    if 48 <= kc <= 57 or 96 <= kc <= 105:     return 1  # digit
    if kc in (32, 188, 190, 186, 222, 219, 221, 220, 191, 192, 189, 187):
        return 2                                          # space/punct
    return 3                                              # control/bksp

def keycode_to_onehot(keycode):
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[keycode_to_type(keycode)] = 1.0
    return oh


# ══════════════════════════════════════════════════════════════════════════
# Load model artifacts — all at module level, never per-request
# ══════════════════════════════════════════════════════════════════════════

KS_AVAILABLE      = False
ks_model          = None
ks_label_encoder  = None
ks_user_baselines = {}
_GLOBAL_BASELINE  = np.zeros(7, dtype=np.float32)
_SEQ_MEAN = _SEQ_STD = _STAT_MEAN = _STAT_STD = None

try:
    _SEQ_MEAN  = np.load(str(KEYSTROKE_SEQ_MEAN_PATH))
    _SEQ_STD   = np.load(str(KEYSTROKE_SEQ_STD_PATH))
    _STAT_MEAN = np.load(str(KEYSTROKE_STAT_MEAN_PATH))
    _STAT_STD  = np.load(str(KEYSTROKE_STAT_STD_PATH))

    ks_label_encoder  = joblib.load(str(KEYSTROKE_LABEL_ENCODER_PATH))
    ks_user_baselines = joblib.load(str(KEYSTROKE_USER_BASELINES_PATH))

    # Global baseline = mean of all training users' personal baselines.
    # Used for users not in the training set so their timing features
    # stay in the same deviation range (~±30ms) as training data.
    # Falling back to zeros would feed raw absolute values (~100ms) into
    # a model trained on deviations — guaranteed OOD stats.
    if ks_user_baselines:
        _GLOBAL_BASELINE = np.array(
            list(ks_user_baselines.values()), dtype=np.float32
        ).mean(0)

    ks_model = CNN_LSTM()
    state_dict = torch.load(str(KEYSTROKE_DEPLOY_MODEL_PATH), map_location='cpu')
    ks_model.load_state_dict(state_dict)
    ks_model.eval()

    KS_AVAILABLE = True
    print(f"[KeystrokeEngine] CNN-LSTM loaded — classes: {list(ks_label_encoder.classes_)}")
    print(f"[KeystrokeEngine] Global baseline from {len(ks_user_baselines)} training users")

except Exception as e:
    print(f"[KeystrokeEngine] Model not available: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Background Calibration (Learning user's pattern over time)
# ══════════════════════════════════════════════════════════════════════════

def update_rolling_baseline(uid: str, timing_rows: list, alpha: float = 0.15):
    """
    Incrementally updates the user's personal baseline using an Exponential Moving Average (EMA).
    This allows the system to slowly 'learn' the user's natural typing speed in the background.
    """
    if uid not in ks_user_baselines:
        ks_user_baselines[uid] = _GLOBAL_BASELINE.copy()

    if not timing_rows:
        return

    batch_timings = []
    for row in timing_rows:
        batch_timings.append([row.get(col, 0.0) for col in TIMING_COLS])
    
    batch_timings = np.array(batch_timings, dtype=np.float32)
    batch_median = np.median(batch_timings, axis=0)

    current_baseline = ks_user_baselines[uid]
    ks_user_baselines[uid] = (1.0 - alpha) * current_baseline + alpha * batch_median


# ══════════════════════════════════════════════════════════════════════════
# Timing feature computation
# ══════════════════════════════════════════════════════════════════════════

def compute_timing_features(events: list) -> list:
    """
    Convert raw keystroke events into per-keystroke timing rows.

    Frontend sends keyDown/keyUp in SECONDS (normalised from first keydown).
    Training data uses MILLISECONDS — multiply by 1000 here.
    """
    rows = []
    for i, evt in enumerate(events):
        kd = evt.get("keyDown", 0) * 1000.0
        ku = evt.get("keyUp",   0) * 1000.0
        kc = evt.get("keyCode", 0)

        row = {"keyCode": kc, "textType": "free", "D1U1": ku - kd}

        if i > 0:
            prev_kd = events[i-1].get("keyDown", 0) * 1000.0
            prev_ku = events[i-1].get("keyUp",   0) * 1000.0
            row["D1D2"] = kd - prev_kd
            row["D1U2"] = ku - prev_kd
            row["U1D2"] = kd - prev_ku
            row["U1U2"] = ku - prev_ku
        else:
            row["D1D2"] = row["D1U2"] = row["U1D2"] = row["U1U2"] = 0.0

        if i > 1:
            prev2_kd   = events[i-2].get("keyDown", 0) * 1000.0
            row["D1U3"] = ku - prev2_kd
            row["D1D3"] = kd - prev2_kd
        else:
            row["D1U3"] = row["D1D3"] = 0.0

        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Inference sequence builder (matches training notebook exactly)
# ══════════════════════════════════════════════════════════════════════════

def build_inference_sequence(timing_rows: list, uid: str = "default") -> tuple:
    """
    Returns:
        seq  (1, 100, 12) float32 — normalised calibrated padded sequence
        mask (1, 100)     bool    — True=real key, False=padding
        stat (1, 21)      float32 — mean/std/median of timing cols
    """
    baseline = ks_user_baselines.get(uid, _GLOBAL_BASELINE)

    rows = []
    for ev in timing_rows:
        row = [float(ev.get(col, 0.0)) - float(baseline[i])
               for i, col in enumerate(TIMING_COLS)]
        row.extend(keycode_to_onehot(ev.get("keyCode", 0)))
        row.append(1.0 if ev.get("textType", "free") == "free" else 0.0)
        rows.append(row)

    seq  = np.nan_to_num(np.clip(np.array(rows, np.float32), -500, 500))
    n    = seq.shape[0]
    mask = np.zeros(MAX_SEQ_LEN, dtype=bool)
    mask[:min(n, MAX_SEQ_LEN)] = True

    if n >= MAX_SEQ_LEN:
        seq = seq[:MAX_SEQ_LEN]
    else:
        seq = np.vstack([seq, np.zeros((MAX_SEQ_LEN - n, SEQ_FEAT_DIM), np.float32)])

    # Stat features from raw calibrated values — BEFORE normalisation.
    # _STAT_MEAN/_STAT_STD were fitted on pre-norm values; computing
    # stats after normalisation would produce a distribution mismatch.
    real_timing = seq[mask, :7]
    stat = np.zeros(STAT_DIM, dtype=np.float32)
    if len(real_timing) > 0:
        stat[:7]    = real_timing.mean(0)
        stat[7:14]  = real_timing.std(0)
        stat[14:21] = np.median(real_timing, axis=0)

    seq[mask] = (seq[mask] - _SEQ_MEAN) / (_SEQ_STD  + 1e-8)
    stat      = (stat      - _STAT_MEAN) / (_STAT_STD + 1e-8)

    return seq[np.newaxis], mask[np.newaxis], stat[np.newaxis]


# ══════════════════════════════════════════════════════════════════════════
# Main prediction entry point
# ══════════════════════════════════════════════════════════════════════════

def predict_keystroke_emotion(events: list, uid: str = "live_user", update_baseline: bool = True) -> dict:
    """
    Predict emotion from raw keystroke events sent by the frontend.

    Args:
        events: list of dicts, each with keyCode (int),
                keyDown and keyUp (float, seconds from first keydown)

    Returns:
        signal dict compatible with fusion.py
    """
    if not KS_AVAILABLE:
        return _neutral_result("keystroke_unavailable")

    valid = [e for e in events if e.get("keyUp") is not None]
    if len(valid) < 5:
        return _neutral_result("keystroke_insufficient_data")

    timing_rows = compute_timing_features(valid)
    
    if update_baseline:
        update_rolling_baseline(uid, timing_rows)

    seq, mask, stat = build_inference_sequence(timing_rows, uid)

    with torch.no_grad():
        logits = ks_model(
            torch.tensor(seq,  dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(stat, dtype=torch.float32),
        )

    probs      = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_idx   = int(probs.argmax())
    emotion    = ks_label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    va           = EMOTION_VA_MAP.get(emotion, {"valence": 0.0, "arousal": 0.0})
    delete_count = sum(1 for e in valid if e.get("keyCode") in [8, 46])

    return {
        "emotion":      emotion,
        "valence":      va["valence"],
        "arousal":      va["arousal"],
        "confidence":   confidence,
        "source":       "keystroke",
        "n_keystrokes": len(valid),
        "error_ratio":  delete_count / len(valid),
    }


def _neutral_result(source: str = "keystroke") -> dict:
    return {
        "emotion": "neutral", "valence": 0.0, "arousal": 0.0,
        "confidence": 0.2, "source": source,
        "n_keystrokes": 0, "error_ratio": 0.0,
    }