"""
Facial Expression Engine — Brain First Model Tuning Toolkit
Uses DeepFace pretrained model. No training needed.
Returns signals in the same format as keystroke engine
so fusion.py needs zero changes.
"""
import base64
import numpy as np

# Lazy import — DeepFace loads slowly, do it once on first call
_deepface = None

def _get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace
        _deepface = DeepFace
    return _deepface


# DeepFace 7 emotions → your 4 classes → valence/arousal
EMOTION_MAP = {
    'happy':    ('positive_high',  0.7,  0.6),
    'surprise': ('positive_high',  0.5,  0.7),
    'neutral':  ('neutral',        0.0,  0.0),
    'sad':      ('negative_low',  -0.5, -0.4),
    'fear':     ('negative_low',  -0.4, -0.3),
    'angry':    ('negative_high', -0.6,  0.8),
    'disgust':  ('negative_high', -0.5,  0.6),
}


def _decode_frame(b64_string: str):
    """Decode base64 JPEG from browser canvas → OpenCV BGR numpy array."""
    try:
        import cv2
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_bytes = base64.b64decode(b64_string)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def _preprocess(frame):
    """CLAHE lighting normalization — handles dark rooms and bright windows."""
    try:
        import cv2
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return frame


def _analyze_frame(frame):
    """
    Analyze one frame. Returns (zone, valence, arousal, confidence)
    or neutral fallback on any failure.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return 'neutral', 0.0, 0.0, 0.0
    try:
        frame = _preprocess(frame)
        result = _get_deepface().analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        emotions   = result[0]['emotion']           # {emotion: score 0-100}
        dominant   = max(emotions, key=emotions.get)
        confidence = emotions[dominant] / 100.0

        if confidence < 0.40:                       # ambiguous — no clear face
            return 'neutral', 0.0, 0.0, confidence * 0.3

        zone, valence, arousal = EMOTION_MAP.get(
            dominant, ('neutral', 0.0, 0.0)
        )
        return zone, valence, arousal, confidence

    except Exception:
        return 'neutral', 0.0, 0.0, 0.0


def predict_facial_emotion(b64_frames: list) -> dict:
    """
    Main entry point called from main.py.

    b64_frames: list of base64 JPEG strings captured from browser webcam.

    Returns a signal dict in the SAME FORMAT as keystroke engine
    so fusion.py.fuse() works with zero changes:
      {
        source, emotion, zone, valence, arousal,
        confidence, dynamic_confidence (set by fusion),
        n_frames, camera_active
      }
    """
    if not b64_frames:
        return _neutral_signal(camera_active=False)

    results = []
    for b64 in b64_frames:
        frame = _decode_frame(b64)
        if frame is not None:
            results.append(_analyze_frame(frame))

    # Keep only confident reads
    valid = [r for r in results if r[3] > 0.30]

    if not valid:
        return _neutral_signal(camera_active=True)

    # Weighted average by confidence
    total_conf = sum(r[3] for r in valid)
    avg_v      = sum(r[1] * r[3] for r in valid) / total_conf
    avg_a      = sum(r[2] * r[3] for r in valid) / total_conf
    avg_conf   = total_conf / len(valid)

    # Dominant zone by weighted vote
    from collections import Counter
    votes = Counter()
    for zone, v, a, c in valid:
        votes[zone] += c
    dominant_zone = votes.most_common(1)[0][0]

    # Map zone back to simple emotion label for display
    zone_to_emotion = {
        'positive_high': 'positive',
        'positive_low':  'positive',
        'negative_high': 'angry',
        'negative_low':  'sad',
        'neutral':       'neutral',
    }

    return {
        'source':        'facial',
        'emotion':       zone_to_emotion.get(dominant_zone, 'neutral'),
        'zone':          dominant_zone,
        'valence':       round(float(avg_v), 3),
        'arousal':       round(float(avg_a), 3),
        'confidence':    round(float(avg_conf), 3),
        'n_frames':      len(valid),
        'camera_active': True,
    }


def _neutral_signal(camera_active: bool) -> dict:
    return {
        'source':        'facial',
        'emotion':       'neutral',
        'zone':          'neutral',
        'valence':       0.0,
        'arousal':       0.0,
        'confidence':    0.0,
        'n_frames':      0,
        'camera_active': camera_active,
    }


def warmup():
    """Pre-load DeepFace weights at server startup. First call takes 3-5s."""
    try:
        print("  Warming up DeepFace model...")
        dummy = np.zeros((240, 320, 3), dtype=np.uint8)
        _get_deepface().analyze(
            dummy,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        print("   DeepFace ready")
    except Exception as e:
        print(f"   DeepFace warmup failed: {e} — facial detection disabled")