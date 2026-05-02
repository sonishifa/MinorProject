"""
Multimodal Fusion Engine — merges emotion signals from text, keystroke,
and facial into a unified emotional state.
"""
from config import (
    MESSAGE_ALPHA, SESSION_ALPHA, SCALE_BLEND,
    CONFIDENCE_THRESHOLD,
)


class FusionEngine:
    def __init__(self):
        self.msg_valence     = 0.0
        self.msg_arousal     = 0.0
        self.ses_valence     = 0.0
        self.ses_arousal     = 0.0
        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_emotion = "neutral"
        self.history         = []

    def _compute_dynamic_confidence(self, signal: dict) -> float:
        source     = signal.get("source", "").split("_")[0]
        model_conf = signal.get("confidence", 0.5)

        if source == "text":
            msg_len       = signal.get("message_length", 20)
            length_factor = min(msg_len / 20.0, 1.0)
            return model_conf * length_factor

        elif source == "keystroke":
            n_keys        = signal.get("n_keystrokes", 0)
            key_factor    = min(n_keys / 50.0, 1.0)
            error_ratio   = signal.get("error_ratio", 0.0)
            error_penalty = max(1.0 - (error_ratio * 0.3), 0.5)
            return model_conf * key_factor * error_penalty

        elif source == "facial":
            # Facial confidence scaled by number of frames analyzed
            # More frames = more stable read
            n_frames      = signal.get("n_frames", 1)
            frame_factor  = min(n_frames / 3.0, 1.0)   # peaks at 3 frames
            return model_conf * frame_factor

        return model_conf * 0.3

    def fuse(self, signals: list) -> dict:
        if not signals:
            return self._current_state()

        # Step 1: Dynamic confidence per signal
        enriched = []
        for s in signals:
            dyn_conf = self._compute_dynamic_confidence(s)
            enriched.append({**s, "dynamic_confidence": dyn_conf})

        # Step 2: Filter by threshold
        valid = [s for s in enriched if s["dynamic_confidence"] >= CONFIDENCE_THRESHOLD]
        if not valid:
            return self._current_state()

        # Step 3: Confidence-weighted average in VA space
        total_weight = sum(s["dynamic_confidence"] for s in valid)
        raw_valence  = sum(s["dynamic_confidence"] * s["valence"] for s in valid) / total_weight
        raw_arousal  = sum(s["dynamic_confidence"] * s["arousal"] for s in valid) / total_weight
        best_emotion = max(valid, key=lambda s: s["dynamic_confidence"]).get("emotion", "neutral")

        # Step 4: Dual-scale EMA smoothing
        self.msg_valence = MESSAGE_ALPHA * raw_valence + (1 - MESSAGE_ALPHA) * self.msg_valence
        self.msg_arousal = MESSAGE_ALPHA * raw_arousal + (1 - MESSAGE_ALPHA) * self.msg_arousal
        self.ses_valence = SESSION_ALPHA * raw_valence + (1 - SESSION_ALPHA) * self.ses_valence
        self.ses_arousal = SESSION_ALPHA * raw_arousal + (1 - SESSION_ALPHA) * self.ses_arousal

        self.current_valence = SCALE_BLEND * self.msg_valence + (1 - SCALE_BLEND) * self.ses_valence
        self.current_arousal = SCALE_BLEND * self.msg_arousal + (1 - SCALE_BLEND) * self.ses_arousal
        self.current_emotion = best_emotion

        # Step 5: Zone classification
        zone = self._classify_zone(self.current_valence, self.current_arousal)

        effective_weights = {}
        for s in enriched:
            src   = s.get("source", "unknown").split("_")[0]
            eff_w = s["dynamic_confidence"] / total_weight if total_weight > 0 else 0
            effective_weights[src] = round(eff_w, 3)

        state = {
            "valence":  round(self.current_valence, 3),
            "arousal":  round(self.current_arousal, 3),
            "emotion":  self.current_emotion,
            "zone":     zone,
            "signals":  [
                {
                    "source":             s.get("source"),
                    "emotion":            s.get("emotion"),
                    "confidence":         round(s.get("confidence", 0), 3),
                    "dynamic_confidence": round(s.get("dynamic_confidence", 0), 3),
                }
                for s in enriched
            ],
            "effective_weights": effective_weights,
            "smoothing": {
                "message_scale": {
                    "valence": round(self.msg_valence, 3),
                    "arousal": round(self.msg_arousal, 3),
                },
                "session_scale": {
                    "valence": round(self.ses_valence, 3),
                    "arousal": round(self.ses_arousal, 3),
                },
            },
        }

        self.history.append(state)
        if len(self.history) > 100:
            self.history = self.history[-100:]

        return state

    def _classify_zone(self, valence: float, arousal: float) -> str:
        if valence > 0.15:
            return "positive_high" if arousal > 0.0 else "positive_low"
        elif valence < -0.15:
            return "negative_high" if arousal > 0.0 else "negative_low"
        else:
            return "neutral"

    def _current_state(self) -> dict:
        zone = self._classify_zone(self.current_valence, self.current_arousal)
        return {
            "valence":  round(self.current_valence, 3),
            "arousal":  round(self.current_arousal, 3),
            "emotion":  self.current_emotion,
            "zone":     zone,
            "signals":  [],
            "effective_weights": {},
            "smoothing": {
                "message_scale": {
                    "valence": round(self.msg_valence, 3),
                    "arousal": round(self.msg_arousal, 3),
                },
                "session_scale": {
                    "valence": round(self.ses_valence, 3),
                    "arousal": round(self.ses_arousal, 3),
                },
            },
        }

    def reset(self):
        self.msg_valence     = 0.0
        self.msg_arousal     = 0.0
        self.ses_valence     = 0.0
        self.ses_arousal     = 0.0
        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_emotion = "neutral"
        self.history         = []


fusion_engine = FusionEngine()