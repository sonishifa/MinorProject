"""
Multimodal Fusion Engine — merges emotion signals from text, keystroke, and EEG
into a unified emotional state.

Architecture (EmotionTFN-inspired):
  1. Dynamic Confidence — each source computes its own reliability score
     based on data quality (message length, keystroke count, signal quality).
     No fixed per-source weights.
  2. Confidence-Weighted Average — fuse in valence-arousal space using
     normalized dynamic weights.
  3. Dual-Scale Temporal Smoothing — two EMA timescales prevent jitter
     while still reacting to genuine mood shifts.
"""
from config import (
    MESSAGE_ALPHA, SESSION_ALPHA, SCALE_BLEND,
    CONFIDENCE_THRESHOLD,
)


class FusionEngine:
    """
    Fuses multiple emotion detection signals into a single stable emotional state.

    Key improvements over fixed-weight fusion:
    - Weights auto-adjust based on data quality per-message
    - Two-timescale smoothing prevents jitter while reacting to real shifts
    - Transparent: returns effective weights so the frontend can show them
    """

    def __init__(self):
        # Message-scale EMA state (fast, α=0.4)
        self.msg_valence = 0.0
        self.msg_arousal = 0.0

        # Session-scale EMA state (slow, α=0.1)
        self.ses_valence = 0.0
        self.ses_arousal = 0.0

        # Final blended state
        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_emotion = "neutral"

        self.history = []

    # ══════════════════════════════════════════════════
    # STEP 1: Dynamic Confidence Estimation
    # ══════════════════════════════════════════════════

    def _compute_dynamic_confidence(self, signal: dict) -> float:
        """
        Compute how reliable this signal is RIGHT NOW based on data quality.

        Instead of fixed weights (text=0.5, ks=0.35, eeg=0.15), each source
        computes its own reliability from the actual data this message.
        """
        source = signal.get("source", "").split("_")[0]
        model_conf = signal.get("confidence", 0.5)

        if source == "text":
            # Text confidence scales with message length.
            # A 1-word message like "ok" is less informative than
            # "I've been stuck on this bug for 3 hours and nothing works"
            msg_len = signal.get("message_length", 20)
            length_factor = min(msg_len / 20.0, 1.0)  # peaks at 20 chars
            return model_conf * length_factor

        elif source == "keystroke":
            # More keystrokes = more reliable statistics.
            # GB model needs ~50+ keystrokes for stable quartile features.
            n_keys = signal.get("n_keystrokes", 0)
            key_factor = min(n_keys / 50.0, 1.0)  # peaks at 50 keys

            # Many backspaces = noisy/frustrated editing, reduce confidence
            error_ratio = signal.get("error_ratio", 0.0)
            error_penalty = 1.0 - (error_ratio * 0.3)
            error_penalty = max(error_penalty, 0.5)  # never drop below 50%

            return model_conf * key_factor * error_penalty

        elif source == "eeg":
            # EEG from the simulator is capped at 0.5 confidence
            # (it's synthetic data, not a real headset)
            is_simulated = signal.get("is_simulated", True)
            if is_simulated:
                return min(model_conf, 0.5)
            # Real hardware (future): use full confidence
            return model_conf

        # Unknown source: minimal trust
        return model_conf * 0.3

    # ══════════════════════════════════════════════════
    # MAIN FUSION
    # ══════════════════════════════════════════════════

    def fuse(self, signals: list[dict]) -> dict:
        """
        Fuse multiple emotion signals into a unified state.

        Pipeline:
          1. Compute dynamic confidence per signal
          2. Filter by threshold
          3. Confidence-weighted average in VA space
          4. Dual-scale EMA smoothing
          5. Zone classification for LLM adaptation

        Args:
            signals: list of dicts from detector engines, each with:
                - valence, arousal, confidence, source, emotion
                - text: also message_length
                - keystroke: also n_keystrokes, error_ratio
                - eeg: also is_simulated

        Returns:
            dict with valence, arousal, emotion, zone, signals, weights
        """
        if not signals:
            return self._current_state()

        # ── Step 1: Dynamic Confidence ──
        enriched = []
        for s in signals:
            dyn_conf = self._compute_dynamic_confidence(s)
            enriched.append({**s, "dynamic_confidence": dyn_conf})

        # ── Step 2: Filter by threshold ──
        valid = [s for s in enriched if s["dynamic_confidence"] >= CONFIDENCE_THRESHOLD]

        if not valid:
            return self._current_state()

        # ── Step 3: Confidence-weighted average in VA space ──
        total_weight = sum(s["dynamic_confidence"] for s in valid)
        raw_valence = sum(s["dynamic_confidence"] * s["valence"] for s in valid) / total_weight
        raw_arousal = sum(s["dynamic_confidence"] * s["arousal"] for s in valid) / total_weight

        # Track the most confident emotion label
        best = max(valid, key=lambda s: s["dynamic_confidence"])
        best_emotion = best.get("emotion", "neutral")

        # ── Step 4: Dual-Scale Temporal Smoothing ──
        # Message-scale (fast) — reacts to THIS message
        self.msg_valence = MESSAGE_ALPHA * raw_valence + (1 - MESSAGE_ALPHA) * self.msg_valence
        self.msg_arousal = MESSAGE_ALPHA * raw_arousal + (1 - MESSAGE_ALPHA) * self.msg_arousal

        # Session-scale (slow) — tracks overall conversation mood
        self.ses_valence = SESSION_ALPHA * raw_valence + (1 - SESSION_ALPHA) * self.ses_valence
        self.ses_arousal = SESSION_ALPHA * raw_arousal + (1 - SESSION_ALPHA) * self.ses_arousal

        # Blend: 60% message-scale + 40% session-scale
        self.current_valence = SCALE_BLEND * self.msg_valence + (1 - SCALE_BLEND) * self.ses_valence
        self.current_arousal = SCALE_BLEND * self.msg_arousal + (1 - SCALE_BLEND) * self.ses_arousal
        self.current_emotion = best_emotion

        # ── Step 5: Zone classification ──
        zone = self._classify_zone(self.current_valence, self.current_arousal)

        # Build effective weights for transparency
        effective_weights = {}
        for s in enriched:
            src = s.get("source", "unknown").split("_")[0]
            eff_w = s["dynamic_confidence"] / total_weight if total_weight > 0 else 0
            effective_weights[src] = round(eff_w, 3)

        state = {
            "valence": round(self.current_valence, 3),
            "arousal": round(self.current_arousal, 3),
            "emotion": self.current_emotion,
            "zone": zone,
            "signals": [
                {
                    "source": s.get("source"),
                    "emotion": s.get("emotion"),
                    "confidence": round(s.get("confidence", 0), 3),
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
        """Map valence/arousal to one of 5 adaptation zones."""
        if valence > 0.15:
            return "positive_high" if arousal > 0.0 else "positive_low"
        elif valence < -0.15:
            return "negative_high" if arousal > 0.0 else "negative_low"
        else:
            return "neutral"

    def _current_state(self) -> dict:
        zone = self._classify_zone(self.current_valence, self.current_arousal)
        return {
            "valence": round(self.current_valence, 3),
            "arousal": round(self.current_arousal, 3),
            "emotion": self.current_emotion,
            "zone": zone,
            "signals": [],
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
        self.msg_valence = 0.0
        self.msg_arousal = 0.0
        self.ses_valence = 0.0
        self.ses_arousal = 0.0
        self.current_valence = 0.0
        self.current_arousal = 0.0
        self.current_emotion = "neutral"
        self.history = []


# Global instance
fusion_engine = FusionEngine()
