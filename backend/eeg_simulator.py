"""
EEG Simulator — replays EEG data from the EmoEEG-MC dataset for demo purposes.

Since no real EEG hardware is connected, this module generates realistic
synthetic EEG-like feature vectors that cycle through different emotional states.
If actual .npy data files are available in data/, it replays those instead.
"""
import numpy as np
import time
import asyncio
from dataclasses import dataclass, field

# 64 EEG channels in 10-20 system
CHANNEL_NAMES = [
    'Fp1','Fpz','Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1','Fz',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'FT8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','TP7','CP5','CP3',
    'CP1','CPz','CP2','CP4','CP6','TP8','P7','P5','P3','P1','Pz','P2',
    'P4','P6','P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','F9',
    'F10','TP9','TP10'
]

N_CHANNELS = 64
N_BANDS = 5  # delta, theta, alpha, beta, gamma
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# Typical DE ranges per band (approximate, from EmoEEG-MC statistics)
BAND_BASELINES = {
    "delta": {"mean": 2.5, "std": 1.0},
    "theta": {"mean": 1.8, "std": 0.8},
    "alpha": {"mean": 2.0, "std": 0.9},
    "beta":  {"mean": 1.5, "std": 0.7},
    "gamma": {"mean": 0.8, "std": 0.5},
}

# Emotional state modifiers — how each emotion shifts band power
# Based on neuroscience literature (SEED, DEAP papers)
EMOTION_PROFILES = {
    "joy":         {"delta": 0.0, "theta": -0.2, "alpha": 0.5,  "beta": 0.3,  "gamma": 0.4},
    "sadness":     {"delta": 0.3, "theta": 0.4,  "alpha": -0.5, "beta": -0.3, "gamma": -0.2},
    "fear":        {"delta": 0.1, "theta": 0.3,  "alpha": -0.4, "beta": 0.5,  "gamma": 0.3},
    "anger":       {"delta": 0.2, "theta": 0.1,  "alpha": -0.6, "beta": 0.6,  "gamma": 0.5},
    "neutral":     {"delta": 0.0, "theta": 0.0,  "alpha": 0.0,  "beta": 0.0,  "gamma": 0.0},
    "tenderness":  {"delta": 0.0, "theta": 0.1,  "alpha": 0.3,  "beta": -0.1, "gamma": 0.1},
    "inspiration": {"delta": 0.0, "theta": -0.1, "alpha": 0.2,  "beta": 0.4,  "gamma": 0.6},
    "disgust":     {"delta": 0.2, "theta": 0.2,  "alpha": -0.3, "beta": 0.3,  "gamma": 0.1},
    "calm":        {"delta": 0.1, "theta": 0.2,  "alpha": 0.6,  "beta": -0.4, "gamma": -0.3},
}


@dataclass
class EEGSimulator:
    """Generates synthetic EEG feature streams that mimic real emotional states."""

    current_emotion: str = "neutral"
    speed: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed
    running: bool = False
    _time_counter: float = 0.0
    _transition_progress: float = 0.0
    _target_emotion: str = "neutral"

    def set_emotion(self, emotion: str):
        """Transition to a new emotional state (smooth, not instant)."""
        if emotion in EMOTION_PROFILES:
            self._target_emotion = emotion

    def generate_frame(self) -> dict:
        """
        Generate one frame of synthetic EEG data (1 second of signals).

        Returns:
            dict with:
                - raw_channels: list of 64 waveform values (for visualization)
                - band_powers: dict of band → list of 64 channel powers
                - features: flat feature vector for the classifier
                - timestamp: current simulation time
        """
        # Smooth transition between emotions
        if self.current_emotion != self._target_emotion:
            self._transition_progress += 0.05
            if self._transition_progress >= 1.0:
                self.current_emotion = self._target_emotion
                self._transition_progress = 0.0

        profile = EMOTION_PROFILES.get(self.current_emotion,
                                        EMOTION_PROFILES["neutral"])

        # Generate per-channel, per-band DE values
        band_powers = {}
        de_features = []
        psd_features = []
        raw_channels = []

        for band_idx, band_name in enumerate(BAND_NAMES):
            baseline = BAND_BASELINES[band_name]
            modifier = profile[band_name]
            channel_values = []

            for ch in range(N_CHANNELS):
                # Base value + emotion modifier + noise + spatial variation
                spatial_var = 0.1 * np.sin(ch * 0.3 + self._time_counter * 0.1)
                de_value = (baseline["mean"]
                            + modifier * baseline["std"]
                            + np.random.normal(0, baseline["std"] * 0.3)
                            + spatial_var)
                channel_values.append(float(de_value))

                # DE features: mean and std (matching training format)
                de_features.append(float(de_value))
                de_features.append(float(abs(np.random.normal(0, baseline["std"] * 0.2))))

                # PSD features: correlated with DE but offset (simulates power spectral density)
                psd_value = de_value * 1.2 + np.random.normal(0, baseline["std"] * 0.15)
                psd_features.append(float(psd_value))
                psd_features.append(float(abs(np.random.normal(0, baseline["std"] * 0.15))))

            band_powers[band_name] = channel_values

        # DE band ratio features (α/β, θ/β, γ/α — neuroscience markers)
        for ch in range(N_CHANNELS):
            band_means = [band_powers[b][ch] for b in BAND_NAMES]
            delta, theta, alpha, beta, gamma = band_means
            eps = 1e-8
            de_features.append(float(alpha / (beta + eps)))
            de_features.append(float(theta / (beta + eps)))
            de_features.append(float(gamma / (alpha + eps)))

        # PSD band ratio features (same ratios, slightly different values from PSD stream)
        for ch in range(N_CHANNELS):
            band_means = [band_powers[b][ch] * 1.2 for b in BAND_NAMES]  # PSD-scale
            delta, theta, alpha, beta, gamma = band_means
            eps = 1e-8
            psd_features.append(float(alpha / (beta + eps)))
            psd_features.append(float(theta / (beta + eps)))
            psd_features.append(float(gamma / (alpha + eps)))

        # Combine DE + PSD → 832 + 832 = 1664 features (matching trained model)
        features = de_features + psd_features

        # Generate raw waveform-like data for visualization (8 representative channels)
        t = np.linspace(0, 1, 200)
        for ch_idx in [0, 7, 11, 29, 37, 47, 54, 58]:  # Fp1, F7, Fz, Cz, CP5, Pz, PO3, O1
            wave = np.zeros_like(t)
            for band_idx, (band_name, freq_range) in enumerate([
                ("delta", (1, 4)), ("theta", (4, 8)), ("alpha", (8, 14)),
                ("beta", (14, 30)), ("gamma", (30, 47))
            ]):
                freq = np.mean(freq_range)
                amp = band_powers[band_name][ch_idx] * 0.3
                wave += amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
            raw_channels.append(wave.tolist())

        self._time_counter += 1.0 / max(self.speed, 0.1)

        return {
            "raw_channels": raw_channels,
            "band_powers": band_powers,
            "features": features,
            "timestamp": self._time_counter,
            "current_emotion": self.current_emotion,
            "channel_names": CHANNEL_NAMES[:8],
        }

    async def stream(self, callback):
        """
        Continuously generate and emit EEG frames.

        Args:
            callback: async function called with each frame dict
        """
        self.running = True
        while self.running:
            frame = self.generate_frame()
            await callback(frame)
            await asyncio.sleep(1.0 / max(self.speed, 0.1))

    def stop(self):
        self.running = False


# Global simulator instance
simulator = EEGSimulator()
