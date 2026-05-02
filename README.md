# Brain-First Model Tuning Toolkit

The **Brain-First Model Tuning Toolkit** is a multimodal affective computing engine designed to continuously align Large Language Model (LLM) responses with the real-time emotional state of the user. It leverages deep learning (CNN-LSTM) over keystroke dynamics, facial expression recognition, and textual sentiment analysis to dynamically adapt system prompts, generating highly contextual, empathetic, and responsive AI interactions.

## Architecture & Modalities

1. **Keystroke Dynamics Engine:** 
   - A PyTorch-based **CNN-LSTM** model that passively tracks typing cadence, dwell times (keydown to keyup), and digraph latencies.
   - Evaluates to a 4-class emotional state (`positive`, `angry`, `sad`, `neutral`).
   - Features **rolling background calibration**: The model continuously learns a user's personal baseline typing speed to accurately detect deviations.

2. **Facial Expression Recognition:**
   - Utilizes `DeepFace` (with the built-in FER CNN) on real-time browser-captured frames.
   - Converts standard facial expressions into continuous Valence/Arousal metrics.
   - Preprocessed using CLAHE for robust lighting normalization.

3. **Text Sentiment Analysis:**
   - Powered by the Google Gemini API.
   - Extracts contextual emotional sentiment directly from the message content.

4. **Fusion Engine:**
   - Implements **Dynamic Confidence Weighting** (ignoring weak signals) and **Dual-Scale Temporal Smoothing** (fast message-scale EMA + slow session-scale EMA) to merge the three modalities into a stable, two-dimensional Valence/Arousal space.
   - Outputs a distinct "zone" (e.g., `negative_high`, `positive_low`) which triggers specific adaptation profiles.

5. **LLM Adaptation:**
   - Alters the LLM's system prompt (Tone), Temperature, and Memory Depth based on the recognized emotional zone, producing responses that mirror and validate the user's state.

## Setup Instructions

### 1. Requirements
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the project root and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite
```

### 3. Run the Server
The application runs on a FastAPI backend. Start it using uvicorn:
```bash
cd backend
uvicorn main:app --reload
```
*(Note: On the very first launch, DeepFace will download its pre-trained weights and take ~3-5 seconds to warm up).*

### 4. Access the UI
Open your browser and navigate to:
**http://localhost:8000**

## Project Structure
- `/backend/`: FastAPI application, fusion engine, and all modality modules.
- `/frontend/`: Pure HTML/CSS/JS frontend interface featuring live VA gauges, telemetry visualizers, and signal cards.
- `/models/`: Trained model weights (`.pt`) and normalization parameters (`.npy`, `.pkl`).
- `/notebooks/`: Jupyter notebooks used for research, dataset processing, and model training (including EEG pipelines kept for historical/offline research).

## Acknowledgments
This toolkit relies heavily on concepts derived from the **EmoSurv** (typing biometrics) and **EmoEEG** datasets. All offline data processing and original Deep Learning research was strictly conducted on those controlled benchmarks.
