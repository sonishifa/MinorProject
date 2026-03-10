"""
Brain-First Model Tuning Toolkit — FastAPI Server

Main entry point. Serves the frontend, handles chat requests with the full
emotion detection → fusion → LLM adaptation pipeline, and streams EEG data
over WebSocket.
"""
# Suppress sklearn version mismatch warnings (models trained on Kaggle with different version)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import sys
import asyncio
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure backend/ is on the path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from config import FRONTEND_DIR
from text_analyzer import analyze_text
from keystroke_engine import predict_keystroke_emotion
from eeg_engine import predict_eeg_emotion
from eeg_simulator import simulator
from fusion import fusion_engine
from llm_adapter import generate_response, clear_history


# ── App lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  Brain-First Model Tuning Toolkit")
    print("  Ready at http://localhost:8000")
    print("=" * 60 + "\n")
    yield
    simulator.stop()


app = FastAPI(title="Brain-First Model Tuning Toolkit", lifespan=lifespan)

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ──
class ChatRequest(BaseModel):
    message: str
    keystroke_events: list[dict] = []  # Raw keydown/keyup events from frontend


class SimulationControl(BaseModel):
    action: str  # "start", "stop", "set_emotion", "set_speed"
    emotion: str = "neutral"
    speed: float = 1.0


# ── Chat endpoint (the core pipeline) ──
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Full pipeline: message + keystrokes → emotion detection → fusion → adapted LLM response.

    Flow:
    1. Text analyzer classifies the message content
    2. Keystroke engine classifies the typing behavior
    3. EEG engine classifies the current simulated EEG state
    4. Fusion engine merges all signals
    5. LLM adapter generates an adapted response
    """
    signals = []

    # ── Signal 1: Text sentiment ──
    text_result = analyze_text(request.message)
    signals.append(text_result)

    # ── Signal 2: Keystroke dynamics ──
    if request.keystroke_events:
        ks_result = predict_keystroke_emotion(request.keystroke_events)
        signals.append(ks_result)

    # ── Signal 3: EEG (from simulator) ──
    if simulator.running:
        frame = simulator.generate_frame()
        eeg_result = predict_eeg_emotion(frame["features"])
        signals.append(eeg_result)

    # ── Fuse all signals ──
    fused_state = fusion_engine.fuse(signals)

    # ── Generate adapted LLM response ──
    llm_result = await generate_response(request.message, fused_state)

    return {
        "response": llm_result["response"],
        "emotional_state": fused_state,
        "adaptation": llm_result["adaptation"],
        "signals": {
            "text": text_result,
            "keystroke": signals[1] if len(signals) > 1 and signals[1].get("source", "").startswith("keystroke") else None,
            "eeg": signals[-1] if len(signals) > 0 and signals[-1].get("source", "").startswith("eeg") else None,
        },
    }


# ── Status endpoint ──
@app.get("/api/status")
async def status():
    """Current system state."""
    return {
        "emotional_state": fusion_engine._current_state(),
        "eeg_simulator": {
            "running": simulator.running,
            "emotion": simulator.current_emotion,
            "speed": simulator.speed,
        },
        "fusion_history": fusion_engine.history[-10:],
    }


# ── Simulation control ──
@app.post("/api/simulate")
async def simulate(control: SimulationControl):
    """Control the EEG simulator."""
    if control.action == "start":
        simulator.running = True
        simulator.speed = control.speed
        return {"status": "started", "speed": simulator.speed}
    elif control.action == "stop":
        simulator.stop()
        return {"status": "stopped"}
    elif control.action == "set_emotion":
        simulator.set_emotion(control.emotion)
        return {"status": "emotion_set", "emotion": control.emotion}
    elif control.action == "set_speed":
        simulator.speed = control.speed
        return {"status": "speed_set", "speed": control.speed}
    return {"status": "unknown_action"}


# ── Reset ──
@app.post("/api/reset")
async def reset():
    """Reset all state."""
    fusion_engine.reset()
    clear_history()
    simulator.stop()
    simulator.current_emotion = "neutral"
    return {"status": "reset"}


# ── EEG WebSocket (real-time streaming) ──
@app.websocket("/ws/eeg")
async def eeg_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time EEG data streaming.

    Sends JSON frames containing:
    - raw_channels: waveform data for visualization
    - band_powers: per-band power values for the brain map
    - current_emotion: the simulator's current state
    - eeg_prediction: classification result from the EEG engine
    """
    await websocket.accept()
    print("[WebSocket] EEG client connected")

    try:
        while True:
            if simulator.running:
                frame = simulator.generate_frame()

                # Also run EEG classification on the simulated data
                eeg_result = predict_eeg_emotion(frame["features"])

                await websocket.send_json({
                    "type": "eeg_frame",
                    "raw_channels": frame["raw_channels"],
                    "band_powers": frame["band_powers"],
                    "timestamp": frame["timestamp"],
                    "simulator_emotion": frame["current_emotion"],
                    "predicted_emotion": eeg_result["emotion"],
                    "prediction_confidence": eeg_result["confidence"],
                    "channel_names": frame["channel_names"],
                })
            else:
                await websocket.send_json({"type": "idle"})

            await asyncio.sleep(1.0 / max(simulator.speed, 0.1))
    except WebSocketDisconnect:
        print("[WebSocket] EEG client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")


# ── Serve frontend ──
@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend not yet built. API is running.",
                         "docs": "/docs"})


# Mount static assets (CSS, JS) AFTER explicit routes
# Use /static prefix to avoid catching API routes
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    # Also mount at root for direct file access (styles.css, app.js)
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
