"""
Brain-First Model Tuning Toolkit — FastAPI Server
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import sys
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

sys.path.insert(0, str(Path(__file__).parent))

from config import FRONTEND_DIR
from text_analyzer import analyze_text
from keystroke_engine import predict_keystroke_emotion
from fusion import fusion_engine
from llm_adapter import generate_response, clear_history
from facial_engine import predict_facial_emotion, warmup as facial_warmup


# ── App lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("  Brain-First Model Tuning Toolkit")
    print("=" * 60)
    facial_warmup()     # pre-load DeepFace weights before first request
    print("  Ready at http://localhost:8000")
    print("=" * 60 + "\n")
    yield


app = FastAPI(title="Brain-First Model Tuning Toolkit", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ──
class ChatRequest(BaseModel):
    message:          str
    keystroke_events: List[dict] = []
    facial_frames:    List[str]  = []   # base64 JPEG strings from browser


# ── Chat endpoint ──
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Full pipeline:
      message + keystrokes + facial frames
      → emotion detection (text / keystroke / facial)
      → fusion
      → adapted LLM response
    """
    signals = []

    # ── Signal 1: Text sentiment ──
    text_result = analyze_text(request.message)
    signals.append(text_result)

    # ── Signal 2: Keystroke dynamics ──
    if request.keystroke_events:
        ks_result = predict_keystroke_emotion(request.keystroke_events)
        signals.append(ks_result)

    # ── Signal 3: Facial expression ──
    facial_result = predict_facial_emotion(request.facial_frames)
    # Only add to fusion if camera was active and got a real read
    if facial_result.get('camera_active') and facial_result.get('confidence', 0) > 0:
        signals.append(facial_result)

    # ── Fuse all signals ──
    fused_state = fusion_engine.fuse(signals)

    # ── Generate adapted LLM response ──
    llm_result = await generate_response(request.message, fused_state)

    # ── Build signals dict for frontend display ──
    ks_signal = next((s for s in signals if s.get('source', '').startswith('keystroke')), None)

    return {
        "response":       llm_result["response"],
        "emotional_state": fused_state,
        "adaptation":     llm_result["adaptation"],
        "signals": {
            "text":      text_result,
            "keystroke": ks_signal,
            "facial":    facial_result,
        },
    }


# ── Background Telemetry Endpoint ──
@app.post("/api/telemetry/keystrokes")
async def background_keystrokes(events: List[dict]):
    """
    Silently receives global keystrokes from the desktop tracker.
    Updates the rolling baseline without invoking the LLM or updating the chat UI.
    """
    if events:
        # Calling this automatically runs the internal update_rolling_baseline logic
        predict_keystroke_emotion(events)
    return {"status": "success", "message": "Baseline updated"}


# ── Status endpoint ──
@app.get("/api/status")
async def status():
    return {
        "emotional_state": fusion_engine._current_state(),
        "fusion_history": fusion_engine.history[-10:],
    }


# ── Reset ──
@app.post("/api/reset")
async def reset():
    fusion_engine.reset()
    clear_history()
    return {"status": "reset"}


# ── Serve frontend ──
@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend not built. API running.", "docs": "/docs"})


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)