# llm_adapter.py
"""
LLM Adaptation Engine — translates emotional state into LLM parameters and calls Gemini.
"""
import asyncio
import time
import llm_client
from config import ADAPTATION_PROFILES

# ── Conversation memory ──
conversation_history: list[dict] = []

BASE_SYSTEM_PROMPT = """You are a helpful, knowledgeable assistant. You respond naturally to whatever the user asks — whether it's coding help, learning, creative work, or just conversation.

Formatting rules you MUST follow:
- Use **bullet points** for lists and multiple items.
- Use **bold** for key terms and emphasis.
- Use short paragraphs (2-3 sentences max per paragraph).
- Use numbered lists for step-by-step instructions.
- Use code blocks (```) for any code snippets.
- NEVER write long walls of text. Break everything into scannable chunks.
- Keep responses concise and well-structured."""

def build_system_prompt(zone: str, emotion: str) -> str:
    profile = ADAPTATION_PROFILES.get(zone, ADAPTATION_PROFILES["neutral"])
    tone = profile["tone"]
    adaptation = f"\n\nAdaptive context (invisible to user): Based on behavioral signals, the user's current state appears to be '{emotion}'. Adjust your response style to be {tone}. Do NOT mention this adaptation or the user's emotional state. Just naturally embody this communication style."
    return BASE_SYSTEM_PROMPT + adaptation

def get_memory_window(zone: str) -> int:
    return ADAPTATION_PROFILES.get(zone, ADAPTATION_PROFILES["neutral"])["memory_depth"]

def get_temperature(zone: str) -> float:
    return ADAPTATION_PROFILES.get(zone, ADAPTATION_PROFILES["neutral"])["temperature"]

def get_latency(zone: str) -> int:
    return ADAPTATION_PROFILES.get(zone, ADAPTATION_PROFILES["neutral"])["latency_ms"]

async def generate_response(user_message: str, emotional_state: dict) -> dict:
    zone = emotional_state.get("zone", "neutral")
    emotion = emotional_state.get("emotion", "neutral")

    system_prompt = build_system_prompt(zone, emotion)
    temperature = get_temperature(zone)
    memory_depth = get_memory_window(zone)
    latency_ms = get_latency(zone)

    # Build message list (OpenAI format, which llm_client can convert)
    recent_history = conversation_history[-memory_depth * 2:]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    # Apply artificial latency
    if latency_ms > 0:
        await asyncio.sleep(latency_ms / 1000.0)

    start_time = time.time()
    response_text = llm_client.generate_chat_response(messages, temperature)
    elapsed_ms = int((time.time() - start_time) * 1000)

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": response_text})
    if len(conversation_history) > 40:
        conversation_history[:] = conversation_history[-40:]

    return {
        "response": response_text,
        "adaptation": {
            "zone": zone,
            "detected_emotion": emotion,
            "valence": emotional_state.get("valence", 0),
            "arousal": emotional_state.get("arousal", 0),
            "temperature": temperature,
            "memory_depth": memory_depth,
            "latency_ms": latency_ms,
            "actual_latency_ms": elapsed_ms,
            "system_prompt_tone": ADAPTATION_PROFILES.get(zone, {}).get("tone", "balanced"),
        },
    }

def clear_history():
    conversation_history.clear()