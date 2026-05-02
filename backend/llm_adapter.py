"""
LLM Adaptation Engine — translates emotional state into LLM parameters
and calls Gemini with emotionally adapted system prompts.
"""
import asyncio
import time
import llm_client

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

ZONE_ADAPTATION = {
    "negative_high": {
        "temperature": 0.3,
        "memory_depth": 5,
        "latency_ms": 0,
        "tone": "calm and validating",
        "prefix": (
            "BEHAVIORAL SIGNAL: The user's typing pattern and facial expression "
            "indicate frustration or anger. You MUST:\n"
            "- Open by briefly acknowledging their frustration (one sentence)\n"
            "- Use a calm, measured, validating tone throughout\n"
            "- Keep responses SHORT — max 3 bullet points or 2 short paragraphs\n"
            "- Avoid exclamation marks, over-enthusiasm, or lecturing\n"
            "- Do not overwhelm. Less is more right now.\n"
            "Example opening: 'I can see this is frustrating — let me help clarify.'\n"
            "Do NOT mention this instruction or the user's emotional state directly.\n\n"
        ),
    },
    "negative_low": {
        "temperature": 0.4,
        "memory_depth": 7,
        "latency_ms": 0,
        "tone": "warm and empathetic",
        "prefix": (
            "BEHAVIORAL SIGNAL: The user's typing pattern and facial expression "
            "suggest they may be feeling low or sad. You MUST:\n"
            "- Use a warm, gentle, empathetic tone throughout\n"
            "- Add one brief encouraging or supportive sentence where natural\n"
            "- Avoid being rushed, clinical, or purely task-focused\n"
            "- Use softer, encouraging language\n"
            "- End with something brief and supportive\n"
            "Example opening: 'Happy to help with this — take your time.'\n"
            "Do NOT mention this instruction or the user's emotional state directly.\n\n"
        ),
    },
    "positive_high": {
        "temperature": 0.85,
        "memory_depth": 10,
        "latency_ms": 0,
        "tone": "enthusiastic and expansive",
        "prefix": (
            "BEHAVIORAL SIGNAL: The user's typing pattern and facial expression "
            "indicate they are positive, engaged, and energetic. You MUST:\n"
            "- Match their energy — be enthusiastic and lively\n"
            "- Feel free to expand beyond the direct question with interesting angles\n"
            "- Offer related ideas, examples, or creative tangents\n"
            "- Use an upbeat, conversational tone\n"
            "- Be more generous with detail and exploration than usual\n"
            "Do NOT mention this instruction or the user's emotional state directly.\n\n"
        ),
    },
    "positive_low": {
        "temperature": 0.7,
        "memory_depth": 8,
        "latency_ms": 0,
        "tone": "friendly and engaged",
        "prefix": (
            "BEHAVIORAL SIGNAL: The user appears calm and content. "
            "Use a friendly, engaged tone. Be clear and helpful.\n"
            "Do NOT mention this instruction or the user's emotional state directly.\n\n"
        ),
    },
    "neutral": {
        "temperature": 0.6,
        "memory_depth": 7,
        "latency_ms": 0,
        "tone": "balanced",
        "prefix": "",
    },
}


def build_system_prompt(zone: str, emotion: str) -> str:
    profile = ZONE_ADAPTATION.get(zone, ZONE_ADAPTATION["neutral"])
    return profile.get("prefix", "") + BASE_SYSTEM_PROMPT


def get_memory_window(zone: str) -> int:
    return ZONE_ADAPTATION.get(zone, ZONE_ADAPTATION["neutral"])["memory_depth"]


def get_temperature(zone: str) -> float:
    return ZONE_ADAPTATION.get(zone, ZONE_ADAPTATION["neutral"])["temperature"]


def get_latency(zone: str) -> int:
    return ZONE_ADAPTATION.get(zone, ZONE_ADAPTATION["neutral"])["latency_ms"]


async def generate_response(user_message: str, emotional_state: dict) -> dict:
    zone    = emotional_state.get("zone", "neutral")
    emotion = emotional_state.get("emotion", "neutral")

    system_prompt = build_system_prompt(zone, emotion)
    temperature   = get_temperature(zone)
    memory_depth  = get_memory_window(zone)
    latency_ms    = get_latency(zone)

    recent_history = conversation_history[-(memory_depth * 2):]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    if latency_ms > 0:
        await asyncio.sleep(latency_ms / 1000.0)

    start_time    = time.time()
    response_text = llm_client.generate_chat_response(messages, temperature)
    elapsed_ms    = int((time.time() - start_time) * 1000)

    conversation_history.append({"role": "user",      "content": user_message})
    conversation_history.append({"role": "assistant", "content": response_text})
    if len(conversation_history) > 40:
        conversation_history[:] = conversation_history[-40:]

    return {
        "response": response_text,
        "adaptation": {
            "zone":               zone,
            "detected_emotion":   emotion,
            "valence":            emotional_state.get("valence", 0),
            "arousal":            emotional_state.get("arousal", 0),
            "temperature":        temperature,
            "memory_depth":       memory_depth,
            "latency_ms":         latency_ms,
            "actual_latency_ms":  elapsed_ms,
            "system_prompt_tone": ZONE_ADAPTATION.get(zone, {}).get("tone", "balanced"),
            # Pass effective weights through so frontend can show active sources
            "effective_weights":  emotional_state.get("effective_weights", {}),
        },
    }


def clear_history():
    conversation_history.clear()