# llm_client.py
import json
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL

# Configure Gemini
_use_gemini = bool(GEMINI_API_KEY and "your-" not in GEMINI_API_KEY)
if _use_gemini:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("[LLMClient] No valid Gemini API key – using fallback responses.")
    client = None

# ------------------------------------------------------------------
# Emotion classification prompt (same as before)
# ------------------------------------------------------------------
CLASSIFIER_PROMPT = """You are an emotion classifier. Analyze the user's message and return ONLY a JSON object with:
- "emotion": one of [joy, sadness, frustration, anger, anxiety, curiosity, excitement, neutral, fear, disgust]
- "valence": float from -1.0 (very negative) to 1.0 (very positive)
- "arousal": float from -1.0 (very calm) to 1.0 (very agitated/excited)
- "confidence": float from 0.0 to 1.0 (how certain you are)

Examples:
User: "ugh I keep getting this stupid error"
{"emotion": "frustration", "valence": -0.6, "arousal": 0.8, "confidence": 0.9}

User: "this is so cool, I just learned about neural networks!"
{"emotion": "excitement", "valence": 0.9, "arousal": 0.9, "confidence": 0.95}

User: "what is 2+2"
{"emotion": "neutral", "valence": 0.0, "arousal": 0.0, "confidence": 0.85}

User: "I don't know if I can do this anymore"
{"emotion": "sadness", "valence": -0.7, "arousal": -0.4, "confidence": 0.8}

Return ONLY the JSON. No explanation."""

def _parse_gemini_response(text: str) -> dict:
    """Extract JSON from Gemini response (handles possible markdown code blocks)."""
    text = text.strip()
    if text.startswith("```"):
        # Remove code fences
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)

def classify_emotion(message: str) -> dict:
    """
    Use Gemini to classify the emotional content of a message.
    Returns a dict with keys: emotion, valence, arousal, confidence, source.
    If Gemini fails, falls back to keyword analysis.
    """
    if not message or not message.strip():
        return _neutral_result("text_empty")

    if not _use_gemini or client is None:
        return _keyword_fallback(message)

    try:
        # Combine prompt and user message
        full_prompt = f"{CLASSIFIER_PROMPT}\n\nUser: {message}"
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,          # low temperature for classification
                top_p=0.95,
                top_k=20,
            )
        )
        result = _parse_gemini_response(response.text)
        result["source"] = "text"
        result.setdefault("confidence", 0.7)
        return result
    except Exception as e:
        print(f"[LLMClient] Gemini classification error: {e}")
        return _keyword_fallback(message)

def generate_chat_response(messages: list, temperature: float = 0.7) -> str:
    """
    Generate a chat response using Gemini.
    messages: list of dicts with 'role' and 'content' (like OpenAI format)
    temperature: controls randomness
    Returns response text, or a fallback message if Gemini fails.
    """
    if not _use_gemini or client is None:
        return _mock_response(messages)

    try:
        # Convert OpenAI-style messages to a single prompt for Gemini.
        # Gemini does not natively support multi-turn conversation history in a simple call,
        # but we can concatenate the conversation into a single prompt.
        # Extract system prompt and user/assistant turns.
        system_prompt = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation.append(f"Assistant: {msg['content']}")

        # Build the full prompt with system instruction at the top.
        full_prompt = system_prompt + "\n\n" + "\n".join(conversation) + "\nAssistant:"

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=20,
            )
        )
        return response.text
    except Exception as e:
        print(f"[LLMClient] Gemini chat error: {e}")
        return _mock_response(messages)

# ------------------------------------------------------------------
# Fallback functions (unchanged)
# ------------------------------------------------------------------
def _keyword_fallback(message: str) -> dict:
    """Simple keyword‑based emotion detection."""
    msg = message.lower()
    frustration_words = ["ugh", "stuck", "error", "bug", "broken", "stupid",
                         "can't", "won't work", "frustrated", "annoying",
                         "fed up", "hate", "damn", "failing"]
    sad_words = ["sad", "depressed", "lonely", "hopeless", "crying",
                 "give up", "anymore", "tired", "exhausted", "can't do this"]
    joy_words = ["amazing", "awesome", "cool", "love", "great", "fantastic",
                 "excited", "happy", "wonderful", "brilliant", "perfect",
                 "finally", "yay", "wow"]
    anxiety_words = ["worried", "nervous", "scared", "anxious", "panic",
                     "overwhelmed", "stress", "afraid", "unsure", "confused"]

    frust_score = sum(1 for w in frustration_words if w in msg)
    sad_score = sum(1 for w in sad_words if w in msg)
    joy_score = sum(1 for w in joy_words if w in msg)
    anx_score = sum(1 for w in anxiety_words if w in msg)

    scores = {
        "frustration": frust_score,
        "sadness": sad_score,
        "joy": joy_score,
        "anxiety": anx_score,
    }

    if max(scores.values()) == 0:
        return _neutral_result("text_fallback")

    detected = max(scores, key=scores.get)

    # Use EMOTION_VA_MAP from config (import it dynamically to avoid circular imports)
    from config import EMOTION_VA_MAP
    va = EMOTION_VA_MAP.get(detected, {"valence": 0.0, "arousal": 0.0})

    return {
        "emotion": detected,
        "valence": va["valence"],
        "arousal": va["arousal"],
        "confidence": min(0.3 + max(scores.values()) * 0.15, 0.8),
        "source": "text_fallback",
    }

def _neutral_result(source: str = "text") -> dict:
    return {
        "emotion": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "confidence": 0.5,
        "source": source,
    }

def _mock_response(messages: list) -> str:
    """Simple fallback response when Gemini is unavailable."""
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not last_user:
        return "I'm here to help. Could you tell me more?"
    return f"I'm here to assist you with '{last_user[:50]}...'. (Gemini API not configured)"