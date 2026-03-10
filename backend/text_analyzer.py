
"""
Text Sentiment Analyzer — detects emotion from the CONTENT of the user's message.
Uses Gemini as a classifier, with a keyword fallback if Gemini fails.
"""
import llm_client


def analyze_text(message: str) -> dict:
    """
    Analyze the emotional content of a text message.

    Args:
        message: The user's chat message

    Returns:
        dict with keys: emotion, valence, arousal, confidence, source, message_length
    """
    result = llm_client.classify_emotion(message)
    # Add message length for dynamic confidence calculation
    # Longer messages give the LLM more context → higher reliability
    result["message_length"] = len(message.strip()) if message else 0
    return result