# rag/llm_io.py
from __future__ import annotations
import os
import json

# === Global Client Cache ===
_OPENAI_CLIENT = None
_GEMINI_CLIENT = None

def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            from openai import OpenAI
            base_url = os.getenv("LLM_BASE_URL", None)
            api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "dummy-key"))
            _OPENAI_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
        except ImportError:
            raise RuntimeError("Missing dependency: `pip install openai`")
    return _OPENAI_CLIENT

def get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            _GEMINI_CLIENT = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError("Missing dependency: `pip install google-genai`")
    return _GEMINI_CLIENT

def chat(system: str, user: str) -> str:
    """
    Universal LLM call interface supporting Gemini, OpenAI, and local protocols.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower() # Default to gemini

    try:
        # ==========================================
        # Route 1: Gemini API (Recommended)
        # ==========================================
        if provider == "gemini":
            client = get_gemini_client()
            model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")

            from google.genai import types

            response = client.models.generate_content(
                model=model_name,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0, # Ensure consistent decision-making across calls
                    response_mime_type="application/json", # Force Gemini to output only valid JSON
                )
            )
            return response.text

        # ==========================================
        # Route 2: OpenAI / Local Ollama Protocol
        # ==========================================
        elif provider in ["openai", "local"]:
            client = get_openai_client()
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.0,
            }

            if os.getenv("LLM_JSON_MODE", "true").lower() == "true":
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            # Strip <think>...</think> blocks from reasoning models (Qwen3, DeepSeek-R1)
            if content and "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    except Exception as e:
        print(f"[LLM Error] Provider: {provider}, Error: {e}")
        # Fallback response to prevent disrupting downstream processes
        return json.dumps({
            "decision": "deny",
            "risk_score": 1.0,
            "rationale": f"System 2 LLM Failed: {str(e)}",
            "evidence_ids": [],
            "transform": {}
        })