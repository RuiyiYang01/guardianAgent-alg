# rag/llm_io.py
from __future__ import annotations
import os
import json

# Global Client Cache
_OPENAI_CLIENT = None

def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        try:
            from openai import OpenAI

            # === Key Modifications ===
            # Read base_url; if not set, default to None (connect to official OpenAI)
            base_url = os.getenv("LLM_BASE_URL", None)
            api_key = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY"))

            # Local models typically don't need a real key, but Client initialization requires a non-empty string
            if not api_key and base_url:
                api_key = "dummy-key"

            _OPENAI_CLIENT = OpenAI(base_url=base_url, api_key=api_key)
            print(f"[LLM] Client initialized. Target: {base_url if base_url else 'OpenAI Cloud'}")

        except ImportError:
            raise RuntimeError("Missing dependency. Please `pip install openai`.")
    return _OPENAI_CLIENT

# Read configuration
OPENAI_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | local

def chat(system: str, user: str) -> str:
    """
    Universal chat interface supporting OpenAI and local models compatible with OpenAI protocol (Ollama/vLLM)
    """
    client = get_openai_client()
    try:
        # Construct parameters
        kwargs = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.0,
        }

        # === JSON Mode Compatibility Handling ===
        # GPT series and some excellent open-source inference engines (vLLM/Ollama latest) support json_object
        # If your legacy model doesn't support it, you can comment out this line or rely on regex cleaning in parse.py
        # For robustness, we only use JSON mode when provider is openai or JSON mode is explicitly enabled
        if os.getenv("LLM_JSON_MODE", "true").lower() == "true":
             kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    except Exception as e:
        print(f"[LLM Error] Provider: {LLM_PROVIDER}, Error: {e}")
        # Return error info in valid JSON structure to prevent parse failure
        return json.dumps({
            "decision": "deny",
            "risk_score": 1.0,
            "rationale": f"LLM Inference Failed: {str(e)}",
            "evidence_ids": [],
            "transform": {}
        })