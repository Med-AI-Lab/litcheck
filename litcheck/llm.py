import os

import requests
from dotenv import load_dotenv


# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"


# ─────────────────────────────────────────────
# Утилита: LLM-запрос
# ─────────────────────────────────────────────

def llm(system: str, user: str, max_tokens: int = 1024) -> str:
    """Единая точка обращения к OpenRouter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://semantic-pubmed-search",
        "X-Title": "Semantic PubMed Search",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # убираем возможные ```json … ``` обёртки
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()