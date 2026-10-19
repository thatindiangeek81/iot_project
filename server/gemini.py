"""Every call to Google's Gemini API lives here.

  chat()    OpenAI-compatible chat endpoint, with function tools (same endpoint the Worker uses)
  search()  native generateContent with the built-in Google Search tool
  embed()   native embedContent, for memory similarity
"""

from __future__ import annotations

import json
import math
import re

import requests

import config


class GeminiError(RuntimeError):
    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


_session = requests.Session()   # keeps the TLS connection to Google open between calls


def _post(url: str, body: dict, headers: dict, timeout: float | None = None) -> dict:
    try:
        r = _session.post(url, json=body, headers=headers, timeout=timeout or config.GEMINI_TIMEOUT)
    except requests.Timeout:
        raise GeminiError("gemini timed out", 504) from None
    except requests.RequestException as exc:
        raise GeminiError(f"could not reach gemini: {exc}", 502) from None
    if r.status_code == 429:
        raise GeminiError("gemini rate limit hit, try again shortly", 429)
    if r.status_code != 200:
        # Pass Gemini's own message through: it names a wrong model or key.
        raise GeminiError(f"gemini {r.status_code}: {r.text[:300]}", 502)
    try:
        return r.json()
    except ValueError:
        raise GeminiError("gemini sent a non-JSON reply", 502) from None


def _native_headers() -> dict:
    return {"x-goog-api-key": config.GEMINI_API_KEY}


# ---------------------------------------------------------------- chat

_reasoning_ok = True   # set False if this model refuses reasoning_effort, so we stop sending it


def chat(messages: list[dict], tools: list[dict] | None = None, model: str | None = None,
         max_tokens: int = 1024, json_mode: bool = False, reasoning: str | None = None) -> dict:
    """One chat completion. Returns the assistant message dict (content and/or tool_calls).
    reasoning: low | medium | high, how long Gemini thinks first (less = faster)."""
    global _reasoning_ok
    body: dict = {
        "model": model or config.GEMINI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if reasoning and _reasoning_ok:
        body["reasoning_effort"] = reasoning
    url = f"{config.GEMINI_BASE}/openai/chat/completions"
    headers = {"Authorization": f"Bearer {config.GEMINI_API_KEY}"}
    try:
        data = _post(url, body, headers)
    except GeminiError as exc:
        if "reasoning_effort" not in body or "400" not in str(exc):
            raise
        print(f"[gemini] this model won't take reasoning_effort={reasoning}; using its default ({exc})")
        _reasoning_ok = False
        body.pop("reasoning_effort")
        data = _post(url, body, headers)
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise GeminiError("gemini returned no message", 502) from None


def chat_json(prompt: str, model: str | None = None) -> dict | list | None:
    """Ask for a JSON answer; returns the parsed value or None."""
    msg = chat([{"role": "user", "content": prompt}], model=model, max_tokens=800, json_mode=True)
    text = (msg.get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        return None


# ---------------------------------------------------------------- web search

def search(query: str) -> str:
    """Answer `query` from a live Google Search, in a couple of plain sentences."""
    prompt = (
        "Search the web and answer for a voice assistant in at most three short plain sentences, "
        "with numbers and dates spelled out the way a person would say them. "
        f"Say when the information is from if it matters. Question: {query}"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    url = f"{config.GEMINI_BASE}/models/{config.GEMINI_SEARCH_MODEL}:generateContent"
    level = config.GEMINI_REASONING
    if level and _reasoning_ok:                       # think briefly: a search answer needs little
        body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": level}}
    try:
        data = _post(url, body, _native_headers(), timeout=config.GEMINI_TIMEOUT + 10)
    except GeminiError as exc:
        if "generationConfig" not in body or "400" not in str(exc) or "thinking" not in str(exc).lower():
            raise
        body.pop("generationConfig")
        data = _post(url, body, _native_headers(), timeout=config.GEMINI_TIMEOUT + 10)
    parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
    text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise GeminiError("search returned nothing", 502)
    return text


# ---------------------------------------------------------------- embeddings

def embed(text: str) -> list[float]:
    """Unit-length embedding of `text` (EMBED_DIMS numbers)."""
    body = {
        "content": {"parts": [{"text": text[:2000]}]},
        "outputDimensionality": config.EMBED_DIMS,
    }
    data = _post(f"{config.GEMINI_BASE}/models/{config.GEMINI_EMBED_MODEL}:embedContent",
                 body, _native_headers(), timeout=10)
    values = (data.get("embedding") or {}).get("values")
    if not isinstance(values, list) or not values:
        raise GeminiError("no embedding in the reply", 502)
    # Shortened embeddings are not unit length; normalise so a dot product is cosine similarity.
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]
