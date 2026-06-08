"""
scripts/llm_clients.py
Unified LLM client abstraction — Ollama (local) + cloud fallbacks.

Ollama models (primary, used in this research)
-----------------------------------------------
  llama3.1        → llama3.1:8b-instruct-q4_K_M   (Llama 3.1 8B Instruct)
  qwen2.5         → qwen2.5:14b-instruct-q4_K_M    (Qwen 2.5 14B Instruct)
  mistral         → mistral:7b-instruct-v0.3        (Mistral 7B Instruct)

Ollama must be running locally:  ollama serve
Pull a model before use:         ollama pull llama3.1:8b-instruct-q4_K_M

Cloud models (optional, kept for reference)
-------------------------------------------
  gpt    → OpenAI API   (OPENAI_API_KEY)
  gemini → Google AI    (GOOGLE_API_KEY)

Model registry
--------------
All models are defined in MODEL_REGISTRY below.
To add a new model, just add one entry — no other file needs changing.
The model's short alias (e.g. "llama3.1") is used everywhere as the
identifier: in filenames, CLI flags, and result CSVs.
"""

from __future__ import annotations
import os
import re
import time
import logging

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# One entry per model.  Fields:
#   model_string  – exact name passed to the API
#   backend       – "ollama" | "openai" | "google"
#   rpm           – requests per minute (for proactive throttling)
#   description   – human-readable label used in figures / reports
# ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    # ── Ollama / local ──────────────────────────────────────────
    "llama3.1": {
        "model_string": "llama3.1:8b",
        "backend":      "ollama",
        "rpm":          120,   # local — limited only by hardware
        "description":  "Llama 3.1 8B",
    },
    "qwen2.5": {
        "model_string": "qwen2.5:14b",
        "backend":      "ollama",
        "rpm":          60,
        "description":  "Qwen 2.5 14B Instruct",
    },
    "mistral": {
        "model_string": "mistral:7b",
        "backend":      "ollama",
        "rpm":          120,
        "description":  "Mistral 7B Instruct v0.3",
    },
    # ── Cloud (optional) ────────────────────────────────────────
    "gpt": {
        "model_string": "gpt-4o-mini",
        "backend":      "openai",
        "rpm":          500,
        "description":  "GPT-4o-mini",
    },
    "gemini": {
        "model_string": "gemini-2.5-flash",
        "backend":      "google",
        "rpm":          4,     # free tier: 5 RPM
        "description":  "Gemini 2.5 Flash",
    },
}

# Derived helpers
SUPPORTED_MODELS   = list(MODEL_REGISTRY.keys())
OLLAMA_MODELS      = [k for k, v in MODEL_REGISTRY.items() if v["backend"] == "ollama"]
CLOUD_MODELS       = [k for k, v in MODEL_REGISTRY.items() if v["backend"] != "ollama"]


def get_model_string(alias: str) -> str:
    """Return the exact model string for a given alias."""
    if alias not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model alias '{alias}'. Available: {SUPPORTED_MODELS}")
    return MODEL_REGISTRY[alias]["model_string"]


def get_model_description(alias: str) -> str:
    return MODEL_REGISTRY.get(alias, {}).get("description", alias)


# ─────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────

class DailyQuotaExhausted(Exception):
    """Per-day free-tier limit hit. No point retrying until tomorrow."""

class OllamaNotRunning(Exception):
    """Ollama server is not reachable at OLLAMA_BASE_URL."""

class ModelNotPulled(Exception):
    """The requested Ollama model has not been pulled yet."""


# ─────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────

def parse_yes_no(raw: str) -> int:
    """
    Extract binary label from LLM response.
    Returns 1 (defect), 0 (no defect), or -1 (parse failure).
    """
    text = raw.strip().lower()
    # CoT: "Final Answer: Yes / No"
    m = re.search(r"final answer[:\s]+([a-z]+)", text)
    if m:
        w = m.group(1)
        if w.startswith("yes"): return 1
        if w.startswith("no"):  return 0
    if re.search(r"\byes\b", text): return 1
    if re.search(r"\bno\b",  text): return 0
    logger.warning("Could not parse response: %s", raw[:120])
    return -1


# ─────────────────────────────────────────────
# RATE-LIMIT HELPERS  (cloud only)
# ─────────────────────────────────────────────

def _parse_retry_seconds(error_str: str) -> float | None:
    m = re.search(r"retry_delay\s*\{[^}]*seconds:\s*(\d+(?:\.\d+)?)", error_str)
    if m: return float(m.group(1)) + 2
    m = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", error_str, re.IGNORECASE)
    if m: return float(m.group(1)) + 2
    return None

def _is_daily_quota(error_str: str) -> bool:
    return any(k in error_str for k in ("PerDay", "per_day", "daily"))


# ─────────────────────────────────────────────
# BASE CLIENT
# ─────────────────────────────────────────────

class BaseLLMClient:
    name: str        = "base"
    model_string: str = ""
    rpm: int         = 60
    _last_call_time: float = 0.0

    def _call_api(self, prompt: str) -> str:
        raise NotImplementedError

    def _min_interval(self) -> float:
        return 60.0 / max(self.rpm, 1)

    def predict(self, prompt: str, retries: int = 5, base_delay: float = 5.0) -> tuple[int, str]:
        """
        Returns (label, raw_response).
        Handles rate limits, exponential backoff, and daily quota errors.
        """
        elapsed = time.monotonic() - self._last_call_time
        gap = self._min_interval()
        if elapsed < gap:
            time.sleep(gap - elapsed)

        for attempt in range(retries):
            try:
                self._last_call_time = time.monotonic()
                raw = self._call_api(prompt)
                return parse_yes_no(raw), raw

            except (OllamaNotRunning, ModelNotPulled, DailyQuotaExhausted):
                raise   # don't retry unrecoverable errors

            except Exception as exc:
                err_str = str(exc)

                if "429" in err_str and _is_daily_quota(err_str):
                    logger.error("[%s] Daily quota exhausted.", self.name)
                    raise DailyQuotaExhausted(f"{self.name} daily quota exhausted.") from exc

                if "429" in err_str:
                    wait = _parse_retry_seconds(err_str) or base_delay * (2 ** attempt)
                    logger.warning("[%s] Rate limited (attempt %d/%d). Waiting %.1fs...",
                                   self.name, attempt + 1, retries, wait)
                    time.sleep(wait)
                    continue

                wait = base_delay * (2 ** attempt)
                logger.warning("[%s] attempt %d/%d failed: %s — retrying in %.1fs",
                               self.name, attempt + 1, retries, err_str[:120], wait)
                time.sleep(wait)

        logger.error("[%s] All %d attempts failed.", self.name, retries)
        return -1, "ERROR"


# ─────────────────────────────────────────────
# OLLAMA CLIENT  (local, used in this research)
# ─────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """
    Calls a locally running Ollama server via its REST API.
    No API key required. Requires:
      1. ollama serve          (start the server)
      2. ollama pull <model>   (download the model once)
    """

    def __init__(self, alias: str):
        if alias not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model alias '{alias}'. Available: {SUPPORTED_MODELS}")
        entry = MODEL_REGISTRY[alias]
        self.name         = alias
        self.model_string = entry["model_string"]
        self.rpm          = entry["rpm"]
        self._last_call_time = 0.0

        # Verify Ollama is reachable on init
        self._check_ollama()

    def _check_ollama(self):
        """Ping Ollama and confirm the model is available."""
        try:
            import urllib.request, json as _json
            with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
                data = _json.loads(r.read())
            available = [m["name"] for m in data.get("models", [])]
            # Partial match: "llama3.1:8b-instruct-q4_K_M" matches "llama3.1"
            matched = any(self.model_string in m or m.startswith(self.model_string.split(":")[0])
                          for m in available)
            if not matched:
                raise ModelNotPulled(
                    f"Model '{self.model_string}' not found in Ollama.\n"
                    f"Pull it with:  ollama pull {self.model_string}\n"
                    f"Available models: {available}"
                )
        except ModelNotPulled:
            raise
        except Exception as exc:
            if "Connection refused" in str(exc) or "urlopen error" in str(exc):
                raise OllamaNotRunning(
                    f"Cannot reach Ollama at {OLLAMA_BASE_URL}.\n"
                    "Start it with:  ollama serve"
                ) from exc
            # If we can't check (timeout on CI etc.), warn and continue
            logger.warning("Could not verify Ollama availability: %s", exc)

    def _call_api(self, prompt: str) -> str:
        import urllib.request, urllib.error, json as _json

        payload = _json.dumps({
            "model":  self.model_string,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed":        42,    # deterministic output
                "num_predict": 256,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = _json.loads(resp.read())
            return data.get("response", "").strip()
        except urllib.error.URLError as exc:
            raise OllamaNotRunning(
                f"Lost connection to Ollama at {OLLAMA_BASE_URL}: {exc}"
            ) from exc


# ─────────────────────────────────────────────
# CLOUD CLIENTS  (optional)
# ─────────────────────────────────────────────

class GPTClient(BaseLLMClient):
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        self.name         = "gpt"
        self.model_string = MODEL_REGISTRY["gpt"]["model_string"]
        self.rpm          = MODEL_REGISTRY["gpt"]["rpm"]
        self._last_call_time = 0.0
        self._client      = OpenAI(api_key=api_key)

    def _call_api(self, prompt: str) -> str:
        r = self._client.chat.completions.create(
            model=self.model_string,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=512,
        )
        return r.choices[0].message.content.strip()


class GeminiClient(BaseLLMClient):
    def __init__(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set")
        self.name         = "gemini"
        self.model_string = MODEL_REGISTRY["gemini"]["model_string"]
        self.rpm          = MODEL_REGISTRY["gemini"]["rpm"]
        self._last_call_time = 0.0
        genai.configure(api_key=api_key)
        self._model  = genai.GenerativeModel(self.model_string)
        self._genai  = genai

    def _call_api(self, prompt: str) -> str:
        r = self._model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=0, max_output_tokens=512),
        )
        return r.text.strip()


# ─────────────────────────────────────────────
# MOCK CLIENT  (testing — no Ollama needed)
# ─────────────────────────────────────────────

class MockClient(BaseLLMClient):
    rpm = 10000

    def __init__(self, name: str = "mock", seed: int = 0):
        import random
        self.name         = name
        self.model_string = f"mock-{name}"
        self._last_call_time = 0.0
        self._rng         = random.Random(seed)

    def _call_api(self, prompt: str) -> str:
        return self._rng.choice(["Yes", "No"])


# ─────────────────────────────────────────────
# FACTORY  — single entry point
# ─────────────────────────────────────────────

def get_client(alias: str, mock: bool = False) -> BaseLLMClient:
    """
    Return the appropriate client for `alias`.

    mock=True  → MockClient (no Ollama / API key needed, for CI / testing)
    mock=False → OllamaClient for ollama models, cloud clients for others
    """
    if alias not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model alias '{alias}'.\n"
            f"Available: {SUPPORTED_MODELS}\n"
            f"To add a new model, edit MODEL_REGISTRY in scripts/llm_clients.py"
        )

    if mock:
        seed = abs(hash(alias)) % 10000
        return MockClient(name=alias, seed=seed)

    backend = MODEL_REGISTRY[alias]["backend"]

    if backend == "ollama":
        return OllamaClient(alias)
    elif backend == "openai":
        return GPTClient()
    elif backend == "google":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown backend '{backend}' for model '{alias}'")
