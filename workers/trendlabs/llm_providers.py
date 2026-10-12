"""Multi-provider LLM clients for the shared research agent (never executes trades).

Uses requests-based HTTP against free-tier / local chat APIs. Secrets stay in env;
this module only reports configured=True/False for status surfaces.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import requests

from trendlabs import config


class LLMError(RuntimeError):
    """Provider call failed (auth, rate limit, parse, or network)."""


@dataclass(frozen=True)
class ProviderInfo:
    id: str
    label: str
    configured: bool
    default_model: str
    available: bool = True


class LLMProvider(ABC):
    id: str
    label: str

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def default_model(self) -> str:
        ...

    def is_available(self) -> bool:
        return self.is_configured()

    @abstractmethod
    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        ...

    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id,
            label=self.label,
            configured=self.is_configured(),
            default_model=self.default_model(),
            available=self.is_available(),
        )


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMError("Empty model response")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Fenced or prose-wrapped JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        data = json.loads(fence.group(1))
        if isinstance(data, dict):
            return data
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise LLMError("Model response was not valid JSON")


def _openai_chat_json(
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    system: str,
    user: str,
    extra: dict[str, Any] | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    if extra:
        body.update(extra)
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Network error: {exc}") from exc

    if resp.status_code == 429:
        raise LLMError(f"Rate limited ({resp.status_code})")
    if resp.status_code in {401, 403}:
        raise LLMError(f"Auth failed ({resp.status_code})")
    if resp.status_code >= 400:
        detail = (resp.text or "")[:300]
        raise LLMError(f"HTTP {resp.status_code}: {detail}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise LLMError("Non-JSON HTTP response") from exc

    # OpenAI-compatible
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        if content:
            return _extract_json(str(content))

    # Some gateways nest differently
    if isinstance(payload.get("content"), str):
        return _extract_json(payload["content"])
    if isinstance(payload.get("response"), str):
        return _extract_json(payload["response"])

    raise LLMError("No chat content in provider response")


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible chat/completions client."""

    def __init__(
        self,
        *,
        provider_id: str,
        label: str,
        api_key: str,
        base_url: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format_json: bool = True,
    ) -> None:
        self.id = provider_id
        self.label = label
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or {}
        self._response_format_json = response_format_json

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def default_model(self) -> str:
        return self._model

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if not self.is_configured():
            raise LLMError(f"{self.id} is not configured (missing API key)")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        extra = dict(self._extra_body)
        if self._response_format_json:
            extra.setdefault("response_format", {"type": "json_object"})
        url = f"{self._base_url}/chat/completions"
        return _openai_chat_json(
            url=url,
            headers=headers,
            model=model or self._model,
            system=system,
            user=user,
            extra=extra or None,
        )


class GoogleGeminiProvider(LLMProvider):
    id = "google"
    label = "Google AI Studio (Gemini)"

    def __init__(self) -> None:
        self._api_key = (config.GOOGLE_AI_API_KEY or "").strip()
        self._model = config.GOOGLE_AI_MODEL

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def default_model(self) -> str:
        return self._model

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if not self.is_configured():
            raise LLMError("google is not configured (missing GOOGLE_AI_API_KEY / GEMINI_API_KEY)")
        model_id = model or self._model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent"
        )
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = requests.post(
                url,
                params={"key": self._api_key},
                headers={"Content-Type": "application/json"},
                json=body,
                timeout=90,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Network error: {exc}") from exc

        if resp.status_code == 429:
            raise LLMError("Rate limited (429)")
        if resp.status_code in {401, 403}:
            raise LLMError(f"Auth failed ({resp.status_code})")
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {(resp.text or '')[:300]}")

        payload = resp.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMError("No Gemini candidates")
        parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        return _extract_json(text)


class CloudflareWorkersAIProvider(LLMProvider):
    id = "cloudflare"
    label = "Cloudflare Workers AI"

    def __init__(self) -> None:
        self._account_id = (config.CLOUDFLARE_ACCOUNT_ID or "").strip()
        self._token = (config.CLOUDFLARE_API_TOKEN or "").strip()
        self._model = config.CLOUDFLARE_AI_MODEL

    def is_configured(self) -> bool:
        return bool(self._account_id and self._token)

    def default_model(self) -> str:
        return self._model

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if not self.is_configured():
            raise LLMError(
                "cloudflare is not configured (need CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN)"
            )
        model_id = model or self._model
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{self._account_id}"
            f"/ai/v1/chat/completions"
        )
        return _openai_chat_json(
            url=url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            model=model_id,
            system=system,
            user=user,
            extra={"response_format": {"type": "json_object"}},
        )


class OllamaProvider(LLMProvider):
    id = "ollama"
    label = "Ollama (local)"

    def __init__(self) -> None:
        self._host = config.OLLAMA_HOST.rstrip("/")
        self._model = config.OLLAMA_MODEL

    def is_configured(self) -> bool:
        # Local daemon — "configured" means host is set (always true by default).
        return bool(self._host)

    def default_model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self._host}/api/tags", timeout=2)
            if resp.status_code != 200:
                return False
            names = [m.get("name", "") for m in (resp.json().get("models") or [])]
            if not names:
                return False
            model = self._model
            return any(model in name or name.startswith(model.split(":")[0]) for name in names) or bool(
                names
            )
        except Exception:
            return False

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        if not self.is_available():
            raise LLMError("ollama is not available (daemon down or no models)")
        model_id = model or self._model
        # Prefer HTTP API to avoid requiring the ollama SDK at runtime in CI.
        url = f"{self._host}/api/chat"
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        try:
            resp = requests.post(url, json=body, timeout=180)
        except requests.RequestException as exc:
            raise LLMError(f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(f"HTTP {resp.status_code}: {(resp.text or '')[:300]}")
        payload = resp.json()
        content = ((payload.get("message") or {}).get("content")) or ""
        return _extract_json(str(content))


class FallbackProvider(LLMProvider):
    """Marker provider — research_llm generates the deterministic report itself."""

    id = "fallback"
    label = "Rule-based fallback"

    def is_configured(self) -> bool:
        return True

    def default_model(self) -> str:
        return "rule_based_fallback"

    def is_available(self) -> bool:
        return True

    def chat_json(self, system: str, user: str, model: str | None = None) -> dict[str, Any]:
        raise LLMError("fallback provider does not call an LLM")


def _build_registry() -> dict[str, LLMProvider]:
    return {
        "google": GoogleGeminiProvider(),
        "groq": OpenAICompatibleProvider(
            provider_id="groq",
            label="Groq",
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            model=config.GROQ_MODEL,
        ),
        "openrouter": OpenAICompatibleProvider(
            provider_id="openrouter",
            label="OpenRouter",
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            model=config.OPENROUTER_MODEL,
            extra_headers={
                "HTTP-Referer": "https://trendlabs.app",
                "X-Title": "TrendLab Research",
            },
        ),
        "cloudflare": CloudflareWorkersAIProvider(),
        "cerebras": OpenAICompatibleProvider(
            provider_id="cerebras",
            label="Cerebras",
            api_key=config.CEREBRAS_API_KEY,
            base_url=config.CEREBRAS_BASE_URL,
            model=config.CEREBRAS_MODEL,
        ),
        "mistral": OpenAICompatibleProvider(
            provider_id="mistral",
            label="Mistral",
            api_key=config.MISTRAL_API_KEY,
            base_url=config.MISTRAL_BASE_URL,
            model=config.MISTRAL_MODEL,
        ),
        "ollama": OllamaProvider(),
        "fallback": FallbackProvider(),
    }


# Preferred free-cloud order for TRENDLAB_LLM_PROVIDER=auto
AUTO_CLOUD_ORDER = ("google", "groq", "openrouter", "cloudflare", "cerebras", "mistral")


PROVIDER_IDS = (
    "google",
    "groq",
    "openrouter",
    "cloudflare",
    "cerebras",
    "mistral",
    "ollama",
    "fallback",
)


def get_registry(
    factory: Callable[[], dict[str, LLMProvider]] | None = None,
) -> dict[str, LLMProvider]:
    return (factory or _build_registry)()


def list_provider_status(
    registry: dict[str, LLMProvider] | None = None,
) -> list[dict[str, Any]]:
    reg = registry or get_registry()
    rows = []
    for pid in PROVIDER_IDS:
        p = reg[pid]
        info = p.info()
        rows.append(
            {
                "id": info.id,
                "label": info.label,
                "configured": info.configured if pid != "ollama" else info.available,
                "available": info.available,
                "default_model": info.default_model,
            }
        )
    return rows


def configured_provider_flags(
    registry: dict[str, LLMProvider] | None = None,
) -> dict[str, bool]:
    """Booleans only — never expose secret values."""
    reg = registry or get_registry()
    flags: dict[str, bool] = {}
    for pid in PROVIDER_IDS:
        p = reg[pid]
        if pid == "ollama":
            flags[pid] = p.is_available()
        elif pid == "fallback":
            flags[pid] = True
        else:
            flags[pid] = p.is_configured()
    return flags


def resolve_provider(
    preferred: str | None = None,
    *,
    registry: dict[str, LLMProvider] | None = None,
    allow_ollama: bool = True,
) -> tuple[LLMProvider, str]:
    """Pick a provider. Returns (provider, selection_reason).

    preferred: explicit id, or None/"auto" to use TRENDLAB_LLM_PROVIDER / auto chain.
    """
    reg = registry or get_registry()
    choice = (preferred or config.TRENDLAB_LLM_PROVIDER or "auto").strip().lower()
    if choice in {"", "auto"}:
        choice = "auto"

    if choice == "fallback":
        return reg["fallback"], "explicit:fallback"

    if choice != "auto":
        if choice not in reg:
            raise LLMError(f"Unknown provider id: {choice}")
        provider = reg[choice]
        if choice == "ollama":
            if not allow_ollama or not provider.is_available():
                return reg["fallback"], f"{choice}:unavailable->fallback"
            return provider, f"explicit:{choice}"
        if not provider.is_configured():
            return reg["fallback"], f"{choice}:missing_key->fallback"
        return provider, f"explicit:{choice}"

    # auto: free cloud keys → ollama → fallback
    for pid in AUTO_CLOUD_ORDER:
        p = reg[pid]
        if p.is_configured():
            return p, f"auto:{pid}"
    if allow_ollama:
        ollama = reg["ollama"]
        if ollama.is_available():
            return ollama, "auto:ollama"
    return reg["fallback"], "auto:fallback"


def chat_json_with_provider(
    provider_id: str | None,
    system: str,
    user: str,
    model: str | None = None,
    *,
    registry: dict[str, LLMProvider] | None = None,
    allow_ollama: bool = True,
) -> tuple[dict[str, Any], str, str]:
    """Call chat_json on the resolved provider.

    Returns (data, used_provider_id, source_detail).
    """
    provider, reason = resolve_provider(
        provider_id, registry=registry, allow_ollama=allow_ollama
    )
    if provider.id == "fallback":
        raise LLMError(reason)
    data = provider.chat_json(system, user, model=model)
    used_model = model or provider.default_model()
    return data, provider.id, f"{reason}:{used_model}"
