"""OpenAI provider -- the real LLM used in the live demo (PRD s6, s13)."""

from __future__ import annotations

import threading

from .base import LLMError


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings):
        missing = [
            k for k, v in {
                "OPENAI_API_KEY": settings.openai_key(),
                "OPENAI_MODEL": settings.openai_model,
            }.items() if not v
        ]
        if missing:
            raise LLMError(f"openai provider missing config: {', '.join(missing)}")

        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("the 'openai' package is required for LLM_PROVIDER=openai") from e

        self._model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._client = OpenAI(
            api_key=settings.openai_key(),
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )
        self.total_tokens = 0
        self._usage_lock = threading.Lock()

    def complete(self, system: str, user: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:  # openai raises many subclasses; treat all as call failure
            raise LLMError(f"OpenAI call failed: {e}") from e

        usage = getattr(resp, "usage", None)
        if usage is not None:
            with self._usage_lock:
                self.total_tokens += getattr(usage, "total_tokens", 0) or 0

        choice = (resp.choices or [None])[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if not content:
            raise LLMError("OpenAI returned an empty response")
        return content
