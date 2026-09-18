"""OpenAI provider -- the real LLM used in the live demo (PRD s6, s13).

A single provider instance is shared by every agent; ``complete()`` takes an optional
per-call ``model`` override so the four specialist agents and the Judge Agent can each
use a different model (PRD-multi-agent-p0: cheaper/faster model for the high-volume
specialist calls, a stronger one for the single, more consequential Judge call).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from ..context.tokens import count_tokens
from .base import LLMError
from .rate_limit import TokenBucket

# Reasoning-family models (gpt-5, o1/o3/o4, ...) reject `temperature` overrides and
# use `max_completion_tokens` instead of the legacy `max_tokens` param.
_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Rough pre-call output-token estimate for the rate limiter, corrected against real
# usage afterwards (TokenBucket.adjust) -- reasoning models burn more on hidden
# "thinking" tokens before any visible output, so they get a bigger reservation.
_REASONING_OUTPUT_ESTIMATE = 3000
_STANDARD_OUTPUT_ESTIMATE = 800


def _is_reasoning_model(model: str) -> bool:
    return model.startswith(_REASONING_MODEL_PREFIXES)


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings):
        if not settings.openai_key():
            raise LLMError("openai provider missing config: OPENAI_API_KEY")

        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("the 'openai' package is required for LLM_PROVIDER=openai") from e

        self._default_model = settings.openai_model
        self._max_tokens = settings.llm_max_tokens
        self._reasoning_effort = settings.openai_reasoning_effort
        self._client = OpenAI(
            api_key=settings.openai_key(),
            base_url=settings.openai_base_url or None,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.total_tokens = 0
        self._usage_lock = threading.Lock()

        # One bucket per role/model, so the Judge's budget is never stolen by a burst
        # of specialist calls (or vice versa) -- see PRD note in llm/rate_limit.py.
        self._buckets: Dict[str, TokenBucket] = {}
        specialist_model = settings.specialist_model()
        judge_model = settings.judge_model()
        if specialist_model and settings.openai_specialist_tpm_limit:
            self._buckets[specialist_model] = TokenBucket(settings.openai_specialist_tpm_limit)
        if judge_model and settings.openai_judge_tpm_limit:
            self._buckets.setdefault(judge_model, TokenBucket(settings.openai_judge_tpm_limit))

    def complete(self, system: str, user: str, *, model: Optional[str] = None) -> str:
        resolved_model = model or self._default_model
        if not resolved_model:
            raise LLMError("no OpenAI model configured (set OPENAI_MODEL, "
                           "OPENAI_SPECIALIST_MODEL, or OPENAI_JUDGE_MODEL)")

        kwargs = dict(
            model=resolved_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        reasoning = _is_reasoning_model(resolved_model)
        if reasoning:
            # Reasoning tokens are billed against this same budget and are invisible
            # to us -- too tight a cap means the model can burn it all on hidden
            # reasoning and return empty content. Give it headroom on top of whatever
            # LLM_MAX_TOKENS is set to.
            kwargs["max_completion_tokens"] = max(self._max_tokens, 8000)
            kwargs["reasoning_effort"] = self._reasoning_effort
        else:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = self._max_tokens

        bucket = self._buckets.get(resolved_model)
        estimate = 0
        if bucket is not None:
            estimate = (count_tokens(system) + count_tokens(user)
                       + (_REASONING_OUTPUT_ESTIMATE if reasoning else _STANDARD_OUTPUT_ESTIMATE))
            bucket.acquire(estimate)

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # openai raises many subclasses; treat all as call failure
            raise LLMError(f"OpenAI call failed: {e}") from e

        usage = getattr(resp, "usage", None)
        actual_tokens = getattr(usage, "total_tokens", 0) or 0 if usage is not None else 0
        if usage is not None:
            with self._usage_lock:
                self.total_tokens += actual_tokens
        if bucket is not None:
            bucket.adjust(estimate, actual_tokens)

        choice = (resp.choices or [None])[0]
        content = getattr(getattr(choice, "message", None), "content", None)
        if not content:
            raise LLMError("OpenAI returned an empty response")
        return content
