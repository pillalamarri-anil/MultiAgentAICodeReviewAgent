"""Deterministic mock provider -- tests and offline dev only (PRD s6, AC11).

Never used in the demo. Given a mapping of ``{path-substring: response}`` it returns
the matching canned JSON; otherwise a clean (no-findings) review. ``response`` may be a
JSON string, or a dict that will be ``json.dumps``-ed.

The Judge Agent call is distinguished by its distinct system prompt (starts with "You
are the JUDGE AGENT", see ``prompts/judge_system.txt``) rather than by scanning the
user text, so it doesn't interfere with the per-file ``responses`` matching used by the
four specialist agents. With no ``judge_response`` configured, the mock Judge confirms
nothing (safe default) -- tests that expect findings to survive the Judge must set
``judge_response`` explicitly (PRD-multi-agent-p0 s12).
"""

from __future__ import annotations

import json
import re
import threading
from typing import Dict, Optional, Union

from ..context.tokens import count_tokens

_TARGET_RE = re.compile(r'the file "([^"]+)"')
_JUDGE_SYSTEM_MARKER = "You are the JUDGE AGENT"

Response = Union[str, dict]


class MockProvider:
    name = "mock"

    def __init__(self, responses: Optional[Dict[str, Response]] = None,
                 default: Optional[Response] = None, raw: Optional[str] = None,
                 judge_response: Optional[Response] = None):
        self.responses = responses or {}
        self.default = default
        self.raw = raw
        self.judge_response = judge_response
        self.calls = []  # list of (system, user) for assertions
        self.total_tokens = 0
        self._usage_lock = threading.Lock()

    def _match(self, user: str) -> Optional[Response]:
        m = _TARGET_RE.search(user)
        target = m.group(1) if m else ""
        for key, resp in self.responses.items():
            if key in target or key in user:
                return resp
        return self.default

    def complete(self, system: str, user: str, *, model: Optional[str] = None) -> str:
        self.calls.append((system, user))
        if self.raw is not None:
            content = self.raw
        elif system.strip().startswith(_JUDGE_SYSTEM_MARKER):
            resp = self.judge_response
            if resp is None:
                resp = {"summary": "No candidate findings to confirm.",
                        "findings": [], "rejected": []}
            content = resp if isinstance(resp, str) else json.dumps(resp)
        else:
            resp = self._match(user)
            if resp is None:
                resp = {"summary": "No issues found in the provided context.", "findings": []}
            content = resp if isinstance(resp, str) else json.dumps(resp)

        # No real API usage to report -- estimate with the same offline approximation
        # used for context budgeting, so demo/offline runs still show a token count.
        with self._usage_lock:
            self.total_tokens += count_tokens(system) + count_tokens(user) + count_tokens(content)
        return content
