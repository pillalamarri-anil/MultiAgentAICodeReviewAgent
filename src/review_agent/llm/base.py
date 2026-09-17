"""LLM provider protocol + prompt loading.

One real provider is used live in the demo (``openai``); ``mock`` exists only for
tests and offline dev (PRD s6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LLMError(RuntimeError):
    """Provider call failed (network, auth, quota, timeout)."""


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response (expected to be a JSON object)."""


def load_prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def system_prompt() -> str:
    """Shared system prompt for the four specialist agents (PRD-multi-agent-p0 s10)."""
    return load_prompt("system.txt")


def user_prompt(*, template: str, context: str, target_file: str, pr_id, repo: str,
                target_branch: str, source_branch: str) -> str:
    """Render one specialist agent's per-file user prompt. ``template`` is the prompt
    file name, e.g. ``"bug_agent.txt"`` (PRD-multi-agent-p0 s6)."""
    return load_prompt(template).format(
        context=context,
        target_file=target_file,
        pr_id=pr_id if pr_id is not None else "?",
        repo=repo or "?",
        target_branch=target_branch or "?",
        source_branch=source_branch or "?",
    )


def judge_system_prompt() -> str:
    return load_prompt("judge_system.txt")


def judge_user_prompt(*, candidates: str, pr_id, repo: str,
                      target_branch: str, source_branch: str) -> str:
    return load_prompt("judge_agent.txt").format(
        candidates=candidates,
        pr_id=pr_id if pr_id is not None else "?",
        repo=repo or "?",
        target_branch=target_branch or "?",
        source_branch=source_branch or "?",
    )


def build_provider(settings):
    provider = (settings.llm_provider or "mock").lower()
    if provider == "mock":
        from .mock import MockProvider

        return MockProvider()
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(settings)
    raise LLMError(f"unknown LLM_PROVIDER: {settings.llm_provider!r} (use 'openai' or 'mock')")
