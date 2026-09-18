"""Environment-driven configuration (PRD s13).

Secrets come only from the environment / Jenkins / GitHub Actions credentials and are
never logged -- ``repr(Settings)`` scrubs them.
"""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_FIELDS = {"openai_api_key", "github_token"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider ---------------------------------------------------
    llm_provider: str = "mock"
    openai_api_key: Optional[SecretStr] = None
    openai_model: Optional[str] = "gpt-4o"  # fallback used when a role-specific model isn't set
    openai_specialist_model: Optional[str] = "gpt-4.1-mini"  # bug/security/performance/quality agents
    openai_judge_model: Optional[str] = "gpt-5"  # Judge Agent (once per PR)
    openai_base_url: Optional[str] = None
    llm_max_tokens: int = 4000
    llm_timeout_seconds: int = 90
    # reasoning-family models (gpt-5, o1/o3/o4) only: "minimal"|"low"|"medium"|"high".
    # Lower effort means fewer hidden reasoning tokens billed per call -- the Judge's
    # job here (dedup/classify an already-distilled candidate list) doesn't need deep
    # reasoning, so a low default keeps cost down without an empty-response risk.
    openai_reasoning_effort: str = "low"
    llm_max_retries: int = 5  # SDK-level backoff retries per call (429/5xx/timeouts)
    # Client-side token-bucket caps (tokens/minute), keyed by role, so specialist calls
    # are paced under the account's TPM limit instead of bursting 4-at-a-time and
    # relying on reactive retries. None disables pacing for that role.
    openai_specialist_tpm_limit: Optional[int] = 25000
    openai_judge_tpm_limit: Optional[int] = 25000

    # --- GitHub ------------------------------------------------------
    github_token: Optional[SecretStr] = None
    github_repository: Optional[str] = None  # "owner/repo"
    github_api_url: str = "https://api.github.com"

    # --- Review knobs ---------------------------------------------
    min_confidence: float = 0.75
    max_context_tokens: int = 8000

    # --- Context builder knobs (PRD s5 / context-prep PRD) ------
    max_knowledge_tokens: int = 3500
    max_code_tokens: int = 6000
    always_include_docs: Tuple[str, ...] = ("docs/ARCHITECTURE.md",)
    context_docs_glob: str = "docs/**/*.md"
    schema_doc: str = "docs/schema.md"
    rules_file: str = "review-rules.yaml"
    surrounding_lines: int = 40
    full_method_lines: int = 120
    full_file_max_lines: int = 400

    # --- Quality gate (PRD s7) --------------------------------
    max_critical: int = 0
    max_high: int = 0
    max_medium: int = 5

    # ------------------------------------------------------------------
    @property
    def github_owner_repo(self) -> Tuple[Optional[str], Optional[str]]:
        if not self.github_repository or "/" not in self.github_repository:
            return None, None
        owner, _, repo = self.github_repository.partition("/")
        return owner, repo

    def openai_key(self) -> Optional[str]:
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    def specialist_model(self) -> Optional[str]:
        return self.openai_specialist_model or self.openai_model

    def judge_model(self) -> Optional[str]:
        return self.openai_judge_model or self.openai_model

    def gh_token(self) -> Optional[str]:
        return self.github_token.get_secret_value() if self.github_token else None

    def __repr__(self) -> str:  # never leak secrets into logs / tracebacks
        parts = []
        for name, value in self.__dict__.items():
            if name in _SECRET_FIELDS:
                parts.append(f"{name}={'***set***' if value else 'None'}")
            else:
                parts.append(f"{name}={value!r}")
        return f"Settings({', '.join(parts)})"

    __str__ = __repr__


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
