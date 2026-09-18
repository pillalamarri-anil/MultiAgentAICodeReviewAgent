"""Specialist review agent (PRD-multi-agent-p0 s5, s6).

Every specialist agent reviews the SAME per-file context the single-agent P0 already
built (``context/`` is untouched); only the prompt template and category label differ.
Reuses the existing strict-JSON-contract + one-repair-retry mechanism
(``llm.contract.review_file``) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..llm.base import LLMProvider, user_prompt
from ..llm.contract import review_file
from ..models import AgentResult


@dataclass(frozen=True)
class SpecialistAgent:
    name: str          # e.g. "bug-agent"
    category: str      # report label only -- individual findings still use models.Category
    prompt_file: str   # e.g. "bug_agent.txt"

    def review(self, provider: LLMProvider, system: str, *, context: str, target_file: str,
              pr_id, repo: str, target_branch: str, source_branch: str,
              model: Optional[str] = None) -> AgentResult:
        user = user_prompt(template=self.prompt_file, context=context, target_file=target_file,
                           pr_id=pr_id, repo=repo, target_branch=target_branch,
                           source_branch=source_branch)
        res = review_file(provider, system, user, target_file, model=model)
        if res.status == "failed":
            return AgentResult(agent=self.name, category=self.category, file=target_file,
                               findings=[], status="failed", error=res.error, repaired=res.repaired)
        return AgentResult(agent=self.name, category=self.category, file=target_file,
                           findings=list(res.findings), status="ok", repaired=res.repaired)
