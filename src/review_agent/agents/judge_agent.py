"""Judge Agent -- validates, dedupes across agents, and re-scores specialist findings
(PRD-multi-agent-p0 s7). Runs ONCE per PR (called by the orchestrator only after every
file's specialist agents have completed), not once per file.

It only ever sees the ``evidence`` snippet already embedded in each candidate
``Finding`` -- never full file content again -- which keeps the Judge prompt small
regardless of PR size. Whether a finding's file/line actually appears in the diff is a
separate, deterministic check that runs after the Judge (``review.validator``); the
Judge's job is credibility, duplication, severity and confidence, not diff placement.
"""

from __future__ import annotations

from typing import List

from ..llm.base import LLMProvider, judge_system_prompt, judge_user_prompt
from ..llm.contract import judge_review
from ..models import AgentResult, JudgeRunResult


def _render_candidates(agent_results: List[AgentResult]) -> str:
    lines = []
    n = 0
    for r in agent_results:
        for f in r.findings:
            n += 1
            lines.append(
                f"[{n}] agent={r.agent} severity={f.severity.value} category={f.category.value} "
                f"file={f.file} line={f.line} confidence={f.confidence:.2f}\n"
                f"    title: {f.title}\n"
                f"    description: {f.description}\n"
                f"    impact: {f.impact}\n"
                f"    recommendation: {f.recommendation}\n"
                f"    evidence: {f.evidence}"
            )
    return "\n".join(lines) if lines else "(no candidate findings)"


def run_judge(provider: LLMProvider, agent_results: List[AgentResult], *, pr_id, repo: str,
             target_branch: str, source_branch: str) -> JudgeRunResult:
    system = judge_system_prompt()
    user = judge_user_prompt(candidates=_render_candidates(agent_results), pr_id=pr_id, repo=repo,
                             target_branch=target_branch, source_branch=source_branch)
    return judge_review(provider, system, user)
