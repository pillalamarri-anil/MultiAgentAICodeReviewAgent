"""Specialist agent + Judge Agent unit tests (PRD-multi-agent-p0 s12)."""

from review_agent.agents.bug_agent import BUG_AGENT
from review_agent.agents.judge_agent import run_judge
from review_agent.agents.security_agent import SECURITY_AGENT
from review_agent.llm.base import system_prompt
from review_agent.llm.mock import MockProvider
from review_agent.models import AgentResult, Finding

_FINDING = {
    "severity": "HIGH", "category": "SECURITY",
    "file": "A.java", "line": 10, "title": "SQL injection",
    "description": "d", "impact": "i", "recommendation": "r",
    "suggested_fix": None, "evidence": 'the file "A.java"', "confidence": 0.9,
}


def _agent_result(agent: str, category: str, **finding_overrides) -> AgentResult:
    return AgentResult(agent=agent, category=category, file="A.java",
                       findings=[Finding(**dict(_FINDING, **finding_overrides))])


def test_specialist_agent_uses_its_own_prompt_and_returns_agent_result():
    provider = MockProvider(responses={"A.java": {"summary": "s", "findings": [_FINDING]}})
    result = BUG_AGENT.review(provider, system_prompt(), context="some context",
                              target_file="A.java", pr_id=1, repo="o/r",
                              target_branch="main", source_branch="feat")

    assert result.agent == "bug-agent" and result.status == "ok"
    assert len(result.findings) == 1

    _, user = provider.calls[0]
    assert "BUG AGENT" in user and "A.java" in user


def test_specialist_agent_marks_failed_on_bad_json_without_raising():
    provider = MockProvider(raw="{ not valid")
    result = SECURITY_AGENT.review(provider, system_prompt(), context="ctx",
                                   target_file="A.java", pr_id=1, repo="o/r",
                                   target_branch="main", source_branch="feat")

    assert result.status == "failed"
    assert result.error and "repair" in result.error


def test_judge_merges_duplicates_across_agents_and_reports_rejects():
    bug_result = _agent_result("bug-agent", "BUG", title="Unsafe SQL construction")
    sec_result = _agent_result("security-agent", "SECURITY")

    judge_response = {
        "summary": "One confirmed SQL injection.",
        "findings": [dict(_FINDING, reported_by=["security-agent", "bug-agent"])],
        "rejected": [{"title": "Speculative memory leak", "reason": "not evidenced"}],
    }
    provider = MockProvider(judge_response=judge_response)

    result = run_judge(provider, [bug_result, sec_result], pr_id=1, repo="o/r",
                       target_branch="main", source_branch="feat")

    assert result.status == "ok"
    assert len(result.findings) == 1
    assert result.findings[0].reported_by == ["security-agent", "bug-agent"]
    assert len(result.rejected) == 1

    system, user = provider.calls[0]
    assert system.strip().startswith("You are the JUDGE AGENT")
    assert "agent=bug-agent" in user and "agent=security-agent" in user


def test_judge_default_confirms_nothing_when_unconfigured():
    """MockProvider's safe default for an un-primed Judge call: reject everything
    rather than silently letting unvetted findings through (see llm/mock.py)."""
    provider = MockProvider(responses={"A.java": {"summary": "s", "findings": [_FINDING]}})
    result = run_judge(provider, [_agent_result("bug-agent", "BUG")], pr_id=1, repo="o/r",
                       target_branch="main", source_branch="feat")

    assert result.status == "ok"
    assert result.findings == []


def test_judge_fails_after_one_repair_retry_on_bad_json():
    provider = MockProvider(raw="not json")
    result = run_judge(provider, [_agent_result("bug-agent", "BUG")], pr_id=1, repo="o/r",
                       target_branch="main", source_branch="feat")

    assert result.status == "failed"
    assert "repair" in result.error
