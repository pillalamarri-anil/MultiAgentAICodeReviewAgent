"""AC11: the full pipeline runs offline on the mock provider with no network."""

import sys

import pytest

from review_agent import pipeline as pipeline_mod
from review_agent.config import Settings
from review_agent.pipeline import JudgeFailure, RunInputs, run

REPO_FILE = "src/main/java/com/example/repository/OrderRepository.java"

_NATIVE_QUERY_FINDING = {
    "severity": "HIGH",
    "category": "SECURITY",
    "file": REPO_FILE,
    "line": 17,
    "title": "String-concatenated native SQL query",
    "description": "findNewOrders builds a native SQL string by concatenation.",
    "impact": "SQL injection risk and brittle queries.",
    "recommendation": "Use a bound parameter and JPQL.",
    "suggested_fix": None,
    "evidence": "nativeQuery = true",
    "confidence": 0.93,
}

# same line-bucket (17//5 == 18//5), same category, title normalizes identically -> merged
_DUP_FINDING = dict(_NATIVE_QUERY_FINDING, line=18,
                    title="String-concatenated  native SQL  query!!", confidence=0.88)


def _settings():
    return Settings(_env_file=None, llm_provider="mock", min_confidence=0.75,
                    max_context_tokens=8000, max_critical=0, max_high=0, max_medium=5)


def _patch_provider(monkeypatch, responses, judge_response=None):
    from review_agent.llm.mock import MockProvider

    monkeypatch.setattr(pipeline_mod, "build_provider",
                        lambda s: MockProvider(responses=responses, judge_response=judge_response))


def test_run_flags_finding_and_fails_gate(monkeypatch, sample_repo, diff):
    # every specialist agent returns the same candidate (file-keyed mock matching is
    # agent-agnostic) -- the Judge is primed to confirm exactly one merged finding,
    # exercising the same "4 agents -> Judge -> 1 finding" corroboration the real
    # pipeline demonstrates (PRD-multi-agent-p0 s7, s15).
    _patch_provider(monkeypatch, {
        REPO_FILE: {"summary": "native query issue", "findings": [_NATIVE_QUERY_FINDING, _DUP_FINDING]},
    }, judge_response={
        "summary": "One SQL injection risk confirmed.",
        "findings": [dict(_NATIVE_QUERY_FINDING, reported_by=["security-agent", "bug-agent"])],
        "rejected": [],
    })

    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("repo_change.diff"), publish=False,
    ))

    assert len(report.findings) == 1
    f = report.findings[0]
    assert f.file == REPO_FILE and f.category.value == "SECURITY"
    assert f.reported_by == ["security-agent", "bug-agent"]

    assert report.gate.status == "CHANGES REQUESTED"
    assert report.gate.exit_code == 1
    assert report.review_ok is True
    assert "openai" not in sys.modules and "anthropic" not in sys.modules


def test_run_passes_when_clean(monkeypatch, sample_repo, diff):
    _patch_provider(monkeypatch, {})  # mock returns no findings
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("method_edit.diff"), publish=False,
    ))
    assert report.findings == []
    assert report.gate.status == "PASS" and report.gate.exit_code == 0


def test_run_reports_tokens_consumed(monkeypatch, sample_repo, diff):
    _patch_provider(monkeypatch, {})  # mock still makes real LLM calls, just no findings
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("method_edit.diff"), publish=False,
    ))
    assert report.tokens_used > 0


def test_run_drops_unevidenced_finding(monkeypatch, sample_repo, diff):
    # the Judge confirms it (LLM-level check missed the fabricated evidence) -- the
    # deterministic validator that runs AFTER the Judge is the backstop that catches it
    # (PRD-multi-agent-p0 s7 "defense in depth").
    bad = dict(_NATIVE_QUERY_FINDING, evidence="text that is nowhere in the context")
    _patch_provider(monkeypatch, {REPO_FILE: {"summary": "x", "findings": [bad]}},
                    judge_response={"summary": "x", "findings": [bad], "rejected": []})
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("repo_change.diff"), publish=False,
    ))
    assert report.findings == []
    assert len(report.dropped) == 1
    assert "verbatim" in report.dropped[0].reason


def test_run_invokes_all_four_specialist_agents_per_file(monkeypatch, sample_repo, diff):
    """AC1/AC2: Bug, Security, Performance and Quality/Test agents all run for the
    same changed file (PRD-multi-agent-p0 s13)."""
    _patch_provider(monkeypatch, {REPO_FILE: {"summary": "x", "findings": []}},
                    judge_response={"summary": "clean", "findings": [], "rejected": []})
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("repo_change.diff"), publish=False,
    ))
    agent_names = {r.agent for r in report.agent_results}
    assert agent_names == {"bug-agent", "security-agent", "performance-agent", "quality-agent"}
    assert all(r.status == "ok" and r.file == REPO_FILE for r in report.agent_results)


def test_run_skips_judge_when_no_candidate_findings(monkeypatch, sample_repo, diff):
    """Judge is not called at all when every specialist agent came back clean --
    matters for AC8/G5: an unconfigured/failing Judge must not abort a clean review."""
    _patch_provider(monkeypatch, {})  # no findings from any specialist, no judge_response set
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("method_edit.diff"), publish=False,
    ))
    assert report.findings == []
    assert report.gate.status == "PASS"


def test_run_raises_judge_failure_and_publishes_nothing(monkeypatch, sample_repo, diff):
    """AC8: if the Judge Agent's response is invalid even after one repair retry, the
    run must abort as REVIEW_INCOMPLETE rather than publish unvalidated findings."""
    _patch_provider(monkeypatch, {REPO_FILE: {"summary": "x", "findings": [_NATIVE_QUERY_FINDING]}},
                    judge_response="not valid json")
    with pytest.raises(JudgeFailure):
        run(_settings(), RunInputs(
            repo_path=sample_repo, base="main", head="feat/x",
            diff_text=diff("repo_change.diff"), publish=False,
        ))


def test_run_marks_review_failed_on_bad_json(monkeypatch, sample_repo, diff):
    from review_agent.llm.mock import MockProvider

    monkeypatch.setattr(pipeline_mod, "build_provider",
                        lambda s: MockProvider(raw="{ not valid json"))
    report = run(_settings(), RunInputs(
        repo_path=sample_repo, base="main", head="feat/x",
        diff_text=diff("repo_change.diff"), publish=False,
    ))
    assert report.review_ok is False
    assert report.gate.exit_code == 1
    assert any(fr.status == "failed" for fr in report.files_reviewed)
