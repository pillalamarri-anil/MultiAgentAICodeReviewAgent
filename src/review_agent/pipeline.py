"""End-to-end review pipeline (PRD-multi-agent-p0 s9).

diff -> Java context (per file, unchanged)
      -> [Bug | Security | Performance | Quality] agents in parallel, per file
      -> aggregate candidate findings across all files
      -> Judge Agent (once per PR)
      -> validate -> dedup -> score -> publish -> report.

Failures are explicit: an agent whose LLM call or JSON parse failed is recorded as
``status='failed'`` and never counted as a successful review (G5); the other agents'
results for that file are unaffected. If the Judge itself fails, the run aborts before
anything is published (AC8).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import logging as L
from .agents import SPECIALIST_AGENTS
from .agents.judge_agent import run_judge
from .context.builder import build_review_context
from .context.diff_parser import hunk_new_line_span
from .context.models import FileChange, PRInfo
from .context.render import render_for_file
from .context.repo_view import LocalRepoView
from .git_ops import commit_messages, current_sha, resolve_sha, unified_diff
from .llm.base import build_provider, system_prompt
from .models import AgentResult, Finding, FileReviewResult, GateResult, JudgeRunResult, ReviewReport
from .publish.formatter import inline_comment, summary_comment
from .publish.github_client import GitHubClient, GitHubError
from .review.dedup import dedupe_findings
from .review.scoring import evaluate_gate
from .review.validator import validate_findings

_REVIEWABLE_STATUSES = {"added", "modified", "renamed"}

_AGENT_LABELS = {
    "bug-agent": "Bug Agent",
    "security-agent": "Security Agent",
    "performance-agent": "Performance Agent",
    "quality-agent": "Quality/Test Agent",
}


class JudgeFailure(RuntimeError):
    """The Judge Agent's LLM call/parse failed -- the review must abort as
    REVIEW_INCOMPLETE without publishing anything (AC8)."""


@dataclass
class RunInputs:
    repo_path: str
    pr_id: Optional[int] = None
    base: Optional[str] = None
    head: Optional[str] = None
    commit: Optional[str] = None
    overlay_dir: str = ""
    publish: bool = False
    status_url: str = ""
    diff_text: Optional[str] = None  # override: skip GitHub / git and use this diff


def _make_client(settings) -> Optional[GitHubClient]:
    owner, repo = settings.github_owner_repo
    if owner and repo and settings.gh_token():
        return GitHubClient(owner, repo, settings.gh_token(), settings.github_api_url)
    return None


def _gather_pr_info(inp: RunInputs, client: Optional[GitHubClient]) -> PRInfo:
    if client and inp.pr_id is not None:
        pr = client.get_pull_request(inp.pr_id)
        msgs = [m.split("\n")[0] for m in client.get_pr_commits(inp.pr_id)]
        return PRInfo(
            id=inp.pr_id,
            title=pr.get("title") or "",
            description=pr.get("body") or "",
            source_branch=(pr.get("head") or {}).get("ref") or inp.head or "",
            target_branch=(pr.get("base") or {}).get("ref") or inp.base or "",
            commit_messages=msgs,
        )
    return PRInfo(
        id=inp.pr_id,
        title="",
        description="",
        source_branch=inp.head or "",
        target_branch=inp.base or "",
        commit_messages=(commit_messages(inp.repo_path, inp.base, inp.head)
                         if inp.base and inp.head else []),
    )


def _resolve_diff(inp: RunInputs, client: Optional[GitHubClient]) -> str:
    if inp.diff_text is not None:
        L.step("using supplied diff text")
        return inp.diff_text
    if client and inp.pr_id is not None:
        L.step(f"fetching PR #{inp.pr_id} diff from GitHub")
        return client.get_diff(inp.pr_id)
    if not (inp.base and inp.head):
        raise ValueError("need --pr (with GitHub creds) or both --base and --head")
    L.step(f"computing local diff {inp.base}...{inp.head}")
    return unified_diff(inp.repo_path, inp.base, inp.head)


def _resolve_commit(inp: RunInputs, client: Optional[GitHubClient], pr_info: PRInfo) -> Optional[str]:
    if inp.commit:
        return inp.commit
    if client and inp.pr_id is not None:
        try:
            return (client.get_pull_request(inp.pr_id).get("head") or {}).get("sha")
        except GitHubError:
            return None
    for attempt in (lambda: resolve_sha(inp.repo_path, inp.head) if inp.head else None,
                    lambda: current_sha(inp.repo_path)):
        try:
            sha = attempt()
            if sha:
                return sha
        except Exception:
            continue
    return None


def _spans_by_file(changes: List[FileChange]) -> Dict[str, list]:
    return {c.file: [hunk_new_line_span(h) for h in c.hunks] for c in changes}


def _context_summary(ctx) -> dict:
    """Everything the LLM actually saw, for the demo / audit trail (not raw content)."""
    return {
        "changed_files": [
            {"file": c.file, "status": c.status, "language": c.language,
             "hunks": len(c.hunks)}
            for c in ctx.changes
        ],
        "rules_loaded": len(ctx.rules),
        "knowledge_docs": [
            {"path": k.path, "selector": k.selector, "tokens": k.tokens}
            for k in ctx.knowledge
        ],
        "code_context": [
            {"file": c.file, "reason": c.reason, "symbol": c.symbol, "tokens": c.tokens}
            for c in ctx.code
        ],
        "comments_included": len(ctx.comments),
    }


def run(settings, inp: RunInputs) -> ReviewReport:
    started = time.monotonic()
    client = _make_client(settings)
    provider = build_provider(settings)
    specialist_model = settings.specialist_model()
    judge_model = settings.judge_model()
    L.step(f"LLM provider: {provider.name} (specialists: {specialist_model}, judge: {judge_model})")

    diff_text = _resolve_diff(inp, client)
    pr_info = _gather_pr_info(inp, client)
    raw_comments = client.get_pr_comments(inp.pr_id) if (client and inp.pr_id is not None) else None

    repo = LocalRepoView(inp.repo_path, inp.overlay_dir)
    ctx = build_review_context(pr_info, diff_text, repo, settings, raw_comments)
    L.step(f"diff parsed: {len(ctx.changes)} changed file(s); "
           f"context {ctx.budget.get('used')}/{ctx.budget.get('limit')} tok, "
           f"{len(ctx.budget.get('dropped', []))} dropped")
    L.step("changed files: " + ", ".join(c.file for c in ctx.changes))
    if ctx.knowledge:
        L.step("docs used: " + ", ".join(f"{k.path} ({k.selector})" for k in ctx.knowledge))
    else:
        L.step("docs used: none")
    L.step(f"rules loaded: {len(ctx.rules)}")
    if ctx.code:
        by_reason: Dict[str, int] = {}
        for slice_ in ctx.code:
            by_reason[slice_.reason] = by_reason.get(slice_.reason, 0) + 1
        L.step("code context: " + ", ".join(f"{n} {r}" for r, n in by_reason.items())
               + " -- " + ", ".join(sorted({s.file for s in ctx.code})))

    commit_sha = _resolve_commit(inp, client, pr_info)

    if inp.publish and client and commit_sha:
        _safe(lambda: client.set_build_status(commit_sha, "pending", inp.status_url,
                                              description="AI review running"))

    # --- specialist agents: 4 in parallel per file (PRD-multi-agent-p0 s6, s9) ---
    system = system_prompt()
    owner, repo_name = settings.github_owner_repo
    repo_slug = f"{owner}/{repo_name}" if owner else (settings.github_repository or inp.repo_path)

    file_inputs: Dict[str, str] = {}
    file_review_results: List[FileReviewResult] = []
    all_agent_results: List[AgentResult] = []

    with ThreadPoolExecutor(max_workers=len(SPECIALIST_AGENTS)) as executor:
        for change in ctx.changes:
            if change.status not in _REVIEWABLE_STATUSES or not change.hunks:
                continue
            ctx_text = render_for_file(ctx, change.file)
            file_inputs[change.file] = ctx_text
            L.step(f"[{change.file}] running specialist agents in parallel: "
                   + ", ".join(a.name for a in SPECIALIST_AGENTS))
            file_results = _run_specialists(executor, provider, system, change.file, ctx_text,
                                            pr_info, repo_slug, specialist_model)
            for r in file_results:
                if r.status == "failed":
                    L.warn(f"[{change.file}] {r.agent}: FAILED — {r.error}")
                else:
                    L.step(f"[{change.file}] {r.agent}: {len(r.findings)} raw finding(s)"
                           + (" (repaired)" if r.repaired else ""))
            all_agent_results.extend(file_results)
            ok_agents = [r for r in file_results if r.status == "ok"]
            file_review_results.append(FileReviewResult(
                file=change.file,
                status="ok" if ok_agents else "failed",
                repaired=any(r.repaired for r in ok_agents),
                error=None if len(ok_agents) == len(file_results) else
                    "; ".join(f"{r.agent}: {r.error}" for r in file_results if r.status == "failed"),
            ))

    failed_files = [r.file for r in file_review_results if r.status == "failed"]
    all_failed = bool(file_review_results) and all(r.status == "failed" for r in file_review_results)

    # --- Judge Agent: once per PR, over every file's candidate findings ---------
    candidate_findings = [f for r in all_agent_results for f in r.findings]
    if candidate_findings:
        L.step(f"Judge Agent: reviewing {len(candidate_findings)} candidate finding(s) "
               f"from {len(all_agent_results)} agent run(s)")
        judge_result = run_judge(provider, all_agent_results, pr_id=pr_info.id, repo=repo_slug,
                                 target_branch=pr_info.target_branch,
                                 source_branch=pr_info.source_branch, model=judge_model)
        if judge_result.status == "failed":
            L.error(f"Judge Agent failed: {judge_result.error}")
            raise JudgeFailure(f"REVIEW_INCOMPLETE: Judge Agent failed: {judge_result.error}")
        L.step(f"Judge Agent: confirmed {len(judge_result.findings)}, "
               f"rejected {len(judge_result.rejected)}"
               + (" (repaired)" if judge_result.repaired else ""))
    else:
        L.step("Judge Agent: skipped (no candidate findings)")
        judge_result = JudgeRunResult(status="ok", summary="", findings=[], rejected=[])

    # --- validate -> dedup -> score ------------------------------
    kept, dropped = validate_findings(judge_result.findings, ctx.changes, file_inputs,
                                      settings.min_confidence)
    L.step(f"validation: {len(judge_result.findings)} -> {len(kept)} finding(s) ({len(dropped)} dropped)")
    findings = dedupe_findings(kept)
    if len(findings) != len(kept):
        L.step(f"dedup: {len(kept)} -> {len(findings)} finding(s)")

    gate = evaluate_gate(findings, settings)
    if all_failed:
        gate = GateResult(status="CHANGES REQUESTED", exit_code=1, score=gate.score,
                          counts=gate.counts,
                          reasons=gate.reasons + ["every changed file failed LLM review"])
    L.step(f"gate: {gate.status} (score {gate.score}, exit {gate.exit_code})")

    report = ReviewReport(
        repo=repo_slug, pr=pr_info.id, base=pr_info.target_branch, head=pr_info.source_branch,
        commit=commit_sha, provider=provider.name,
        review_ok=not failed_files,
        summary=_overall_summary(judge_result.summary, gate),
        files_reviewed=file_review_results, findings=findings, dropped=dropped,
        agent_results=all_agent_results, judge_rejected=judge_result.rejected, gate=gate,
        duration_seconds=round(time.monotonic() - started, 2),
        context_budget=ctx.budget,
        context_summary=_context_summary(ctx),
        tokens_used=provider.total_tokens,
    )

    if inp.publish and client:
        _publish(client, pr_info, commit_sha, findings, gate, provider.name, failed_files,
                 inp.status_url, _spans_by_file(ctx.changes), all_agent_results,
                 provider.total_tokens)
    elif inp.publish:
        L.warn("--publish set but no GitHub client (need GITHUB_TOKEN + GITHUB_REPOSITORY); skipping")

    L.step(f"done in {report.duration_seconds}s, {report.tokens_used} token(s) consumed")
    return report


def _run_specialists(executor: ThreadPoolExecutor, provider, system: str, target_file: str,
                     ctx_text: str, pr_info: PRInfo, repo_slug: str,
                     model: Optional[str] = None) -> List[AgentResult]:
    """Run the 4 specialist agents for one file concurrently (threads, since the LLM
    provider call is blocking network I/O). One agent failing never stops the others
    (G5)."""
    future_to_agent = {
        executor.submit(agent.review, provider, system, context=ctx_text, target_file=target_file,
                        pr_id=pr_info.id, repo=repo_slug, target_branch=pr_info.target_branch,
                        source_branch=pr_info.source_branch, model=model): agent
        for agent in SPECIALIST_AGENTS
    }
    results: List[AgentResult] = []
    for future, agent in future_to_agent.items():
        try:
            results.append(future.result())
        except Exception as e:  # pragma: no cover -- SpecialistAgent.review already catches LLMError
            results.append(AgentResult(agent=agent.name, category=agent.category, file=target_file,
                                       findings=[], status="failed", error=str(e)))
    return results


def _overall_summary(judge_summary: str, gate: GateResult) -> str:
    return (f"{gate.status} — risk {gate.score}/100. " + (judge_summary or "")).strip()


def _agent_status_labels(all_agent_results: List[AgentResult]) -> List[str]:
    ok_by_agent = {}
    for r in all_agent_results:
        ok_by_agent[r.agent] = ok_by_agent.get(r.agent, False) or (r.status == "ok")
    return [f"{'✓' if ok_by_agent.get(name) else '✗'} {label}"
            for name, label in _AGENT_LABELS.items()]


def _publish(client, pr_info, commit_sha, findings, gate, provider_name, failed_files,
             status_url, spans_by_file, all_agent_results, tokens_used):
    if pr_info.id is not None:
        unplaced: List[Finding] = []
        for f in findings:
            placeable = commit_sha and any(lo <= f.line <= hi
                                           for lo, hi in spans_by_file.get(f.file, []))
            if placeable:
                try:
                    client.post_inline_comment(pr_info.id, commit_sha, f.file, f.line,
                                               inline_comment(f))
                    continue
                except GitHubError as e:
                    L.warn(f"inline comment failed for {f.file}:{f.line}: {e}")
            unplaced.append(f)
        body = summary_comment(findings, gate, provider=provider_name,
                               failed_files=failed_files, unplaced=unplaced,
                               agent_names=_agent_status_labels(all_agent_results),
                               tokens_used=tokens_used)
        _safe(lambda: client.post_summary_comment(pr_info.id, body))
        L.step("posted summary + inline comments")
    else:
        L.warn("no PR id -> commit status only")

    if commit_sha:
        state = "success" if gate.exit_code == 0 else "failure"
        _safe(lambda: client.set_build_status(
            commit_sha, state, status_url, description=f"{gate.status} · risk {gate.score}/100"))


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # publishing must never crash the run
        L.warn(f"publish step failed: {e}")
        return None
