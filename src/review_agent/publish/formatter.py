"""Comment bodies for the PR (PRD s8; PRD-multi-agent-p0 s9).

Summary comment: risk score, counts by severity, agent status, top-3 issues, gate result.
Inline comment:  ``severity | category``, one-line problem, recommendation, confidence,
detected-by / validated-by agent attribution.
"""

from __future__ import annotations

from typing import List, Optional

from ..context.comments_provider import AI_INLINE_MARKER, AI_SUMMARY_MARKER
from ..models import Finding, GateResult, Severity

_SEV_ORDER = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
_SEV_EMOJI = {Severity.CRITICAL: "🟥", Severity.HIGH: "🟧", Severity.MEDIUM: "🟨", Severity.LOW: "🟦"}


def _rank(f: Finding):
    return (_SEV_ORDER[f.severity], -f.confidence)


def inline_comment(f: Finding) -> str:
    lines = [
        AI_INLINE_MARKER,
        f"**{f.severity.value} | {f.category.value}** — {f.title}",
        "",
        f.description.strip(),
        "",
        f"**Fix:** {f.recommendation.strip()}",
    ]
    if f.suggested_fix:
        lines += ["", "```java", f.suggested_fix.strip(), "```"]
    detected_by = ", ".join(f.reported_by) if f.reported_by else "a specialist agent"
    lines += ["", f"_confidence {f.confidence:.2f} · detected by {detected_by} · "
                   f"validated by Judge Agent_"]
    return "\n".join(lines)


def summary_comment(findings: List[Finding], gate: GateResult, *, provider: str,
                    failed_files: List[str], unplaced: List[Finding],
                    agent_names: Optional[List[str]] = None, tokens_used: int = 0) -> str:
    counts = gate.counts
    icon = "✅" if gate.status == "PASS" else "❌"
    out = [
        AI_SUMMARY_MARKER,
        f"## {icon} AI Review — {gate.status}",
        "",
        f"**Risk score:** {gate.score}/100  ·  "
        f"CRITICAL {counts['CRITICAL']} · HIGH {counts['HIGH']} · "
        f"MEDIUM {counts['MEDIUM']} · LOW {counts['LOW']}",
    ]
    if agent_names:
        out += ["", "**Agents:** " + " · ".join(agent_names) + " · ✓ Judge Agent"]
    if gate.reasons:
        out += ["", "**Gate:** " + "; ".join(gate.reasons)]

    top = sorted(findings, key=_rank)[:3]
    if top:
        out += ["", "### Top issues"]
        for f in top:
            loc = f"`{f.file}`:{f.line}"
            out.append(f"- {_SEV_EMOJI[f.severity]} **{f.severity.value}** {f.title} — {loc}")
    elif not failed_files:
        out += ["", "No issues found on the changed lines. 🎉"]

    if unplaced:
        out += ["", "### Findings without a diff line"]
        for f in unplaced:
            out.append(f"- **{f.severity.value} | {f.category.value}** {f.title} — `{f.file}`")

    if failed_files:
        out += ["", "> ⚠️ The LLM review failed for: " + ", ".join(f"`{p}`" for p in failed_files) +
                ". These files were not reviewed."]

    out += ["", f"<sub>provider: {provider} · {len(findings)} finding(s) confirmed by the "
                 f"Judge Agent, after validation & dedup · {tokens_used} token(s) consumed</sub>"]
    return "\n".join(out)
