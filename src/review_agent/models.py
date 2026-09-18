"""Strict, Pydantic-validated data model for the review (PRD s6, s7).

The LLM returns exactly ``LLMReview``; the risk score is never supplied by the model
-- it is computed in code (see ``review.scoring``).
"""

from __future__ import annotations

import enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Category(str, enum.Enum):
    BUG = "BUG"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    ERROR_HANDLING = "ERROR_HANDLING"
    CONCURRENCY = "CONCURRENCY"
    MAINTAINABILITY = "MAINTAINABILITY"
    TESTING = "TESTING"


class Finding(BaseModel):
    """One issue on one changed line. Matches the PRD s6 JSON contract."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    category: Category
    file: str
    line: int = Field(ge=0)
    title: str
    description: str
    impact: str
    recommendation: str
    suggested_fix: Optional[str] = None
    evidence: str = Field(description="verbatim snippet copied from the input")
    confidence: float = Field(ge=0.0, le=1.0)
    reported_by: List[str] = Field(
        default_factory=list,
        description="agent name(s) that originally reported this finding; filled in by "
                    "the Judge Agent only, never by a specialist (PRD-multi-agent-p0 s7)",
    )

    @field_validator("title", "description", "evidence")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must not be blank")
        return v


class LLMReview(BaseModel):
    """The complete, schema-valid object returned by one LLM call."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    findings: List[Finding] = Field(default_factory=list)


class FileReviewResult(BaseModel):
    """Per-file outcome. ``status='failed'`` means the LLM call or parse failed and
    this file's review is NOT to be reported as successful (PRD s4)."""

    file: str
    status: Literal["ok", "failed", "skipped"]
    summary: str = ""
    findings: List[Finding] = Field(default_factory=list)
    error: Optional[str] = None
    repaired: bool = False


class AgentResult(BaseModel):
    """One specialist agent's output for one file (PRD-multi-agent-p0 s6, s16).

    ``status='failed'`` means that agent's LLM call/parse failed for this file; other
    agents' results for the same file are unaffected (G5)."""

    agent: str
    category: str
    file: str
    findings: List[Finding] = Field(default_factory=list)
    status: Literal["ok", "failed"] = "ok"
    error: Optional[str] = None
    repaired: bool = False


class RejectedFinding(BaseModel):
    title: str
    reason: str


class JudgeReview(BaseModel):
    """The complete, schema-valid object returned by the Judge Agent's LLM call
    (PRD-multi-agent-p0 s7, s20)."""

    model_config = ConfigDict(extra="ignore")

    summary: str
    findings: List[Finding] = Field(default_factory=list)
    rejected: List[RejectedFinding] = Field(default_factory=list)


class JudgeRunResult(BaseModel):
    """Outcome of the (single, PR-level) Judge call. ``status='failed'`` means the run
    must abort as REVIEW_INCOMPLETE and publish nothing (PRD-multi-agent-p0 s9, AC8)."""

    status: Literal["ok", "failed"]
    summary: str = ""
    findings: List[Finding] = Field(default_factory=list)
    rejected: List[RejectedFinding] = Field(default_factory=list)
    error: Optional[str] = None
    repaired: bool = False


class DroppedFinding(BaseModel):
    finding: Finding
    reason: str


class GateResult(BaseModel):
    status: Literal["PASS", "CHANGES REQUESTED"]
    exit_code: int
    score: int
    counts: dict
    reasons: List[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    """The always-written ``review-report.json`` artifact (PRD s4 step 9)."""

    model_config = ConfigDict(extra="ignore")

    repo: Optional[str] = None
    pr: Optional[int] = None
    base: Optional[str] = None
    head: Optional[str] = None
    commit: Optional[str] = None

    provider: str = "mock"
    review_ok: bool = True
    summary: str = ""

    files_reviewed: List[FileReviewResult] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    dropped: List[DroppedFinding] = Field(default_factory=list)

    agent_results: List[AgentResult] = Field(
        default_factory=list, description="raw per-agent, per-file output before the Judge (PRD-multi-agent-p0 s9)")
    judge_rejected: List[RejectedFinding] = Field(default_factory=list)

    gate: Optional[GateResult] = None
    duration_seconds: float = 0.0
    context_budget: dict = Field(default_factory=dict)
    context_summary: dict = Field(default_factory=dict)
    tokens_used: int = Field(
        default=0, description="cumulative LLM tokens consumed across all agent + Judge calls")
