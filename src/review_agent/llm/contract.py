"""Strict JSON contract enforcement (PRD s6).

``review_file`` performs one LLM call, validates the response against ``LLMReview``,
and on failure performs exactly ONE repair retry. If the retry is still invalid the
file's review is marked failed -- never reported as succeeded (PRD s4).
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from pydantic import ValidationError

from ..models import FileReviewResult, JudgeReview, JudgeRunResult, LLMReview
from .base import LLMError, LLMProvider

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_REPAIR_SUFFIX = (
    "\n\nYour previous response was not valid against the required schema.\n"
    "Return ONLY a single JSON object of the exact shape "
    '{"summary": string, "findings": [{"severity","category","file","line","title",'
    '"description","impact","recommendation","suggested_fix","evidence","confidence"}]}. '
    "No prose, no markdown fences. Error was: "
)

_JUDGE_REPAIR_SUFFIX = (
    "\n\nYour previous response was not valid against the required schema.\n"
    "Return ONLY a single JSON object of the exact shape "
    '{"summary": string, "findings": [{"severity","category","file","line","title",'
    '"description","impact","recommendation","suggested_fix","evidence","confidence",'
    '"reported_by": [string]}], "rejected": [{"title","reason"}]}. '
    "No prose, no markdown fences. Error was: "
)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    m = _JSON_OBJ_RE.search(text)
    return m.group(0) if m else text


def parse_review(raw: str) -> LLMReview:
    """Raise ``ValidationError`` / ``ValueError`` if ``raw`` is not a valid ``LLMReview``."""
    data = json.loads(_extract_json(raw))
    return LLMReview.model_validate(data)


def parse_judge(raw: str) -> JudgeReview:
    """Raise ``ValidationError`` / ``ValueError`` if ``raw`` is not a valid ``JudgeReview``."""
    data = json.loads(_extract_json(raw))
    return JudgeReview.model_validate(data)


def review_file(provider: LLMProvider, system: str, user: str, target_file: str, *,
                model: Optional[str] = None) -> FileReviewResult:
    raw, err = _try(provider, system, user, parse_review, model=model)
    if isinstance(raw, LLMReview):
        return FileReviewResult(file=target_file, status="ok",
                                summary=raw.summary, findings=list(raw.findings))

    # one repair retry
    repair_user = user + _REPAIR_SUFFIX + str(err)[:400]
    raw2, err2 = _try(provider, system, repair_user, parse_review, model=model)
    if isinstance(raw2, LLMReview):
        return FileReviewResult(file=target_file, status="ok", repaired=True,
                                summary=raw2.summary, findings=list(raw2.findings))

    return FileReviewResult(
        file=target_file, status="failed", repaired=True,
        error=f"invalid LLM response after one repair retry: {err2 or err}",
    )


def judge_review(provider: LLMProvider, system: str, user: str, *,
                 model: Optional[str] = None) -> JudgeRunResult:
    """Judge Agent counterpart of ``review_file`` -- same strict-parse + one-repair-retry
    contract, against ``JudgeReview`` instead of ``LLMReview`` (PRD-multi-agent-p0 s7)."""
    raw, err = _try(provider, system, user, parse_judge, model=model)
    if isinstance(raw, JudgeReview):
        return JudgeRunResult(status="ok", summary=raw.summary,
                              findings=list(raw.findings), rejected=list(raw.rejected))

    repair_user = user + _JUDGE_REPAIR_SUFFIX + str(err)[:400]
    raw2, err2 = _try(provider, system, repair_user, parse_judge, model=model)
    if isinstance(raw2, JudgeReview):
        return JudgeRunResult(status="ok", repaired=True, summary=raw2.summary,
                              findings=list(raw2.findings), rejected=list(raw2.rejected))

    return JudgeRunResult(
        status="failed", repaired=True,
        error=f"invalid Judge response after one repair retry: {err2 or err}",
    )


def _try(provider: LLMProvider, system: str, user: str, parse_fn, *,
        model: Optional[str] = None) -> Tuple[object, object]:
    try:
        raw = provider.complete(system, user, model=model)
    except LLMError as e:
        return None, e
    try:
        return parse_fn(raw), None
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
        return None, e
