# PRD — Multi-Agent AI Code Review Agent (P0 Enhancement)

Supersedes the single-LLM P0 described in `README.md`. This is an **enhancement**,
not a rewrite: every existing module (`context/`, `llm/`, `review/`, `publish/`,
`report/`) is reused as-is or with minimal changes. The only new concept is
replacing the one generic per-file LLM call with **four specialist agents run in
parallel + one Judge agent** that validates their combined output.

```
diff -> Java context (per file, unchanged)
      -> [Bug | Security | Performance | Quality] agents in parallel, per file
      -> aggregate candidate findings across all files
      -> Judge Agent (once per PR)
      -> existing validator -> dedup -> risk score + gate
      -> publish (PR comments + commit status) -> review-report.json
```

Stays **GitHub + OpenAI, Java/Spring Boot, CLI-driven** — same environment as the
current P0. No Bitbucket, no Jenkins, no service layer. See §4 for the full
omit list.

---

## 1. Why

The current single-LLM pass makes one model juggle bug/security/performance/test
concerns in a single prompt per file, and nothing double-checks its output beyond
deterministic evidence/line checks. Splitting the analysis into specialist agents
gives deeper per-category reasoning, and a Judge agent adds a second reasoning
pass that rejects false positives, merges duplicate findings across agents, and
corrects severity — closing the gap called out in the reference PRD's Judge
Agent section without adopting its infrastructure (Bitbucket/Jenkins/FastAPI).

## 2. Goals (P0)

- **G1** — Four specialist agents (Bug, Security, Performance, Quality/Test)
  independently review each changed file using the existing per-file
  `ReviewContext`.
- **G2** — Specialist agents for a given file run **in parallel**
  (`asyncio.gather`), not sequentially.
- **G3** — A Judge Agent reviews all candidate findings from all files+agents for
  the PR, and: verifies evidence, rejects unsupported/speculative findings,
  merges duplicates across agents, corrects severity, assigns final confidence.
- **G4** — Existing deterministic validator (file-in-diff, line-in-hunk,
  evidence-verbatim, `MIN_CONFIDENCE`), dedup, and risk scorer/gate keep running
  **after** the Judge, unchanged, as a hard backstop.
- **G5** — If one specialist agent fails on a file, the other agents' findings
  for that file still reach the Judge; the failure is recorded, never silently
  dropped. If the Judge itself fails, the run exits `REVIEW_INCOMPLETE` (exit 2)
  rather than publishing unvalidated findings.
- **G6** — Real LLM only (no mock findings in the `openai` provider path); the
  existing `mock` provider is extended to return per-agent fixtures so tests
  stay offline.

## 3. Non-Goals (P0)

Explicitly out of scope for this iteration — do not build:

- Cross-agent conversation / debate loops (agents don't see each other's output;
  only the Judge does — matches reference PRD §22's "lightweight
  cross-validation, not autonomous conversation").
- Agent-specific LLM providers/models — all five agents share the existing
  `LLMProvider` abstraction and one configured model.
- Any change to how context is built — `context/` is reused unmodified; no
  PR-wide context bundle, no chunking beyond the existing per-file granularity.

## 4. Omitted for P0 (carried over from reference PRD, explicitly deferred)

These appear in the reference PRD but are **not** part of this enhancement.
Flagging them so no one assumes they're implicit:

| Omitted | Reference PRD section | Why deferred |
|---|---|---|
| Bitbucket integration | §29 | Current project targets GitHub; already decided in `README.md` |
| Jenkins / `Jenkinsfile` | §31 | GitHub Actions already fills this role (`.github/workflows/ai-review.yml`) |
| FastAPI REST API / webhook receiver | §33 | CLI-only for P0, same as current project |
| Static analysis (SpotBugs/PMD/Checkstyle) integration | §37 | Marked P1 even in the reference PRD; findings would just be extra context, add later |
| Large-diff chunking beyond per-file | §38 | Existing per-file loop + `MAX_CONTEXT_TOKENS` budget already bounds prompt size |
| Agent execution metrics dashboard / HTML report | §41, §43 | Log durations only (see §11); no dashboard, no `review-report.html` |
| Multi-language support | — | Java/Spring Boot only, same as current project |
| Agent debate / autonomous multi-turn agents | §22 | Explicit non-goal already in reference PRD |
| Per-file Judge calls | — | Judge runs once per PR (design decision, §7) — cheaper and gives a single PR-level summary |

## 5. Agent Interface

New package `src/review_agent/agents/`, mirroring the existing `llm/base.py`
Protocol pattern:

```python
# agents/base.py
class ReviewAgent(Protocol):
    name: str            # "bug-agent" | "security-agent" | ...
    category: str        # "BUG" | "SECURITY" | "PERFORMANCE" | "QUALITY"

    async def review(self, context: ReviewContext) -> AgentResult: ...
```

`context: ReviewContext` is the **existing** per-file object the current single
agent already receives (nothing new to build there). Each concrete agent
(`bug_agent.py`, `security_agent.py`, `performance_agent.py`, `quality_agent.py`)
differs only in: which prompt file it loads, its `category` tag, and the
Pydantic schema it validates against (all four share the existing `Finding`
model — no schema changes needed for specialists).

The Judge is a different shape, since it consumes other agents' output rather
than the raw context:

```python
# agents/judge_agent.py
class JudgeAgent:
    async def review(
        self,
        context: PRSummary,          # new, small — see §8
        candidate_findings: list[AgentResult],
    ) -> JudgeResult: ...
```

## 6. Specialist Agents

All four reuse the current per-file context and LLM contract/repair mechanism
(`llm/contract.py`) unchanged — only the prompt and category differ.

| Agent | Focus | Prompt file |
|---|---|---|
| Bug Agent | Functional correctness: wrong conditions, null risks, bad state transitions, boundary conditions, resource leaks, exception handling | `prompts/bug_agent.txt` |
| Security Agent | SQLi, command/path injection, authZ/authN gaps, hardcoded secrets, unsafe deserialization, input validation, OWASP-style issues | `prompts/security_agent.txt` |
| Performance Agent | N+1 queries, queries-in-loops, blocking calls, inefficient algorithms, excessive allocation | `prompts/performance_agent.txt` |
| Quality/Test Agent | Complexity, duplication, coupling, naming, missing/incorrect tests for changed behavior | `prompts/quality_agent.txt` |

Each returns the existing `AgentResult { agent: str, findings: list[Finding] }`
shape (already close to what the reference PRD's §16 describes; only the
`agent` field name is new — everything else is the current `Finding` model
unchanged).

**Orchestration** (`pipeline.py`), per changed file:

```python
results = await asyncio.gather(
    bug_agent.review(file_context),
    security_agent.review(file_context),
    performance_agent.review(file_context),
    quality_agent.review(file_context),
    return_exceptions=True,   # G5 — one failing agent doesn't kill the others
)
```

Failures are logged per-agent (`Bug Agent: FAILED — <reason>`) and excluded from
that file's candidate findings; every other agent's results for that file still
proceed to the Judge.

## 7. Judge Agent

Runs **once per PR**, not once per file — after all files' specialist agents
have completed. Input is the full list of candidate `Finding`s (tagged with
originating agent) across every changed file, plus a small `PRSummary` (repo,
PR id, base/head, changed file list) for orientation. It does **not** re-receive
full file contents — each `Finding.evidence` already carries the code snippet
needed to judge it, keeping the Judge prompt small and cheap regardless of PR
size.

Responsibilities (from reference PRD §19, unchanged conceptually):

1. Is the finding real / supported by its own evidence?
2. Reject speculative or purely stylistic findings.
3. Merge duplicates reported by multiple agents into one finding with
   `reported_by: [agent names]`.
4. Correct severity where justified (may escalate or downgrade).
5. Assign final `confidence`.
6. Mark each finding `CONFIRMED` or omit it (rejected findings are kept in the
   internal report but never published — reference PRD §20).

Output: `JudgeResult { summary: str, findings: list[Finding], rejected: list[RejectedFinding] }`.

**Design decision — why the Judge doesn't re-verify file/line against the diff:**
that check already exists and is deterministic (`review/validator.py`). The
Judge's LLM judgment (is this real? is it a duplicate? what severity?) and the
existing deterministic validator (is the file/line/evidence actually in the
diff? is confidence above threshold?) are complementary layers — both run,
neither replaces the other. This mirrors the current project's existing
defense-in-depth design instead of introducing a new one.

## 8. Data Model Changes (`models.py`)

Minimal additions — the existing `Finding` model is reused as-is:

```python
class AgentResult(BaseModel):
    agent: str
    category: str
    findings: list[Finding]
    error: str | None = None   # set when the agent call failed (G5)

class Finding(BaseModel):
    ...                        # unchanged fields
    reported_by: list[str] = []   # new, optional — filled in by Judge only

class RejectedFinding(BaseModel):
    title: str
    reason: str

class JudgeResult(BaseModel):
    summary: str
    findings: list[Finding]
    rejected: list[RejectedFinding]
```

## 9. Pipeline Changes (`pipeline.py`)

Current `run()` loop (one LLM call per file) becomes:

1. Build per-file `ReviewContext` for every changed file — **unchanged**
   (`context/` package untouched).
2. For each file, run the 4 specialist agents in parallel
   (`asyncio.gather`, §6). Collect `AgentResult`s.
3. Flatten all files' `AgentResult.findings` into one candidate list, tagging
   each with its source file (already present on `Finding.file`) and
   originating agent.
4. Call `JudgeAgent.review()` once with the full candidate list.
5. Feed `JudgeResult.findings` into the **existing, unmodified**
   `review/validator.py` → `review/dedup.py` → `review/scoring.py` chain.
6. Publish (`publish/github_client.py`, `publish/formatter.py`) and write
   `review-report.json` (`report/json_report.py`) — **unchanged**, except the
   summary/inline comment formatter is extended to print `Detected by: X, Y` /
   `Validated by: Judge Agent` per finding (reference PRD §30), since that data
   now exists on `Finding.reported_by`.

If step 4 (Judge) raises after retries, the run stops and exits with
`REVIEW_INCOMPLETE` / exit code `2` — never publish specialist-only findings
that skipped judging (G5).

## 10. Prompts (`prompts/`)

Replace the single `java_review.txt` with five files, keeping `system.txt`
shared (senior-reviewer framing) and adding a per-agent focus block, same
pattern the reference PRD uses in §12–§15, §21:

```
prompts/
  system.txt            (shared, lightly reworded to be agent-agnostic)
  bug_agent.txt
  security_agent.txt
  performance_agent.txt
  quality_agent.txt
  judge_agent.txt
```

All remain plain, version-controlled text files rendered via the existing
`llm/base.py` templating — no new templating mechanism.

## 11. Config, Cost, and Observability

No new required env vars. `MIN_CONFIDENCE`, `MAX_CONTEXT_TOKENS`,
`MAX_CRITICAL/HIGH/MEDIUM` all keep their current meaning, now applied after
the Judge instead of after the single agent.

**Cost note:** for a PR touching *N* files, LLM calls go from *N* (today) to
*4N + 1* (4 specialists per file, parallelized, + 1 Judge for the whole PR).
Wall-clock latency stays close to today's per-file latency since the 4 calls
per file run concurrently; token/cost spend roughly 4x. Worth stating in the
demo — it's the direct tradeoff for the false-positive reduction and deeper
per-category reasoning.

Extend existing logging (`logging.py`) to show per-agent progress per file,
matching the reference PRD §42 style:

```
INFO [file.java] Bug/Security/Performance/Quality agents started (parallel)
INFO [file.java] Security Agent completed findings=2
INFO [file.java] Bug Agent completed findings=1
INFO [file.java] Performance Agent FAILED: <reason>
INFO Judge Agent started candidates=11
INFO Judge Agent completed confirmed=6 rejected=5
```

No metrics dashboard, no persisted timing table — log lines only (§4).

## 12. Testing

Extend the existing offline, mock-only suite (`tests/`):

- Unit test per specialist agent: given a fixed `ReviewContext` fixture and a
  canned `mock` provider response, asserts the correct prompt file was used and
  the `AgentResult` parses.
- Unit test for `JudgeAgent`: given a fixed list of candidate `Finding`s
  (including an intentional duplicate from two agents), asserts dedup into one
  finding with both names in `reported_by`, and that a low-evidence finding is
  rejected.
- Update the existing pipeline mock/integration test to assert: 4 agents
  invoked in parallel per file, one Judge call for the whole PR, and that a
  single failing specialist agent doesn't stop the run (`return_exceptions`
  path).
- Existing validator/dedup/scoring/contract tests need no changes — they
  operate on `Finding` objects exactly as before, now sourced from the Judge
  instead of the single agent.

## 13. Acceptance Criteria (P0)

- AC1 — Bug, Security, Performance, Quality/Test agents each independently
  analyze the same changed file.
- AC2 — The 4 specialist agents for a file run in parallel, not sequentially.
- AC3 — All agents use the real LLM provider (`openai`); no hardcoded findings.
- AC4 — The Judge Agent validates all candidate findings once per PR: rejects
  unsupported findings, merges duplicates across agents, can change severity.
- AC5 — Final findings carry evidence, confidence, and `reported_by`.
- AC6 — The existing deterministic validator/dedup/scoring/gate still run,
  unchanged, on the Judge's output.
- AC7 — If one specialist agent fails on a file, the review still completes
  using the remaining agents' findings for that file; the failure is visible in
  logs and the report.
- AC8 — If the Judge fails, the run exits `REVIEW_INCOMPLETE` (exit 2) and
  nothing is published to the PR.
- AC9 — Changing the PR code (fixing/introducing an issue) changes which
  findings the relevant specialist agent(s) report, and the Judge's confirmed
  set changes accordingly on a re-run (same dynamic-review guarantee the
  current single-agent P0 already has).
- AC10 — Inline/summary PR comments show which agent(s) detected a finding and
  that it was validated by the Judge.

## 14. Implementation Order

1. `models.py` — add `AgentResult`, `RejectedFinding`, `JudgeResult`,
   `reported_by` field on `Finding`.
2. `agents/base.py` — `ReviewAgent` protocol.
3. `prompts/bug_agent.txt`, `security_agent.txt`, `performance_agent.txt`,
   `quality_agent.txt` (split out of current `java_review.txt`).
4. `agents/bug_agent.py`, `security_agent.py`, `performance_agent.py`,
   `quality_agent.py` — thin, near-identical classes differing by prompt +
   category.
5. `pipeline.py` — per-file `asyncio.gather` over the 4 specialists,
   `return_exceptions=True`, aggregate `AgentResult`s across files.
6. `prompts/judge_agent.txt`, `agents/judge_agent.py`.
7. `pipeline.py` — single Judge call after aggregation, `REVIEW_INCOMPLETE`
   path on Judge failure.
8. `publish/formatter.py` — render `reported_by` / "Validated by: Judge Agent".
9. `llm/mock.py` — extend fixtures to cover 5 agent roles for offline tests.
10. Tests (§12).
11. Update `README.md` to describe the multi-agent flow (replace the current
    "1 call / changed file" line with the new diagram at the top of this doc).

Steps 1–7 are the functional core; 8–11 make it demoable and testable. Nothing
here requires touching `context/`, `llm/base.py`, `llm/openai_provider.py`,
`llm/contract.py`, `review/validator.py`, `review/dedup.py`,
`review/scoring.py`, `git_ops.py`, `config.py`, or `publish/github_client.py`.

## 15. Demo Scenario

Same BookMyShow sample-repo flow already documented in `README.md`
(SQL injection + auth issue + N+1 query PR), re-run through the new pipeline to
show, live:

1. Logs showing 4 agents running in parallel per file.
2. Security Agent and Bug Agent both flagging the same SQL injection from
   different angles → Judge log line showing them merged into one
   `HIGH/SECURITY` finding with `reported_by: [security-agent, bug-agent]`.
3. A deliberately weak/speculative finding from one agent → Judge log line
   showing it rejected.
4. Fix the SQL injection in the PR, re-run → Security/Bug agents no longer
   report it, Judge's confirmed count drops, quality gate flips to PASS —
   identical acceptance behavior to the current single-agent P0, now backed by
   two independent agents agreeing instead of one.
