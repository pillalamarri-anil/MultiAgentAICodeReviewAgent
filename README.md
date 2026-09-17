# AI Code Review Agent (P0 — multi-agent)

Reviews real GitHub pull requests on a Java Spring Boot repo with a real LLM, posts
inline + summary findings on the PR, and drives a quality gate (exit 0/1).

Implements **P0** of `coderepo/PRD.md` plus the multi-agent enhancement in
[`PRD-multi-agent-p0.md`](PRD-multi-agent-p0.md). One Python app, config-driven,
GitHub + OpenAI. `mock` provider exists for tests / offline only.

Four specialist agents (Bug, Security, Performance, Quality/Test) review each changed
file in parallel; a Judge agent then validates, dedupes and re-scores their combined
output once per PR before it reaches the existing deterministic validator/gate:

```
diff -> Java context (per file, unchanged)
      -> [Bug | Security | Performance | Quality] agents in parallel, per file
      -> aggregate candidate findings across all files
      -> Judge Agent (once per PR)
      -> validate -> dedup -> risk score + gate
      -> publish (PR comments + commit status) -> review-report.json
```

## Quick start (clean -> CLI result, < 10 min)

```bash
cd implementation
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# offline smoke test -- mock provider, no network
pytest -q

# dry run against the sample repo (writes a report, posts nothing)
LLM_PROVIDER=mock python -m review_agent \
  --repo ../coderepo/BookMyShow \
  --base main --head fix/booking-repository-missing-import \
  --overlay ./overlay --no-publish --report /tmp/review-report.json
```

For a **real** review, copy `.env.example` to `.env` and fill in the OpenAI +
GitHub values, then:

```bash
python -m review_agent --repo ../coderepo/BookMyShow --pr 3 \
  --base main --head my-feature-branch --commit <sha> --overlay ./overlay --publish
```

Exit code: `0` = PASS, `1` = CHANGES REQUESTED (or every file failed LLM review),
`2` = the agent itself errored.

## CLI

| flag | meaning |
|------|---------|
| `--repo` | local checkout to review (default `.`) |
| `--pr` | PR number — enables GitHub diff fetch + publishing |
| `--base` / `--head` | refs for the local `git diff base...head` fallback |
| `--commit` | head SHA for the commit status |
| `--overlay` | dir of `review-rules.yaml` + `docs/` to layer over the repo |
| `--report` | output path (default `review-report.json`, always written) |
| `--publish` / `--no-publish` | post to the PR, or write the report only (default) |

## Configuration (env / `.env`, see `.env.example`)

`LLM_PROVIDER` (`openai` \| `mock`), `OPENAI_API_KEY` / `_MODEL` / `_BASE_URL`,
`GITHUB_TOKEN`, `GITHUB_REPOSITORY`,
`MIN_CONFIDENCE` (0.75), `MAX_CONTEXT_TOKENS` (8000),
`MAX_CRITICAL` / `MAX_HIGH` / `MAX_MEDIUM` (0 / 0 / 5). Secrets come only from the
environment and are never logged.

## Quality gate (PRD §7)

Penalty per finding: CRITICAL 40, HIGH 20, MEDIUM 8, LOW 2.
`score = clamp(100 - Σ, 0, 100)`. The gate fails (exit 1) when any of
`CRITICAL > MAX_CRITICAL`, `HIGH > MAX_HIGH`, `MEDIUM > MAX_MEDIUM`.

## Finding validation (PRD §4.5) — false-positive control

A finding is kept only if its file is in the changed set, its line is inside a changed
hunk, its `evidence` snippet appears verbatim in the model input, and its
`confidence ≥ MIN_CONFIDENCE`. CRITICAL / SECURITY findings are never dropped on
confidence alone. Survivors are de-duplicated on
`file + line-bucket + category + normalized-title`.

This runs **after** the Judge Agent (below) as a deterministic backstop — the two
layers are complementary, neither replaces the other.

## Multi-agent review (`PRD-multi-agent-p0.md`)

- **Specialist agents** (`src/review_agent/agents/{bug,security,performance,quality}_agent.py`)
  each review the same per-file context with a focused prompt
  (`prompts/{bug,security,performance,quality}_agent.txt`) and are run **in parallel**
  per file via a `ThreadPoolExecutor` (the LLM call is blocking network I/O, so threads
  give real concurrency without an async rewrite of the provider layer).
- **Judge Agent** (`agents/judge_agent.py`, `prompts/judge_{system,agent}.txt`) runs
  **once per PR** — not once per file — after every file's specialist agents complete.
  It only sees each candidate finding's `evidence` text, not full file content again, so
  its prompt stays small regardless of PR size. It rejects unsupported/speculative
  findings, merges duplicates reported by multiple agents (`reported_by`), and can
  correct severity/confidence.
- If one specialist agent fails on a file, the others' results still reach the Judge
  (logged, not silently dropped). If the Judge itself fails (bad JSON even after one
  repair retry), the run raises `JudgeFailure` and exits `2` — nothing is published.
  If there are zero candidate findings, the Judge call is skipped entirely.
- Cost: for *N* changed files this is *4N* specialist calls + at most 1 Judge call,
  vs. *N* calls before — parallelized per file, so latency stays close to the
  single-agent baseline while token spend goes up roughly 4x.
- See `agent_results` / `judge_rejected` in `review-report.json`, and the `=== AGENTS
  ===` block the CLI prints, for the full per-agent/Judge breakdown.

## CI (GitHub Actions)

`.github/workflows/ai-review.yml` is written for the **sample repo**, not this one.
Copy it into `BookMyShow/.github/workflows/`, then in that repo set:

- secrets: `OPENAI_API_KEY` (optionally `OPENAI_MODEL`, `OPENAI_BASE_URL`)
- variables: `AI_REVIEW_AGENT_REPO` (this repo's `owner/name`), `AI_REVIEW_AGENT_REF`

It triggers on `pull_request`, runs the agent against the PR diff, posts comments +
commit status, and archives `review-report.json`.

`Jenkinsfile` is intentionally **not** provided — the PRD's Jenkins step is replaced by
GitHub Actions to match the sample repo's existing setup.

## Knowledge overlay

The sample repo does not yet ship review knowledge, so `overlay/` provides
`review-rules.yaml` and `docs/{ARCHITECTURE,schema,persistence}.md`. Commit these into
`BookMyShow` to make them "real", or keep passing `--overlay ./overlay`.

## Layout

```
src/review_agent/
  cli.py  config.py  models.py  pipeline.py  git_ops.py  logging.py
  agents/    SpecialistAgent (bug/security/performance/quality) + JudgeAgent
  context/   diff parse, Java brace-scanner extraction, doc/rule selection, budgeted assembly, render
  llm/       provider protocol, openai_provider, mock, strict-JSON contract + 1 repair retry
  review/    validator, dedup, scoring + gate
  publish/   github_client (sole GitHub API surface), comment formatter
  report/    review-report.json writer
  prompts/   system.txt + bug/security/performance/quality_agent.txt + judge_system/agent.txt
overlay/     review-rules.yaml + docs for BookMyShow
tests/       offline, mock-only; fixtures under tests/fixtures/
```

The `context/` package is ported from the deterministic prototype in
`coderepo/ContextPrep/context_creator.ipynb`.
