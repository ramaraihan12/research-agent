# research-agent

![ci](https://github.com/ramaraihan12/research-agent/actions/workflows/ci.yml/badge.svg)

**Live demo: https://research-agent-mauve-eight.vercel.app** (no login). It runs on Gemini and Tavily with tight caps and 10 runs a day shared by everyone, so it may ask you to come back tomorrow, but there's a finished report on the page either way. After a quiet spell the first request takes a few extra seconds while the function and the database wake up.

Ask it a question, it drafts a short research plan, you approve or edit the plan, then it searches the web, reads a few pages, takes notes and writes a small Markdown report with numbered sources. Built with LangGraph and runs on a local Ollama model by default, so it costs nothing to try.

```text
$ MAX_SUB_QUESTIONS=2 uv run research-agent ask "Which TCP port does PostgreSQL listen on by default, and how do you change it?"
run 9fee079772dc4861b6e4312b658a603e
- drafted a plan with 2 sub-questions

Plan for: Which TCP port does PostgreSQL listen on by default, and how do you change it?
  1. What is the default TCP port for PostgreSQL?
  2. How to change the default PostgreSQL port configuration
[a]pprove, [e]dit or [r]eject? [a]:
left the plan waiting. pick it up with: research-agent resume 9fee079772dc4861b6e4312b658a603e

$ MAX_SUB_QUESTIONS=2 uv run research-agent resume 9fee079772dc4861b6e4312b658a603e --yes --out reports/postgres-port.md
- plan approved (2 sub-questions)
WARNING research_agent.search: search failed for 'How to change the default PostgreSQL port configuration': ConnectError (trimmed)
- searched "How to change the default PostgreSQL port configuration": read 0 pages, kept 0 notes
- searched "What is the default TCP port for PostgreSQL?": read 2 pages, kept 2 notes
- round 1 done, stopping: notes cover the question
- wrote the report
done (notes cover the question)
```

That's a real run on my laptop (RTX 3060, 6 GB) with an IQ4_XS quant of Qwen3.5 9B so it fits in VRAM, about 80 seconds for both commands. The report it wrote is in [examples/postgres-port.md](examples/postgres-port.md). The [examples](examples) folder also has a run through the HTTP API with an edited plan, including the raw SSE stream.

## How it works

```mermaid
flowchart LR
    plan --> review["review_plan (interrupt)"]
    review -- rejected --> stop((end))
    review -- approved --> research["research x N (Send)"]
    research --> reflect
    reflect -- "gaps, under limits" --> research
    reflect -- "enough, or a limit hit" --> write --> done((end))
```

- `plan` asks the model for up to `MAX_SUB_QUESTIONS` search-ready sub-questions.
- `review_plan` calls `interrupt()`, so the run stops and waits. The CLI asks in the terminal, the API returns the plan and waits for a POST. You can approve, replace the list, or reject.
- Each sub-question gets its own `research` branch through `Send`, and they run in parallel: search, fetch pages (10 s timeout, 2 MB cap, main text only via trafilatura), one note-taking call per page.
- `reflect` decides if the notes answer the question. If not, it queues follow-up searches for another round, skipping URLs already read.
- `write` numbers the sources, asks for a cited answer, and drops any `[n]` that points at nothing.
- Every step is checkpointed. With `DATABASE_URL` set that's Postgres, so a paused or crashed run can be picked up later from another process.

Limits are hard caps, not suggestions: rounds, total pages, pages per sub-question, and a token budget that gets split across the parallel branches. Model nodes retry a couple of times because a local Ollama sometimes drops a request while it swaps models.

## Stack

- Python 3.12, uv
- LangGraph (StateGraph, interrupt, Send, Postgres checkpointer)
- LangChain `init_chat_model`: Ollama by default, OpenAI, Gemini or Anthropic through config
- DuckDuckGo via `ddgs` (Tavily if you set a key), httpx + trafilatura
- Typer CLI, FastAPI with server-sent events
- pytest, ruff, pyright, GitHub Actions, Docker
- The public demo: Vercel's Python runtime, Neon Postgres, and one page of plain JavaScript

## Run it

You need [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull qwen3.5:9b
uv sync
cp .env.example .env
uv run research-agent ask "What does HTTP status code 418 mean, and which RFC defines it?"
```

Add `--yes` to skip the plan review and `--out report.md` to save the report. Settings live in `.env`, see [.env.example](.env.example) for all of them. The ones you're most likely to touch:

| Variable | Default | |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen3.5:9b` | `openai`, `google_genai` or `anthropic` after `uv sync --extra <name>`, plus `LLM_API_KEY` |
| `DATABASE_URL` | empty (in memory) | Postgres URL, needed for `resume` |
| `TAVILY_API_KEY` | empty | use Tavily instead of DuckDuckGo (`uv sync --extra tavily`) |
| `MAX_SUB_QUESTIONS`, `MAX_ROUNDS`, `MAX_PAGES`, `PAGES_PER_QUESTION`, `TOKEN_BUDGET` | 3, 2, 10, 2, 60000 | the hard limits |

### HTTP API

```bash
uv run research-agent serve    # http://127.0.0.1:8102, docs at /docs

curl -N -X POST localhost:8102/runs -H 'content-type: application/json' \
  -d '{"question": "What does HTTP status code 418 mean, and which RFC defines it?"}'
# streams progress, ends with a `plan` event that has the thread_id

curl -N -X POST localhost:8102/runs/<thread_id>/plan -H 'content-type: application/json' \
  -d '{"approved": true}'
# or {"approved": true, "plan": ["...", "..."]} to edit, {"approved": false} to reject

curl localhost:8102/runs/<thread_id>    # status, plan, counters, report

curl -N -X POST localhost:8102/runs/<thread_id>/resume
# continues a run left `incomplete` by an error or a dropped connection
```

The research runs inside the request that streams it, so if the client disconnects the run stops where it was and shows up as `incomplete`. Reviewing a run that isn't waiting for a plan, resuming one that isn't incomplete, or touching a run another request is still driving gives a 409. An unknown id gives a 404. Errors that happen mid-stream arrive as an `error` event.

### Docker

```bash
docker compose up --build
```

That starts the API on `127.0.0.1:8102` with its own Postgres. It's only published on localhost because the API has no auth, and anyone who can reach it can start runs on your model key. Ollama stays on the host and the container reaches it through `host.docker.internal`. On Linux, Ollama only listens on 127.0.0.1 by default, which the container can't reach, so start it with `OLLAMA_HOST=0.0.0.0 ollama serve` first.

## The public demo

The API above has no auth, so it isn't what's deployed. `research_agent.demo` is a separate small app around the same graph, `app.py` is its entry point on Vercel, and `public/` is the page. None of the routes above exist there.

- **Time limit.** A Vercel function stops at 300 seconds and a run can take longer. So a request drives the graph for at most `DEMO_SLICE_SECONDS` (240), then ends with a `paused` event. Everything that finished is already checkpointed in Postgres, so the page sends another request and the run carries on, on whichever instance gets it. A run normally takes under a minute and never needs this, but it's what stops a slow model from losing the run. Each step has to fit inside one slice, which is why model calls and searches have their own timeouts.
- **Who sees what.** Run ids are 256 random bits, and a run belongs to the visitor cookie that started it (`HttpOnly`, `SameSite=Strict`, stored hashed). Anyone else gets the same 404 as for a run that doesn't exist, and nothing lists runs.
- **Limits.** Starting a run is limited per address and per day for everyone. The counters live in Postgres because serverless instances share no memory, and an address over its limit can't use up the shared cap. Questions and plans have size caps, and only one request can drive a run at a time, through a lease that runs out if the instance dies.
- **Cost.** With `MAX_SUB_QUESTIONS=2`, `MAX_ROUNDS=1` and `MAX_PAGES=4`, a run costs 2 Tavily credits and 6 Gemini calls: one plan, one note per page, one report. Each one is counted before it happens and a run stops at its budget of 3 credits and 18 calls. A search is paid for once per run, so a retry or a resumed run doesn't buy it again. At 10 runs a day the worst case stays under Tavily's 1,000 free credits a month.
- **A busy Gemini.** The free tier allows 15 calls a minute per model, shared by everything on the key, and it sometimes answers 503 under load. A step that gets refused waits 20 seconds, then 40, which is long enough for the minute to pass, so a run gets slower instead of failing. Retried calls count towards the budget, which is why a run can land above 6. If it still fails, the page offers to continue the run.
- **Fetching.** The fetcher runs on a public host and reads pages it didn't choose, so it only connects to public addresses. The address it checks is the one it connects to, which covers redirects and DNS rebinding as well.

To run your own, create a Vercel project and a Postgres database, set the variables from [.env.example](.env.example), then:

```bash
uv sync --extra gemini --extra tavily
DATABASE_URL=postgresql://... uv run research-agent setup-demo    # once, creates the tables
vercel deploy --prod
```

## Tests

```bash
uv run pytest
```

The graph tests use a scripted fake chat model and a stubbed search and fetcher, so they run offline: routing, the interrupt and resume flow, plan edits and rejection, every limit, retries, the API stream and the CLI prompts. The tests for the public demo and one that checks a paused run survives a restart need real Postgres, because what they prove is that limits and leases hold with no shared memory. They only run when `DATABASE_URL` is set in the environment:

```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/research_agent uv run pytest -m postgres
```

Lint and types: `uv run ruff check . && uv run ruff format --check . && uv run pyright src`

## Notes

- Citations show where a claim came from, not that a 9B model read it right. In [the free-threading example](examples/python-313-free-threading.md) it says you have to build Python 3.13 from source to get the no-GIL build, but the python.org installers already offer it. Check what matters.
- Parallel branches don't know about each other within a round, so two sub-questions can read the same page. In the 418 run both branches ended up on Wikipedia and MDN. Later rounds do skip what was already read.
- The token budget is checked between model calls, so a run can go over by about one call.
- DuckDuckGo through `ddgs` is free but flaky. Rate limits and network errors come back as an empty result for that sub-question (you can see one in the run above) instead of failing the run. Tavily is steadier if you have a key.
- I've run it for real against Ollama and, for the demo, Gemini. The OpenAI and Anthropic paths go through the same `init_chat_model` call but I haven't tested them with live keys.

What I'd add next: share read URLs between parallel branches.
