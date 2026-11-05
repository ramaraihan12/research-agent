"""The half of the public demo that touches the graph.

Kept apart from `demo` because importing LangGraph and the model SDK costs seconds of CPU on
every cold start, and a visitor who only loads the page doesn't need them.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from fastapi.sse import ServerSentEvent
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command, RetryPolicy

from research_agent.config import Settings
from research_agent.db import open_pool
from research_agent.fetch import PageFetcher, make_http_client
from research_agent.graph import Deps, build_graph
from research_agent.guard import Guard, RunBudgetExceeded, RunRow
from research_agent.llm import build_chat_model
from research_agent.runs import Graph, RunInput, RunStatus, get_status, stream_slice, thread_config
from research_agent.search import SearchClient, SearchResult, make_search
from research_agent.state import PlanReview

log = logging.getLogger(__name__)

KEEP_RUNS_DAYS = 7
FAILED = "A service this run needs is busy or down. Wait a minute, then continue the run."
OUT_OF_BUDGET = "This run reached its limit for the demo, so it stopped here."


class ModelMeter(AsyncCallbackHandler):
    """Counts each model call before it's made and stops the run when the budget is gone."""

    raise_error = True

    def __init__(self, guard: Guard, run_id: str, cap: int) -> None:
        self._guard = guard
        self._run_id = run_id
        self._cap = cap

    async def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        if not await self._guard.spend_model_call(self._run_id, self._cap):
            raise RunBudgetExceeded(f"run used all {self._cap} model calls")

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        # LangGraph retries these quietly, and on a shared free key they're worth seeing.
        log.warning("model call failed: %s: %.300s", type(error).__name__, error)


class MeteredSearch:
    """Pays for each query once per run, however many times a retry or a resume repeats it."""

    def __init__(self, inner: SearchClient, guard: Guard, run_id: str, cap: int) -> None:
        self._inner = inner
        self._guard = guard
        self._run_id = run_id
        self._cap = cap

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        rows = await self._guard.cached_search(self._run_id, query)
        if rows is not None:
            return [SearchResult(**row) for row in rows]
        if not await self._guard.spend_search_credit(self._run_id, self._cap):
            log.warning("run used all %d search credits", self._cap)
            return []
        results = await self._inner.search(query, max_results)
        if results:
            await self._guard.save_search(self._run_id, query, [asdict(r) for r in results])
        return results


@dataclass(frozen=True)
class RunSession:
    """One run, opened for the length of one request."""

    run_id: str
    owner: str
    graph: Graph
    guard: Guard
    settings: Settings

    async def start(self, question: str) -> dict[str, Any]:
        await self.graph.ainvoke(
            {"question": question}, thread_config(self.run_id), durability="sync"
        )
        return await self.view()

    async def status(self) -> RunStatus | None:
        return await get_status(self.graph, self.run_id)

    async def view(self) -> dict[str, Any]:
        """What a visitor gets to see of their own run."""
        status = await self.status()
        row = await self.guard.get_run(self.run_id, self.owner)
        assert status is not None and row is not None
        state = status.status
        if state == "incomplete" and self._used_up(row):
            state = "stopped"
        return status.model_dump(exclude={"thread_id"}) | {
            "run_id": self.run_id,
            "status": state,
            "cost": {"tavily_credits": row.search_credits, "model_calls": row.model_calls},
        }

    def _used_up(self, row: RunRow) -> bool:
        limits = self.settings
        return (
            row.slices >= limits.demo_max_slices or row.model_calls >= limits.demo_max_model_calls
        )

    def review(self, approved: bool, plan: list[str] | None) -> AsyncIterator[ServerSentEvent]:
        answer = PlanReview(approved=approved, plan=plan).model_dump()
        return self._drive(Command(resume=answer))

    def resume(self) -> AsyncIterator[ServerSentEvent]:
        return self._drive(None)

    async def _drive(self, payload: RunInput) -> AsyncIterator[ServerSentEvent]:
        seconds = self.settings.demo_slice_seconds
        try:
            async for event in stream_slice(self.graph, payload, self.run_id, seconds):
                if event.type == "done":
                    yield ServerSentEvent(event="done", data=await self.view())
                else:
                    data = event.data | ({"message": event.message} if event.message else {})
                    yield ServerSentEvent(event=event.type, data=data)
        except RunBudgetExceeded:
            yield ServerSentEvent(event="error", data={"message": OUT_OF_BUDGET})
        except Exception:
            # Details stay in the logs. An exception's text can hold URLs and connection info.
            log.exception("run failed")
            yield ServerSentEvent(event="error", data={"message": FAILED})


@asynccontextmanager
async def open_session(
    settings: Settings, guard: Guard, run_id: str, owner: str
) -> AsyncIterator[RunSession]:
    async with make_http_client(settings.fetch_timeout) as http:
        deps = Deps(
            model=build_chat_model(
                settings, callbacks=[ModelMeter(guard, run_id, settings.demo_max_model_calls)]
            ),
            search=MeteredSearch(
                make_search(settings), guard, run_id, settings.demo_max_search_credits
            ),
            fetcher=PageFetcher(
                http,
                timeout=settings.fetch_timeout,
                max_bytes=settings.max_page_bytes,
                max_chars=settings.max_page_chars,
            ),
            limits=settings.limits,
        )
        # The tables already exist: `setup_database` runs once, not on every request.
        # Gemini's free tier counts calls per minute across the whole key, so a retry has to
        # outlast the rest of that minute: 20 seconds, then 40.
        retry = RetryPolicy(max_attempts=3, initial_interval=settings.demo_retry_seconds)
        graph = build_graph(deps, AsyncPostgresSaver(guard.pool), retry)
        yield RunSession(run_id, owner, graph, guard, settings)


async def setup_database(database_url: str) -> None:
    """Create the checkpoint and demo tables. Run once when setting up a deployment."""
    async with open_pool(database_url) as pool:
        await AsyncPostgresSaver(pool).setup()
        await Guard(pool).setup()


async def forget_old_runs(guard: Guard) -> None:
    """Called on the first run of a day, so nothing has to wake the database on a timer."""
    saver = AsyncPostgresSaver(guard.pool)
    for run_id in await guard.expire(KEEP_RUNS_DAYS):
        await saver.adelete_thread(run_id)
