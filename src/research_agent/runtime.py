import asyncio
import sys
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph

from research_agent.config import Settings
from research_agent.db import open_pool
from research_agent.fetch import PageFetcher, make_http_client
from research_agent.graph import Deps, build_graph
from research_agent.llm import build_chat_model
from research_agent.search import make_search
from research_agent.state import ResearchState


def run_async[T](coro: Coroutine[Any, Any, T]) -> T:
    # psycopg's async driver needs a selector loop; Windows defaults to Proactor.
    if sys.platform == "win32":
        return asyncio.run(coro, loop_factory=asyncio.SelectorEventLoop)
    return asyncio.run(coro)


@asynccontextmanager
async def open_checkpointer(database_url: str | None) -> AsyncIterator[BaseCheckpointSaver]:
    """Postgres when DATABASE_URL is set, otherwise memory (runs die with the process)."""
    if not database_url:
        yield InMemorySaver()
        return
    async with open_pool(database_url) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver


@asynccontextmanager
async def open_graph(settings: Settings) -> AsyncIterator[CompiledStateGraph[ResearchState]]:
    async with (
        make_http_client(settings.fetch_timeout) as http,
        open_checkpointer(settings.database_url) as checkpointer,
    ):
        fetcher = PageFetcher(
            http,
            timeout=settings.fetch_timeout,
            max_bytes=settings.max_page_bytes,
            max_chars=settings.max_page_chars,
        )
        deps = Deps(
            model=build_chat_model(settings),
            search=make_search(settings),
            fetcher=fetcher,
            limits=settings.limits,
        )
        yield build_graph(deps, checkpointer)
