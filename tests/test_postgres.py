import os
from uuid import uuid4

import psycopg
import pytest
from langgraph.types import Command

from research_agent.config import Limits
from research_agent.runtime import open_checkpointer
from tests.fakes import Harness, ScriptedChatModel

DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.postgres


@pytest.fixture
async def database_url():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = await psycopg.AsyncConnection.connect(DATABASE_URL, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"postgres not reachable: {exc}")
    await conn.close()
    return DATABASE_URL


async def test_paused_run_survives_a_restart(database_url):
    thread = {"configurable": {"thread_id": f"test-{uuid4().hex}"}}
    async with open_checkpointer(database_url) as saver:
        first = Harness(ScriptedChatModel(), Limits(), saver)
        paused = await first.graph.ainvoke({"question": "How do tide pools work?"}, thread)
    assert "__interrupt__" in paused

    # New connection pool and a new graph: nothing is shared in memory.
    async with open_checkpointer(database_url) as saver:
        second = Harness(ScriptedChatModel(), Limits(), saver)
        final = await second.graph.ainvoke(Command(resume={"approved": True}), thread)

    assert final["report"].startswith("# How do tide pools work?")
    assert second.model.count("plan") == 0
    assert len(second.search.queries) == 2
