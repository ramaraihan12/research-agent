import asyncio
import sys
from collections.abc import Callable
from typing import Any

import pytest

from research_agent.config import Limits
from tests.fakes import Harness, Reply, ScriptedChatModel

if sys.platform == "win32":
    # psycopg's async driver can't run on the default Proactor loop on Windows.
    def pytest_asyncio_loop_factories(config, item):
        return {"selector": asyncio.SelectorEventLoop}


@pytest.fixture
def make_harness() -> Callable[..., Harness]:
    def make(
        replies: dict[str, Reply] | None = None,
        tokens_per_call: int = 100,
        **limits: int,
    ) -> Harness:
        model = ScriptedChatModel(replies=replies or {}, tokens_per_call=tokens_per_call)
        return Harness(model, Limits(**limits))

    return make


@pytest.fixture
def thread() -> dict[str, Any]:
    return {"configurable": {"thread_id": "test-thread"}}
