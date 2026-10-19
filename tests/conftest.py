import asyncio
import json
import re
import sys
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from research_agent import prompts
from research_agent.config import Limits
from research_agent.fetch import Page
from research_agent.graph import Deps, build_graph
from research_agent.search import SearchResult

if sys.platform == "win32":
    # psycopg's async driver can't run on the default Proactor loop on Windows.
    def pytest_asyncio_loop_factories(config, item):
        return {"selector": asyncio.SelectorEventLoop}


Reply = str | Callable[[str], str]


class ScriptedChatModel(BaseChatModel):
    """Fake chat model that answers by prompt type instead of call order.

    Research branches run in parallel, so a fixed list of replies would race.
    """

    replies: dict[str, Reply] = Field(default_factory=dict)
    tokens_per_call: int = 100
    calls: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **_: Any
    ) -> ChatResult:
        system, user = str(messages[0].content), str(messages[-1].content)
        kind = prompt_kind(system)
        self.calls.append((kind, user))
        reply = self.replies.get(kind, DEFAULT_REPLIES[kind])
        text = reply(user) if callable(reply) else reply
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": self.tokens_per_call - 1,
                "output_tokens": 1,
                "total_tokens": self.tokens_per_call,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.calls if k == kind)


def prompt_kind(system: str) -> str:
    for kind, template in {
        "plan": prompts.PLAN_SYSTEM,
        "notes": prompts.NOTES_SYSTEM,
        "reflect": prompts.REFLECT_SYSTEM,
        "write": prompts.WRITE_SYSTEM,
    }.items():
        if system[:30] == template[:30]:
            return kind
    raise AssertionError(f"unexpected system prompt: {system[:60]!r}")


def sub_question_of(user: str) -> str:
    match = re.search(r"Sub-question: (.+)", user)
    return match.group(1) if match else "?"


DEFAULT_REPLIES: dict[str, Reply] = {
    "plan": json.dumps({"sub_questions": ["what are tide pools", "which animals live in them"]}),
    "notes": lambda user: f"- a useful fact about {sub_question_of(user)}",
    "reflect": json.dumps({"enough": True, "missing": "", "follow_up": []}),
    "write": "Tide pools are rocky hollows [1]. Many animals live there [2]. Made up [42].",
}


class StubSearch:
    def __init__(self, results_per_query: int = 5) -> None:
        self.queries: list[str] = []
        self.results_per_query = results_per_query

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        slug = re.sub(r"\W+", "-", query.lower()).strip("-")
        count = min(max_results, self.results_per_query)
        return [
            SearchResult(title=f"{query} ({i})", url=f"https://example.com/{slug}/{i}", snippet="")
            for i in range(count)
        ]


class StubFetcher:
    def __init__(self) -> None:
        self.fetched: list[str] = []

    async def fetch(self, result: SearchResult) -> Page | None:
        self.fetched.append(result.url)
        return Page(url=result.url, title=result.title, text=f"Some text from {result.url}")


class Harness:
    def __init__(self, model: ScriptedChatModel, limits: Limits, checkpointer=None) -> None:
        self.model = model
        self.search = StubSearch()
        self.fetcher = StubFetcher()
        self.deps = Deps(model=model, search=self.search, fetcher=self.fetcher, limits=limits)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_graph(self.deps, self.checkpointer)


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
