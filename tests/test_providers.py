from typing import Any

import pytest

from research_agent import llm
from research_agent.config import Settings
from research_agent.search import TavilySearch


def capture_model_kwargs(monkeypatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_init(model: str, model_provider: str, **kwargs: Any) -> str:
        seen.update(kwargs, model=model, provider=model_provider)
        return "model"

    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    return seen


def test_hosted_models_get_the_configured_timeout_and_retries(monkeypatch):
    seen = capture_model_kwargs(monkeypatch)
    settings = Settings(
        llm_provider="google_genai",
        llm_model="gemini-3.5-flash-lite",
        llm_timeout=30,
        llm_max_retries=1,
        _env_file=None,
    )

    llm.build_chat_model(settings)

    assert seen["timeout"] == 30
    assert seen["max_retries"] == 1


def test_ollama_keeps_its_defaults(monkeypatch):
    seen = capture_model_kwargs(monkeypatch)

    llm.build_chat_model(Settings(_env_file=None))

    assert seen["provider"] == "ollama"
    assert seen["model"] == "qwen3.5:9b"
    assert seen["base_url"] == "http://localhost:11434"
    assert "timeout" not in seen
    assert "max_retries" not in seen
    assert "callbacks" not in seen


def test_callbacks_reach_the_model(monkeypatch):
    seen = capture_model_kwargs(monkeypatch)
    handler = object()

    llm.build_chat_model(Settings(_env_file=None), callbacks=[handler])

    assert seen["callbacks"] == [handler]


@pytest.fixture
def tavily_calls(monkeypatch) -> list[dict[str, Any]]:
    tavily = pytest.importorskip("tavily")
    calls: list[dict[str, Any]] = []

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            pass

        async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"results": [{"title": "t", "url": "https://example.com", "content": "c"}]}

    monkeypatch.setattr(tavily, "AsyncTavilyClient", FakeClient)
    return calls


async def test_tavily_searches_stay_at_one_credit_and_have_a_timeout(tavily_calls):
    search = TavilySearch("key", timeout=15)

    [result] = await search.search("tide pools", 5)

    assert result.url == "https://example.com"
    assert tavily_calls == [{"max_results": 5, "search_depth": "basic", "timeout": 15}]
