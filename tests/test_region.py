"""The region gate, tested without Postgres: a stub guard stands in for the database."""

import httpx
import pytest

from research_agent import demo
from research_agent.config import Settings
from research_agent.demo import create_app, get_guard
from research_agent.guard import Hit

COUNTRY = "x-vercel-ip-country"
ROUTES = ["/api/runs", "/api/runs/some-id/plan", "/api/runs/some-id/resume"]


class BusyGuard:
    """Reports every limit as used up, so a request that gets past the region check ends in a
    429 before anything else runs."""

    async def count_hit(self, key: str, seconds: float) -> Hit:
        return Hit(hits=10**6, retry_after=5)


@pytest.fixture
def tools_touched(monkeypatch) -> list[str]:
    """Records any attempt to start a run, which is where the model gets called."""
    touched: list[str] = []

    def session_tools():
        touched.append("session")
        raise AssertionError("a blocked visitor reached the model")

    monkeypatch.setattr(demo, "session_tools", session_tools)
    return touched


@pytest.fixture
def app():
    settings = Settings(_env_file=None, database_url="postgresql://never-connected")
    app = create_app(settings)

    async def guard():
        yield BusyGuard()

    app.dependency_overrides[get_guard] = guard
    return app


def client(app, country: str | None = None) -> httpx.AsyncClient:
    headers = {COUNTRY: country} if country else {}
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="https://testserver", headers=headers
    )


@pytest.mark.parametrize("country", ["DE", "FR", "GB", "CH", "NO", "AX", "re"])
@pytest.mark.parametrize("path", ROUTES)
async def test_a_blocked_country_gets_451_and_no_model_call(app, tools_touched, path, country):
    async with client(app, country) as http:
        response = await http.post(path, json={"question": "How do tide pools work?"})

    assert response.status_code == 451
    assert "EEA, Switzerland or the UK" in response.json()["detail"]
    assert tools_touched == []


@pytest.mark.parametrize("country", ["ID", "US", None])
async def test_an_allowed_country_or_no_header_gets_past_the_gate(app, country):
    # No header means local dev or a test, so it must be let through.
    async with client(app, country) as http:
        response = await http.post("/api/runs", json={"question": "How do tide pools work?"})

    assert response.status_code == 429


async def test_the_page_can_ask_whether_it_is_blocked(app):
    async with client(app, "SE") as http:
        blocked = await http.get("/api/region")
    async with client(app, "ID") as http:
        allowed = await http.get("/api/region")
    async with client(app) as http:
        unknown = await http.get("/api/region")

    assert blocked.json() == {"blocked": True}
    assert allowed.json() == {"blocked": False}
    assert unknown.json() == {"blocked": False}
    assert blocked.headers["cache-control"] == "private, no-store"
