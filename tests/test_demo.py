import asyncio
import os
import threading
import time
from collections import Counter

import httpx
import psycopg
import pytest

from research_agent import demo_run
from research_agent.config import Settings
from research_agent.db import open_pool
from research_agent.demo import client_bucket, create_app, owner_key
from research_agent.guard import Guard
from tests.fakes import ScriptedChatModel, StubFetcher, StubSearch, sub_question_of
from tests.test_api import parse_sse

DATABASE_URL = os.environ.get("DATABASE_URL")
QUESTION = "How do tide pools work?"
IP = "203.0.113.7"

pytestmark = pytest.mark.postgres


def demo_settings(**overrides) -> Settings:
    values = {
        "database_url": DATABASE_URL,
        "max_sub_questions": 2,
        "max_rounds": 1,
        "max_pages": 4,
        "pages_per_question": 2,
        "demo_retry_seconds": 0.01,
    }
    return Settings(_env_file=None, **(values | overrides))


class World:
    """The services a run talks to, faked. Everything of ours, meters included, stays real."""

    def __init__(self) -> None:
        self.replies = {}
        self.models: list[ScriptedChatModel] = []
        self.search = StubSearch()
        self.fetcher = StubFetcher()

    def model(self, settings, callbacks) -> ScriptedChatModel:
        self.models.append(ScriptedChatModel(replies=self.replies, callbacks=callbacks))
        return self.models[-1]

    @property
    def real_model_calls(self) -> int:
        return sum(len(model.calls) for model in self.models)


@pytest.fixture
def world(monkeypatch) -> World:
    fake = World()
    monkeypatch.setattr(demo_run, "build_chat_model", fake.model)
    monkeypatch.setattr(demo_run, "make_search", lambda settings: fake.search)
    monkeypatch.setattr(demo_run, "PageFetcher", lambda *args, **kwargs: fake.fetcher)
    return fake


@pytest.fixture
async def database():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    try:
        conn = await psycopg.AsyncConnection.connect(DATABASE_URL, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"postgres not reachable: {exc}")
    await conn.close()
    await demo_run.setup_database(DATABASE_URL)
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True) as conn:
        await conn.execute("TRUNCATE demo_hits, demo_runs")


@pytest.fixture
async def guard(database):
    assert DATABASE_URL
    async with open_pool(DATABASE_URL) as pool:
        yield Guard(pool)


@pytest.fixture
def app(database, world):
    return create_app(demo_settings())


def browser(app, ip: str = IP, cookies=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://testserver",
        headers={"x-real-ip": ip},
        cookies=cookies,
    )


async def start(client: httpx.AsyncClient, question: str = QUESTION) -> dict:
    response = await client.post("/api/runs", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


async def approve(client: httpx.AsyncClient, run_id: str, body: dict | None = None):
    response = await client.post(f"/api/runs/{run_id}/plan", json=body or {})
    assert response.status_code == 200, response.text
    return parse_sse(response.text)


async def resume(client: httpx.AsyncClient, run_id: str):
    response = await client.post(f"/api/runs/{run_id}/resume")
    assert response.status_code == 200, response.text
    return parse_sse(response.text)


async def test_a_visitor_goes_from_question_to_cited_report(app):
    async with browser(app) as me:
        run = await start(me)
        assert run["status"] == "awaiting_approval"
        assert run["plan"] == ["what are tide pools", "which animals live in them"]

        events = await approve(me, run["run_id"])
        name, done = events[-1]

        assert name == "done"
        assert done["status"] == "done"
        assert "## Sources" in done["report"]
        assert [e[0] for e in events].count("progress") >= 3

        saved = (await me.get(f"/api/runs/{run['run_id']}")).json()
        assert saved["status"] == "done"
        assert saved["report"] == done["report"]


async def test_editing_and_rejecting_the_plan_work(app, world):
    async with browser(app) as me:
        edited = await start(me)
        await approve(me, edited["run_id"], {"plan": ["tide pool zones"]})
        assert world.search.queries == ["tide pool zones"]

        rejected = await start(me)
        events = await approve(me, rejected["run_id"], {"approved": False})
        assert events[-1][1]["stop_reason"] == "plan rejected"


async def test_a_run_reports_what_it_really_cost(app, world):
    async with browser(app) as me:
        run = await start(me)
        done = (await approve(me, run["run_id"]))[-1][1]

    assert done["cost"] == {"tavily_credits": 2, "model_calls": world.real_model_calls}
    # One plan, two pages for each of the two sub-questions, one report. With one round the
    # reflect step stops on the round limit before it asks the model.
    assert world.real_model_calls == 6
    assert len(world.search.queries) == 2


# who can see what


async def test_run_ids_are_long_and_random(app):
    async with browser(app) as me:
        first, second = await start(me), await start(me)

    assert first["run_id"] != second["run_id"]
    assert len(first["run_id"]) >= 43  # 32 random bytes in base64url


async def test_the_visitor_cookie_cant_be_read_by_scripts_or_sent_cross_site(app):
    async with browser(app) as me:
        response = await me.post("/api/runs", json={"question": QUESTION})

    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "secure" in cookie
    assert "samesite=strict" in cookie


async def test_a_stranger_with_the_run_id_gets_nothing(app):
    async with browser(app) as me, browser(app, "203.0.113.8") as stranger:
        run = await start(me)
        url = f"/api/runs/{run['run_id']}"

        assert (await stranger.get(url)).status_code == 404
        assert (await stranger.post(f"{url}/plan", json={})).status_code == 404
        assert (await stranger.post(f"{url}/resume")).status_code == 404
        assert (await me.get(url)).json()["status"] == "awaiting_approval"


async def test_another_visitors_cookie_doesnt_open_someone_elses_run(app):
    async with browser(app) as me, browser(app, "203.0.113.8") as other:
        mine = await start(me)
        await start(other)  # now holds a valid cookie of its own
        url = f"/api/runs/{mine['run_id']}"

        assert (await other.get(url)).status_code == 404
        assert (await other.post(f"{url}/plan", json={})).status_code == 404
        assert (await me.get(url)).json()["status"] == "awaiting_approval"


async def test_an_unknown_run_and_someone_elses_run_look_the_same(app):
    async with browser(app) as me, browser(app, "203.0.113.8") as other:
        run = await start(me)
        await start(other)

        theirs = await other.get(f"/api/runs/{run['run_id']}")
        missing = await other.get("/api/runs/does-not-exist")

    assert theirs.status_code == missing.status_code == 404
    assert theirs.json() == missing.json()


async def test_runs_cant_be_listed_and_only_the_demo_routes_exist(app):
    async with browser(app) as me:
        await start(me)
        assert (await me.get("/api/runs")).status_code == 405
        for path in ("/docs", "/redoc", "/openapi.json", "/runs", "/health"):
            assert (await me.get(path)).status_code == 404

    paths = {route.path for route in app.routes}
    assert paths == {
        "/api/limits",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/plan",
        "/api/runs/{run_id}/resume",
    }


async def test_visitor_tokens_are_stored_hashed(app):
    async with browser(app) as me:
        await start(me)
        token = next(iter(me.cookies.jar)).value

    assert DATABASE_URL
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cursor = await conn.execute("SELECT owner FROM demo_runs")
        [(stored,)] = await cursor.fetchall()
    assert stored != token
    assert stored == owner_key(token)


# limits


async def test_one_address_is_limited_per_hour(database, world):
    app = create_app(demo_settings(demo_runs_per_ip_hour=2, demo_runs_per_ip_day=5))
    async with browser(app) as me:
        await start(me)
        await start(me)
        blocked = await me.post("/api/runs", json={"question": QUESTION})

    assert blocked.status_code == 429
    assert blocked.json()["detail"]["reason"] == "ip_limit"
    assert 0 < int(blocked.headers["retry-after"]) <= 3600


async def test_one_address_is_limited_per_day(database, world):
    app = create_app(demo_settings(demo_runs_per_ip_hour=10, demo_runs_per_ip_day=2))
    async with browser(app) as me:
        await start(me)
        await start(me)
        blocked = await me.post("/api/runs", json={"question": QUESTION})

    assert blocked.status_code == 429
    assert blocked.json()["detail"]["reason"] == "ip_limit"


async def test_the_limit_is_per_address(database, world):
    app = create_app(demo_settings(demo_runs_per_ip_hour=1))
    async with browser(app) as first, browser(app, "203.0.113.8") as second:
        await start(first)
        await start(second)
        assert (await first.post("/api/runs", json={"question": QUESTION})).status_code == 429


async def test_the_daily_cap_covers_everyone_and_says_so(database, world):
    app = create_app(demo_settings(demo_runs_per_day=2))
    async with browser(app, "203.0.113.1") as a, browser(app, "203.0.113.2") as b:
        await start(a)
        await start(b)
        async with browser(app, "203.0.113.3") as c:
            blocked = await c.post("/api/runs", json={"question": QUESTION})
            limits = (await c.get("/api/limits")).json()

    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["reason"] == "daily_cap"
    assert "today" in detail["message"]
    assert 0 < detail["retry_after"] <= 86_400
    assert limits["runs_left"] == 0


async def test_an_address_over_its_limit_doesnt_spend_the_shared_cap(database, world):
    app = create_app(demo_settings(demo_runs_per_ip_hour=1, demo_runs_per_day=3))
    async with browser(app) as me:
        await start(me)
        for _ in range(5):
            assert (await me.post("/api/runs", json={"question": QUESTION})).status_code == 429
        assert (await me.get("/api/limits")).json()["runs_left"] == 2


async def test_the_daily_cap_holds_across_instances_under_a_burst(database, world):
    settings = demo_settings(demo_runs_per_day=3)
    instances = [create_app(settings) for _ in range(3)]
    clients = [browser(instances[i % 3], f"203.0.113.{i + 1}") for i in range(12)]

    responses = await asyncio.gather(
        *(c.post("/api/runs", json={"question": QUESTION}) for c in clients)
    )
    for client in clients:
        await client.aclose()

    assert sorted(r.status_code for r in responses) == [200] * 3 + [429] * 9


async def test_too_many_requests_of_any_kind_are_refused(database, world):
    app = create_app(demo_settings(demo_requests_per_ip_minute=3))
    async with browser(app) as me:
        run = await start(me)
        url = f"/api/runs/{run['run_id']}"
        assert (await me.get(url)).status_code == 200
        assert (await me.get(url)).status_code == 200
        blocked = await me.get(url)

    assert blocked.status_code == 429
    assert blocked.json()["detail"]["reason"] == "slow_down"


async def test_a_window_starts_over_once_it_has_passed(guard):
    await guard.count_hit("key", 0.05)
    assert (await guard.count_hit("key", 0.05)).hits == 2

    await asyncio.sleep(0.1)

    assert (await guard.count_hit("key", 0.05)).hits == 1


async def test_counting_hits_is_atomic_across_connection_pools(guard):
    assert DATABASE_URL
    async with open_pool(DATABASE_URL) as other_pool:
        other = Guard(other_pool)
        hits = await asyncio.gather(*(g.count_hit("burst", 60) for g in [guard, other] * 10))

    assert sorted(hit.hits for hit in hits) == list(range(1, 21))


def test_ipv6_is_counted_per_slash_64():
    assert client_bucket("2001:db8:1:2::1") == client_bucket("2001:db8:1:2:ffff::9")
    assert client_bucket("2001:db8:1:2::1") != client_bucket("2001:db8:1:3::1")
    assert client_bucket("203.0.113.7") == "203.0.113.7"


async def test_oversized_input_is_refused(app, world):
    async with browser(app) as me:
        too_long = await me.post("/api/runs", json={"question": "x" * 301})
        too_short = await me.post("/api/runs", json={"question": "x"})
        huge = await me.post("/api/runs", json={"question": "x" * 20_000})
        run = await start(me)
        url = f"/api/runs/{run['run_id']}/plan"
        long_item = await me.post(url, json={"plan": ["x" * 301]})
        many_items = await me.post(url, json={"plan": ["x"] * 11})

    assert too_long.status_code == 422
    assert too_short.status_code == 422
    assert huge.status_code == 413
    assert long_item.status_code == 422
    assert many_items.status_code == 422
    assert world.real_model_calls == 1, "only the valid question reached the model"


async def test_a_body_without_a_declared_length_is_refused(app):
    async with browser(app) as me:

        async def body():
            yield b'{"question": "How do tide pools work?"}'

        response = await me.post(
            "/api/runs", content=body(), headers={"content-type": "application/json"}
        )

    assert response.status_code == 413


async def test_a_run_stops_at_its_model_call_budget(database, world):
    app = create_app(demo_settings(demo_max_model_calls=3))
    async with browser(app) as me:
        run = await start(me)
        name, data = (await approve(me, run["run_id"]))[-1]
        status = (await me.get(f"/api/runs/{run['run_id']}")).json()

    assert name == "error"
    assert "limit" in data["message"]
    assert world.real_model_calls == 3
    assert status["status"] == "stopped"
    assert status["cost"]["model_calls"] == 3


async def test_a_run_stops_paying_for_searches_at_its_credit_budget(database, world):
    app = create_app(demo_settings(demo_max_search_credits=1))
    async with browser(app) as me:
        run = await start(me)
        done = (await approve(me, run["run_id"]))[-1][1]

    assert len(world.search.queries) == 1
    assert done["cost"]["tavily_credits"] == 1


# resuming across requests


def slow_notes(user: str) -> str:
    time.sleep(0.2)
    return "- tide pools hold water at low tide"


async def test_a_run_that_runs_out_of_time_continues_on_another_instance(database, world):
    world.replies["notes"] = slow_notes
    first = create_app(demo_settings(demo_slice_seconds=0.1))
    second = create_app(demo_settings())

    async with browser(first) as me:
        run = await start(me)
        events = await approve(me, run["run_id"])
        assert events[-1][0] == "paused"
        cookies = me.cookies

    async with browser(second, cookies=cookies) as me:
        assert (await me.get(f"/api/runs/{run['run_id']}")).json()["status"] == "incomplete"
        events = await resume(me, run["run_id"])

    assert events[-1][0] == "done"
    assert "## Sources" in events[-1][1]["report"]


async def test_continuing_a_run_doesnt_pay_for_the_same_search_twice(database, world):
    world.replies["notes"] = slow_notes
    app = create_app(demo_settings(demo_slice_seconds=0.1))
    long_slices = create_app(demo_settings())

    async with browser(app) as me:
        run = await start(me)
        assert (await approve(me, run["run_id"]))[-1][0] == "paused"
        async with browser(long_slices, cookies=me.cookies) as same_visitor:
            done = (await resume(same_visitor, run["run_id"]))[-1][1]

    assert sorted(world.search.queries) == ["what are tide pools", "which animals live in them"]
    assert done["cost"]["tavily_credits"] == 2


async def test_a_second_instance_cant_drive_a_run_that_is_still_going(database, world):
    in_research = threading.Event()
    release = threading.Event()

    def blocking_notes(user: str) -> str:
        in_research.set()
        release.wait(5)
        return "- tide pools hold water at low tide"

    world.replies["notes"] = blocking_notes
    first, second = create_app(demo_settings()), create_app(demo_settings())

    async with browser(first) as me:
        run = await start(me)
        url = f"/api/runs/{run['run_id']}"
        going = asyncio.create_task(me.post(f"{url}/plan", json={}))
        assert await asyncio.to_thread(in_research.wait, 5)

        async with browser(second, cookies=me.cookies) as same_visitor:
            resumed = await same_visitor.post(f"{url}/resume")
            reviewed = await same_visitor.post(f"{url}/plan", json={})

        release.set()
        finished = await going

    assert resumed.status_code == 409
    assert reviewed.status_code == 409
    assert parse_sse(finished.text)[-1][0] == "done"


async def test_a_lease_left_by_a_dead_instance_runs_out(app, guard, world):
    async with browser(app) as me:
        run = await start(me)
        owner = owner_key(next(iter(me.cookies.jar)).value)

        assert await guard.take_lease(run["run_id"], owner, 0.05)
        assert (await me.post(f"/api/runs/{run['run_id']}/plan", json={})).status_code == 409

        await asyncio.sleep(0.1)

        assert (await approve(me, run["run_id"]))[-1][0] == "done"


async def test_only_the_owner_can_take_a_lease(guard):
    await guard.create_run("run", "owner")

    assert not await guard.take_lease("run", "someone else", 60)
    assert await guard.take_lease("run", "owner", 60)


async def test_a_finished_request_frees_the_run_for_the_next_one(database, world):
    world.replies["notes"] = slow_notes
    app = create_app(demo_settings(demo_slice_seconds=0.1))
    async with browser(app) as me:
        run = await start(me)
        assert (await approve(me, run["run_id"]))[-1][0] == "paused"
        assert (await me.post(f"/api/runs/{run['run_id']}/resume")).status_code == 200


async def test_approving_twice_is_409(app):
    async with browser(app) as me:
        run = await start(me)
        await approve(me, run["run_id"])
        assert (await me.post(f"/api/runs/{run['run_id']}/plan", json={})).status_code == 409


async def test_resuming_only_works_on_incomplete_runs(app):
    async with browser(app) as me:
        run = await start(me)
        assert (await me.post(f"/api/runs/{run['run_id']}/resume")).status_code == 409


async def test_a_run_cant_be_resumed_forever(database, world):
    world.replies["notes"] = slow_notes
    app = create_app(demo_settings(demo_slice_seconds=0.05, demo_max_slices=2))
    async with browser(app) as me:
        run = await start(me)
        url = f"/api/runs/{run['run_id']}"
        assert (await approve(me, run["run_id"]))[-1][0] == "paused"
        assert (await resume(me, run["run_id"]))[-1][0] == "paused"

        refused = await me.post(f"{url}/resume")
        status = (await me.get(url)).json()

    assert refused.status_code == 409
    assert status["status"] == "stopped"


# failures


def leaky_failure(_: str) -> str:
    raise ValueError("connection to postgresql://user:hunter2@db failed")


async def test_a_failed_start_tells_the_visitor_nothing_internal(database, world):
    world.replies["plan"] = leaky_failure
    app = create_app(demo_settings())
    async with browser(app) as me:
        response = await me.post("/api/runs", json={"question": QUESTION})

    assert response.status_code == 503
    assert "hunter2" not in response.text


async def test_a_failed_step_tells_the_visitor_nothing_internal_and_can_continue(database, world):
    world.replies["notes"] = leaky_failure
    app = create_app(demo_settings())
    async with browser(app) as me:
        run = await start(me)
        response = await me.post(f"/api/runs/{run['run_id']}/plan", json={})
        name = parse_sse(response.text)[-1][0]
        status = (await me.get(f"/api/runs/{run['run_id']}")).json()

        assert name == "error"
        assert "hunter2" not in response.text
        assert status["status"] == "incomplete"

        world.replies.pop("notes")
        assert (await resume(me, run["run_id"]))[-1][0] == "done"


async def test_failed_model_calls_are_counted_and_logged_for_the_operator(database, world, caplog):
    failures = iter([True])

    def flaky(_: str) -> str:
        if next(failures, False):
            raise ConnectionError("503 the model is overloaded")
        return "- tide pools hold water at low tide"

    world.replies["notes"] = flaky
    app = create_app(demo_settings())
    async with browser(app) as me:
        run = await start(me)
        done = (await approve(me, run["run_id"]))[-1][1]

    assert "model call failed: ConnectionError: 503 the model is overloaded" in caplog.text
    assert done["status"] == "done"
    assert done["cost"]["model_calls"] == world.real_model_calls
    assert world.real_model_calls > 6, "the retry costs real calls, so it has to show up"


async def test_retries_wait_out_a_per_minute_quota(guard, world, monkeypatch):
    # Gemini's free tier is 15 calls a minute for everything on the key, so a retry has to
    # outlast the rest of a minute. The defaults are what this checks, not the test settings.
    seen = {}
    build_graph = demo_run.build_graph

    def capture(deps, checkpointer, retry):
        seen["retry"] = retry
        return build_graph(deps, checkpointer, retry)

    monkeypatch.setattr(demo_run, "build_graph", capture)

    await guard.create_run("run", "owner")
    async with demo_run.open_session(Settings(_env_file=None), guard, "run", "owner"):
        pass

    retry = seen["retry"]
    assert retry.max_attempts == 3
    assert retry.initial_interval * (1 + retry.backoff_factor) >= 60


async def test_the_budget_covers_every_step_needing_all_its_attempts(database, world):
    calls: Counter[str] = Counter()

    def fails_twice_then(reply: str):
        def answer(user: str) -> str:
            step = "notes" if "Sub-question: " in user else reply
            calls[step] += 1
            if calls[step] <= 2:
                raise ConnectionError("429 quota")
            return reply

        return answer

    def second_page_fails_twice(user: str) -> str:
        # Each branch reads its first page, then fails on the second, so a retry repeats both.
        branch = sub_question_of(user)
        calls[branch] += 1
        if calls[branch] in (2, 4):
            raise ConnectionError("429 quota")
        return "- tide pools hold water at low tide"

    world.replies.update(
        plan=fails_twice_then('{"sub_questions": ["one", "two"]}'),
        notes=second_page_fails_twice,
        write=fails_twice_then("Tide pools are rocky hollows [1]."),
    )
    app = create_app(demo_settings(demo_max_model_calls=Settings().demo_max_model_calls))
    async with browser(app) as me:
        run = await start(me)
        name, done = (await approve(me, run["run_id"]))[-1]

    assert name == "done", done
    assert "## Sources" in done["report"]
    assert done["cost"]["model_calls"] == world.real_model_calls == 18


async def test_the_quota_check_doesnt_need_the_graph(app):
    async with browser(app) as me:
        response = await me.get("/api/limits")

    assert response.status_code == 200
    limits = response.json()
    assert limits["runs_left"] == limits["runs_per_day"] == 10
    assert limits["caps"] == {"sub_questions": 2, "rounds": 1, "pages": 4}
    assert "s-maxage" in response.headers["cache-control"]


async def test_old_runs_are_forgotten_with_their_checkpoints(app, guard):
    assert DATABASE_URL
    async with browser(app) as me:
        old = await start(me)
        async with await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True) as conn:
            await conn.execute(
                "UPDATE demo_runs SET created_at = now() - interval '8 days' WHERE id = %s",
                [old["run_id"]],
            )
        fresh = await start(me)

        await demo_run.forget_old_runs(guard)

        assert (await me.get(f"/api/runs/{old['run_id']}")).status_code == 404
        assert (await me.get(f"/api/runs/{fresh['run_id']}")).status_code == 200

    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cursor = await conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", [old["run_id"]]
        )
        assert await cursor.fetchone() == (0,)
