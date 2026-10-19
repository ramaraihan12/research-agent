import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from research_agent.api import create_app


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


@pytest.fixture
def harness(make_harness):
    return make_harness()


@pytest.fixture
def client(harness):
    with TestClient(create_app(harness.graph)) as client:
        yield client


def start(client: TestClient, question: str = "How do tide pools work?") -> list[tuple[str, dict]]:
    response = client.post("/runs", json={"question": question})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def test_start_streams_progress_and_pauses_on_the_plan(client):
    events = start(client)

    names = [name for name, _ in events]
    assert names == ["run", "progress", "plan"]
    thread_id = events[0][1]["thread_id"]
    plan = events[-1][1]
    assert plan["thread_id"] == thread_id
    assert plan["plan"] == ["what are tide pools", "which animals live in them"]

    status = client.get(f"/runs/{thread_id}").json()
    assert status["status"] == "awaiting_approval"
    assert status["report"] is None


def test_approving_with_an_edited_plan_streams_until_done(client, harness):
    thread_id = start(client)[0][1]["thread_id"]

    response = client.post(f"/runs/{thread_id}/plan", json={"plan": ["tide pool zones"]})
    events = parse_sse(response.text)

    nodes = [data.get("node") for name, data in events if name == "progress"]
    assert nodes == ["review_plan", "research", "reflect", "write"]
    name, done = events[-1]
    assert name == "done"
    assert done["status"] == "done"
    assert done["plan"] == ["tide pool zones"]
    assert "## Sources" in done["report"]
    assert harness.search.queries == ["tide pool zones"]


def test_rejecting_the_plan(client):
    thread_id = start(client)[0][1]["thread_id"]

    events = parse_sse(client.post(f"/runs/{thread_id}/plan", json={"approved": False}).text)

    assert events[-1][0] == "done"
    assert events[-1][1]["stop_reason"] == "plan rejected"


def test_plan_review_on_unknown_run_is_404(client):
    assert client.post("/runs/nope/plan", json={}).status_code == 404
    assert client.get("/runs/nope").status_code == 404


def test_plan_review_twice_is_409(client):
    thread_id = start(client)[0][1]["thread_id"]
    client.post(f"/runs/{thread_id}/plan", json={})

    response = client.post(f"/runs/{thread_id}/plan", json={})

    assert response.status_code == 409


def test_question_is_validated(client):
    assert client.post("/runs", json={"question": "?"}).status_code == 422


def test_model_failure_becomes_an_error_event(make_harness):
    def broken(_: str) -> str:
        raise ConnectionError("model server is down")

    harness = make_harness(replies={"plan": broken})
    with TestClient(create_app(harness.graph)) as client:
        events = start(client)

    assert events[-1] == ("error", {"message": "ConnectionError: model server is down"})


def test_resume_picks_up_a_run_that_failed_partway(make_harness):
    model_is_down = True

    def notes(user: str) -> str:
        if model_is_down:
            raise ConnectionError("model server is down")
        return "- tide pools hold water at low tide"

    harness = make_harness(replies={"notes": notes})
    with TestClient(create_app(harness.graph)) as client:
        thread_id = start(client)[0][1]["thread_id"]
        events = parse_sse(client.post(f"/runs/{thread_id}/plan", json={}).text)
        assert events[-1][0] == "error"
        assert client.get(f"/runs/{thread_id}").json()["status"] == "incomplete"

        model_is_down = False
        events = parse_sse(client.post(f"/runs/{thread_id}/resume").text)

    assert events[-1][0] == "done"
    assert "## Sources" in events[-1][1]["report"]


def test_resume_only_works_on_incomplete_runs(client):
    thread_id = start(client)[0][1]["thread_id"]

    response = client.post(f"/runs/{thread_id}/resume")

    assert response.status_code == 409
    assert client.post("/runs/nope/resume").status_code == 404


def test_a_run_that_is_still_going_cannot_be_driven_again(make_harness):
    in_research = threading.Event()
    release = threading.Event()

    def slow_notes(user: str) -> str:
        in_research.set()
        release.wait(5)
        return "- tide pools hold water at low tide"

    harness = make_harness(replies={"notes": slow_notes})
    with TestClient(create_app(harness.graph)) as client, ThreadPoolExecutor(1) as pool:
        thread_id = start(client)[0][1]["thread_id"]
        first = pool.submit(client.post, f"/runs/{thread_id}/plan", json={})
        assert in_research.wait(5)

        resumed = client.post(f"/runs/{thread_id}/resume")
        reviewed = client.post(f"/runs/{thread_id}/plan", json={})
        release.set()

        assert parse_sse(first.result().text)[-1][0] == "done"
        assert client.get(f"/runs/{thread_id}").json()["status"] == "done"

    assert resumed.status_code == 409
    assert resumed.json()["detail"] == "run is already in progress"
    assert reviewed.status_code == 409
