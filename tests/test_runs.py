import time

from langgraph.types import Command

from research_agent.runs import get_status, progress_event, stream_slice

QUESTION = "How do tide pools work?"
THREAD = "slice-thread"


def test_progress_messages_handle_singular_and_plural():
    one = progress_event(
        "research", {"searched": ["tides"], "read_urls": ["https://a.com"], "notes": [{}]}
    )
    none = progress_event("research", {"searched": ["tides"], "read_urls": [], "notes": []})

    assert one.message == 'searched "tides": read 1 page, kept 1 note'
    assert none.message == 'searched "tides": read 0 pages, kept 0 notes'
    assert progress_event("plan", {"plan": ["a"]}).message == "drafted a plan with 1 sub-question"
    assert (
        progress_event("review_plan", {"approved": True, "plan": ["a", "b"]}).message
        == "plan approved (2 sub-questions)"
    )


def slow_notes(user: str) -> str:
    time.sleep(0.2)
    return "- tide pools hold water at low tide"


async def collect(graph, payload, seconds):
    return [event async for event in stream_slice(graph, payload, THREAD, seconds)]


async def test_a_slice_with_time_left_behaves_like_a_normal_run(make_harness):
    graph = make_harness().graph

    events = await collect(graph, {"question": QUESTION}, 30)
    assert [event.type for event in events] == ["progress", "plan"]

    events = await collect(graph, Command(resume={"approved": True}), 30)
    assert events[-1].type == "done"
    assert "## Sources" in events[-1].data["report"]


async def test_a_slice_that_runs_out_of_time_pauses_and_the_next_one_finishes(make_harness):
    h = make_harness(replies={"notes": slow_notes})
    await collect(h.graph, {"question": QUESTION}, 30)

    events = await collect(h.graph, Command(resume={"approved": True}), 0.1)

    assert events[-1].type == "paused"
    status = await get_status(h.graph, THREAD)
    assert status is not None
    assert status.status == "incomplete"
    assert status.report is None

    events = await collect(h.graph, None, 30)

    assert events[-1].type == "done"
    assert "## Sources" in events[-1].data["report"]


async def test_work_finished_before_the_pause_is_not_done_again(make_harness):
    # One sub-question finishes quickly, the other is still reading when time runs out.
    def notes(user: str) -> str:
        if "slow one" in user:
            time.sleep(0.5)
        return "- tide pools hold water at low tide"

    h = make_harness(replies={"notes": notes}, pages_per_question=1)
    await collect(h.graph, {"question": QUESTION}, 30)

    plan = {"approved": True, "plan": ["quick one", "slow one"]}
    events = await collect(h.graph, Command(resume=plan), 0.25)
    assert events[-1].type == "paused"

    await collect(h.graph, None, 30)

    assert h.search.queries.count("quick one") == 1
    assert h.search.queries.count("slow one") == 2


async def test_an_error_in_a_slice_is_not_mistaken_for_running_out_of_time(make_harness):
    def broken(_: str) -> str:
        raise TimeoutError("model timed out")

    h = make_harness(replies={"plan": broken})

    try:
        await collect(h.graph, {"question": QUESTION}, 30)
    except TimeoutError as exc:
        assert str(exc) == "model timed out"
    else:
        raise AssertionError("the node's own timeout should reach the caller")
