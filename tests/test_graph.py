import json

from langgraph.types import Command

from research_agent.graph import build_graph

QUESTION = "How do tide pools work?"


def reflect_with(*follow_ups: str, enough: bool = False) -> str:
    return json.dumps({"enough": enough, "missing": "more", "follow_up": list(follow_ups)})


async def test_pauses_with_the_drafted_plan(make_harness, thread):
    h = make_harness()

    result = await h.graph.ainvoke({"question": QUESTION}, thread)

    [pause] = result["__interrupt__"]
    assert pause.value == {
        "question": QUESTION,
        "plan": ["what are tide pools", "which animals live in them"],
    }
    assert h.search.queries == []


async def test_approving_runs_research_and_writes_cited_report(make_harness, thread):
    h = make_harness()
    await h.graph.ainvoke({"question": QUESTION}, thread)

    final = await h.graph.ainvoke(Command(resume={"approved": True}), thread)

    assert sorted(h.search.queries) == ["what are tide pools", "which animals live in them"]
    report = final["report"]
    assert report.startswith(f"# {QUESTION}\n")
    assert "rocky hollows [1]" in report
    assert "[42]" not in report, "citations to unknown sources are dropped"
    assert "## Sources" in report
    assert "(https://example.com/what-are-tide-pools/0)" in report
    assert final["stop_reason"] == "notes cover the question"


async def test_edited_plan_replaces_the_draft(make_harness, thread):
    h = make_harness()
    await h.graph.ainvoke({"question": QUESTION}, thread)

    final = await h.graph.ainvoke(
        Command(
            resume={"plan": ["tide pool zones", "  tide pool zones ", "", "tide pool animals"]}
        ),
        thread,
    )

    assert final["plan"] == ["tide pool zones", "tide pool animals"]
    assert sorted(h.search.queries) == ["tide pool animals", "tide pool zones"]


async def test_rejecting_the_plan_stops_without_searching(make_harness, thread):
    h = make_harness()
    await h.graph.ainvoke({"question": QUESTION}, thread)

    final = await h.graph.ainvoke(Command(resume={"approved": False}), thread)

    assert final["stop_reason"] == "plan rejected"
    assert "report" not in final
    assert h.search.queries == []


async def test_follow_up_round_searches_only_new_questions(make_harness, thread):
    verdicts = iter(
        [reflect_with("tide pool zones", "what are tide pools"), reflect_with(enough=True)]
    )
    h = make_harness(replies={"reflect": lambda _: next(verdicts)}, max_rounds=3)
    await h.graph.ainvoke({"question": QUESTION}, thread)

    final = await h.graph.ainvoke(Command(resume={"approved": True}), thread)

    assert h.search.queries[-1] == "tide pool zones"
    assert h.search.queries.count("what are tide pools") == 1
    assert final["rounds"] == 2
    assert final["stop_reason"] == "notes cover the question"


async def test_a_paused_run_resumes_from_the_checkpoint(make_harness, thread):
    first = make_harness()
    await first.graph.ainvoke({"question": QUESTION}, thread)

    # A fresh graph on the same checkpointer stands in for a restarted process.
    second = make_harness()
    graph = build_graph(second.deps, first.checkpointer)
    final = await graph.ainvoke(Command(resume={"approved": True}), thread)

    assert final["report"].startswith(f"# {QUESTION}")
    assert second.model.count("plan") == 0
    assert first.search.queries == []


async def test_plan_falls_back_to_the_question_when_reply_is_garbage(make_harness, thread):
    h = make_harness(replies={"plan": "sorry, I can't do JSON today"})

    result = await h.graph.ainvoke({"question": QUESTION}, thread)

    assert result["__interrupt__"][0].value["plan"] == [QUESTION]


async def test_no_usable_pages_still_produces_a_report(make_harness, thread):
    h = make_harness(replies={"notes": "NOTHING"})
    await h.graph.ainvoke({"question": QUESTION}, thread)

    final = await h.graph.ainvoke(Command(resume={"approved": True}), thread)

    assert "couldn't find any usable pages" in final["report"]
    assert "## Sources" not in final["report"]
    assert h.model.count("write") == 0


async def test_a_dropped_model_call_is_retried(make_harness, thread):
    attempts = []

    def flaky(user: str) -> str:
        attempts.append(user)
        if len(attempts) == 1:
            raise ConnectionError("model runner restarted")
        return json.dumps({"sub_questions": ["what are tide pools"]})

    h = make_harness(replies={"plan": flaky})

    result = await h.graph.ainvoke({"question": QUESTION}, thread)

    assert len(attempts) == 2
    assert result["__interrupt__"][0].value["plan"] == ["what are tide pools"]
