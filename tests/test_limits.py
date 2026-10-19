import itertools
import json

import pytest
from langgraph.types import Command
from pydantic import ValidationError

from research_agent.config import Settings

QUESTION = "How do tide pools work?"


def never_enough():
    """Reflection that always wants one more, brand new search."""
    counter = itertools.count(1)
    return lambda _: json.dumps({"enough": False, "follow_up": [f"follow up {next(counter)}"]})


async def run(h, thread, plan=None):
    await h.graph.ainvoke({"question": QUESTION}, thread)
    resume = {"approved": True} if plan is None else {"plan": plan}
    return await h.graph.ainvoke(Command(resume=resume), thread)


async def test_round_limit_ends_the_reflect_loop(make_harness, thread):
    h = make_harness(replies={"reflect": never_enough()}, max_rounds=3, max_pages=40)

    final = await run(h, thread)

    assert final["rounds"] == 3
    assert final["stop_reason"] == "hit the round limit (3)"
    # The last round hits the limit before asking the model again.
    assert h.model.count("reflect") == 2
    assert final["report"]


async def test_page_limit_caps_pages_across_rounds(make_harness, thread):
    h = make_harness(
        replies={"reflect": never_enough()},
        max_rounds=5,
        max_pages=5,
        pages_per_question=2,
    )

    final = await run(h, thread)

    assert len(h.fetcher.fetched) == 5
    assert len(final["read_urls"]) == 5
    assert final["stop_reason"] == "hit the page limit (5)"


async def test_pages_per_question_caps_each_branch(make_harness, thread):
    h = make_harness(max_pages=40, pages_per_question=1)

    final = await run(h, thread, plan=["one", "two", "three"])

    assert len(final["read_urls"]) == 3
    assert sorted(h.search.queries) == ["one", "three", "two"]


async def test_token_budget_stops_research(make_harness, thread):
    # Every call "costs" 3000 tokens against a 5000 budget: the plan uses 3000,
    # each branch gets 1000 and stops after its first page.
    h = make_harness(
        replies={"reflect": never_enough()},
        tokens_per_call=3_000,
        token_budget=5_000,
        max_rounds=4,
        max_pages=40,
        pages_per_question=3,
    )

    final = await run(h, thread)

    assert h.model.count("notes") == 2
    assert h.model.count("reflect") == 0
    assert final["stop_reason"] == "hit the token budget (5000)"
    assert final["report"], "a report is still written from what was gathered"


async def test_plan_is_capped_to_max_sub_questions(make_harness, thread):
    many = json.dumps({"sub_questions": [f"question {i}" for i in range(8)]})
    h = make_harness(replies={"plan": many}, max_sub_questions=3)

    result = await h.graph.ainvoke({"question": QUESTION}, thread)
    assert len(result["__interrupt__"][0].value["plan"]) == 3

    final = await h.graph.ainvoke(
        Command(resume={"plan": [f"edited {i}" for i in range(6)]}), thread
    )
    assert final["plan"] == ["edited 0", "edited 1", "edited 2"]


def test_settings_reject_a_page_budget_smaller_than_the_plan():
    with pytest.raises(ValidationError):
        Settings(max_sub_questions=5, max_pages=3, _env_file=None)
