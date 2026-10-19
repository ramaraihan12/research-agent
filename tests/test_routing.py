import pytest

from research_agent.config import Limits
from research_agent.graph import (
    dispatch,
    limit_reached,
    parse_plan,
    parse_reflection,
    split_evenly,
)


@pytest.mark.parametrize(
    ("total", "parts", "expected"),
    [(10, 3, [4, 3, 3]), (2, 3, [1, 1, 0]), (0, 2, [0, 0]), (-4, 2, [0, 0]), (5, 0, [])],
)
def test_split_evenly(total, parts, expected):
    assert split_evenly(total, parts) == expected


def test_dispatch_sends_one_branch_per_pending_question():
    limits = Limits(max_pages=10, pages_per_question=2, token_budget=1_000)
    state = {
        "question": "q",
        "pending": ["a", "b"],
        "read_urls": ["https://x.com/1", "https://x.com/1", "https://x.com/2"],
        "tokens_used": 400,
    }

    sends = dispatch(state, limits)

    assert [s.node for s in sends] == ["research", "research"]
    assert [s.arg["sub_question"] for s in sends] == ["a", "b"]
    assert [s.arg["max_pages"] for s in sends] == [2, 2]
    assert [s.arg["max_tokens"] for s in sends] == [300, 300]
    assert sends[0].arg["skip_urls"] == ["https://x.com/1", "https://x.com/2"]


def test_dispatch_drops_branches_with_no_budget_left():
    limits = Limits(max_pages=4, pages_per_question=2)
    state = {"question": "q", "pending": ["a", "b", "c"], "read_urls": ["u1", "u2", "u3"]}

    sends = dispatch(state, limits)

    assert [s.arg["sub_question"] for s in sends] == ["a"]
    assert sends[0].arg["max_pages"] == 1


def test_dispatch_with_nothing_pending_is_empty():
    assert dispatch({"question": "q", "pending": []}, Limits()) == []


@pytest.mark.parametrize(
    ("state", "rounds", "expected"),
    [
        ({"read_urls": [], "tokens_used": 0}, 1, None),
        ({"read_urls": [], "tokens_used": 0}, 2, "hit the round limit (2)"),
        ({"read_urls": ["u"] * 10, "tokens_used": 0}, 1, "hit the page limit (10)"),
        ({"read_urls": [], "tokens_used": 60_000}, 1, "hit the token budget (60000)"),
    ],
)
def test_limit_reached(state, rounds, expected):
    limits = Limits(max_rounds=2, max_pages=10, token_budget=60_000)
    assert limit_reached({"question": "q", **state}, rounds, limits) == expected


@pytest.mark.parametrize(
    "reply",
    [
        '{"sub_questions": ["What is A?", "What is B?"]}',
        'Sure!\n```json\n{"sub_questions": ["What is A?", "What is B?"]}\n```',
        '["What is A?", "What is B?"]',
        "Here is the plan:\n1. What is A?\n2) What is B?\n",
        "- What is A?\n- What is B?\n- what is a?",
    ],
)
def test_parse_plan_accepts_common_reply_shapes(reply):
    assert parse_plan(reply, limit=5) == ["What is A?", "What is B?"]


def test_parse_plan_returns_empty_for_prose():
    assert parse_plan("I think you should search the web.", limit=5) == []


def test_parse_reflection():
    verdict = parse_reflection('{"enough": "false", "follow_up": ["x"]}')
    assert verdict.enough is False
    assert verdict.follow_up == ["x"]


@pytest.mark.parametrize("reply", ["not json", '{"enough": false, "follow_up": "x"}', "[]"])
def test_unreadable_reflection_means_stop(reply):
    assert parse_reflection(reply).enough is True
