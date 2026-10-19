from research_agent.runs import progress_event


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
