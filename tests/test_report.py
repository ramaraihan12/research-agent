from research_agent.llm import extract_json, strip_think
from research_agent.report import clean_citations, collect_sources, render_report


def note(url: str, content: str, title: str = "Title") -> dict:
    return {"sub_question": "q", "url": url, "title": title, "content": content}


def test_sources_are_numbered_once_per_url():
    sources = collect_sources(
        [note("https://a.com", "a1"), note("https://b.com", "b1"), note("https://a.com", "a2")]
    )

    assert [(s.number, s.url, s.notes) for s in sources] == [
        (1, "https://a.com", ["a1", "a2"]),
        (2, "https://b.com", ["b1"]),
    ]


def test_clean_citations():
    text = "One [1]. Two [1, 3]. Bad [4]. Link [docs](https://x.com). Year [2024]."

    assert clean_citations(text, source_count=3) == (
        "One [1]. Two [1][3]. Bad . Link [docs](https://x.com). Year [2024]."
    )


def test_render_report_adds_title_and_sources():
    sources = collect_sources([note("https://a.com", "a1", title="Docs [beta]")])

    report = render_report("Why?", "# A title the model added\n\nBecause [1].", sources)

    assert report == ("# Why?\n\nBecause [1].\n\n## Sources\n\n1. [Docs (beta)](https://a.com)\n")


def test_extract_json_finds_object_in_chatter():
    assert extract_json('ok here: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert extract_json("no json here") is None


def test_strip_think():
    assert strip_think("<think>\nhmm\n</think>\n\nAnswer") == "Answer"
