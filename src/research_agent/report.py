import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from research_agent.state import Note

_CITATION = re.compile(r"(\s*)\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_LEADING_TITLE = re.compile(r"\A\s*#\s[^\n]*\n+")


@dataclass
class Source:
    number: int
    url: str
    title: str
    notes: list[str] = field(default_factory=list)


def collect_sources(notes: Iterable[Note]) -> list[Source]:
    """Number each distinct URL once, in the order it was first noted."""
    by_url: dict[str, Source] = {}
    for note in notes:
        source = by_url.get(note["url"])
        if source is None:
            source = Source(number=len(by_url) + 1, url=note["url"], title=note["title"])
            by_url[note["url"]] = source
        source.notes.append(note["content"])
    return list(by_url.values())


def format_sources_for_prompt(sources: Iterable[Source]) -> str:
    blocks = []
    for source in sources:
        notes = "\n".join(source.notes)
        blocks.append(f"[{source.number}] {source.title}\nURL: {source.url}\n{notes}")
    return "\n\n".join(blocks)


def clean_citations(text: str, source_count: int) -> str:
    """Drop citation numbers that point at no source and split "[1, 2]" into "[1][2]"."""

    def fix(match: re.Match[str]) -> str:
        space, group = match.groups()
        valid = [n for n in map(int, group.split(",")) if 1 <= n <= source_count]
        return space + "".join(f"[{n}]" for n in valid) if valid else ""

    return _CITATION.sub(fix, text)


def render_report(question: str, body: str, sources: list[Source]) -> str:
    body = _LEADING_TITLE.sub("", body.strip(), count=1)
    body = clean_citations(body, len(sources)).strip()
    lines = [f"# {question.strip()}", "", body]
    if sources:
        lines += ["", "## Sources", ""]
        lines += [f"{s.number}. [{_link_text(s.title)}]({s.url})" for s in sources]
    return "\n".join(lines) + "\n"


def _link_text(title: str) -> str:
    return title.replace("[", "(").replace("]", ")").strip() or "untitled"
