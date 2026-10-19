import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, ValidationError

from research_agent import prompts
from research_agent.config import Limits
from research_agent.fetch import Fetcher, read_pages
from research_agent.llm import complete, extract_json
from research_agent.report import collect_sources, format_sources_for_prompt, render_report
from research_agent.search import SearchClient
from research_agent.state import Note, PlanReview, ResearchState, ResearchTask

_LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$")
MAX_QUESTION_CHARS = 300


@dataclass(frozen=True)
class Deps:
    model: BaseChatModel
    search: SearchClient
    fetcher: Fetcher
    limits: Limits


class Reflection(BaseModel):
    enough: bool = True
    follow_up: list[str] = []


def build_graph(
    deps: Deps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph[ResearchState]:
    limits = deps.limits

    async def plan(state: ResearchState) -> dict[str, Any]:
        reply = await complete(
            deps.model,
            prompts.PLAN_SYSTEM.format(max_items=limits.max_sub_questions),
            state["question"],
        )
        sub_questions = parse_plan(reply.text, limits.max_sub_questions)
        return {"plan": sub_questions or [state["question"]], "tokens_used": reply.tokens}

    def review_plan(state: ResearchState) -> dict[str, Any]:
        answer = interrupt({"question": state["question"], "plan": state["plan"]})
        review = PlanReview.model_validate(answer)
        if not review.approved:
            return {"approved": False, "stop_reason": "plan rejected"}
        edited = review.plan if review.plan is not None else state["plan"]
        sub_questions = clean_questions(edited, limits.max_sub_questions)
        if not sub_questions:
            return {"approved": False, "stop_reason": "plan was empty"}
        return {"approved": True, "plan": sub_questions, "pending": sub_questions}

    async def research(task: ResearchTask) -> dict[str, Any]:
        question = task["sub_question"]
        skip = set(task["skip_urls"])
        results = await deps.search.search(question, limits.results_per_query)
        pages = await read_pages(
            deps.fetcher, [r for r in results if r.url not in skip], task["max_pages"]
        )

        notes: list[Note] = []
        read: list[str] = []
        spent = 0
        for page in pages:
            if spent >= task["max_tokens"]:
                break
            reply = await complete(
                deps.model,
                prompts.NOTES_SYSTEM,
                prompts.NOTES_USER.format(
                    sub_question=question, title=page.title, url=page.url, text=page.text
                ),
            )
            spent += reply.tokens
            read.append(page.url)
            if is_useful_note(reply.text):
                notes.append(
                    Note(sub_question=question, url=page.url, title=page.title, content=reply.text)
                )
        return {"searched": [question], "notes": notes, "read_urls": read, "tokens_used": spent}

    async def reflect(state: ResearchState) -> dict[str, Any]:
        rounds = state.get("rounds", 0) + 1
        update: dict[str, Any] = {"rounds": rounds, "pending": []}
        if reason := limit_reached(state, rounds, limits):
            return update | {"stop_reason": reason}

        reply = await complete(
            deps.model,
            prompts.REFLECT_SYSTEM.format(max_items=limits.max_sub_questions),
            prompts.REFLECT_USER.format(
                question=state["question"],
                searched="\n".join(f"- {q}" for q in state.get("searched", [])),
                notes=format_notes(state.get("notes", [])),
            ),
        )
        update["tokens_used"] = reply.tokens
        verdict = parse_reflection(reply.text)
        if verdict.enough:
            return update | {"stop_reason": "notes cover the question"}

        done = {q.casefold() for q in state.get("searched", [])}
        pages_left = limits.max_pages - len(state.get("read_urls", []))
        follow_ups = [
            q
            for q in clean_questions(verdict.follow_up, limits.max_sub_questions)
            if q.casefold() not in done
        ][:pages_left]
        if not follow_ups:
            return update | {"stop_reason": "no new follow-up searches"}
        return update | {"pending": follow_ups}

    async def write(state: ResearchState) -> dict[str, Any]:
        question = state["question"]
        sources = collect_sources(state.get("notes", []))
        if not sources:
            body = (
                "I couldn't find any usable pages for this question, so there is nothing to cite."
            )
            return {"report": render_report(question, body, [])}
        reply = await complete(
            deps.model,
            prompts.WRITE_SYSTEM,
            prompts.WRITE_USER.format(
                question=question, sources=format_sources_for_prompt(sources)
            ),
        )
        return {"report": render_report(question, reply.text, sources), "tokens_used": reply.tokens}

    def after_review(state: ResearchState) -> list[Send] | Literal["write", "__end__"]:
        if not state.get("approved"):
            return END
        return dispatch(state, limits) or "write"

    def after_reflect(state: ResearchState) -> list[Send] | Literal["write"]:
        return dispatch(state, limits) or "write"

    builder = StateGraph(ResearchState)
    builder.add_node("plan", plan)
    builder.add_node("review_plan", review_plan)
    builder.add_node("research", research)
    builder.add_node("reflect", reflect)
    builder.add_node("write", write)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "review_plan")
    builder.add_conditional_edges("review_plan", after_review, ["research", "write", END])
    builder.add_edge("research", "reflect")
    builder.add_conditional_edges("reflect", after_reflect, ["research", "write"])
    builder.add_edge("write", END)
    return builder.compile(checkpointer=checkpointer)


def dispatch(state: ResearchState, limits: Limits) -> list[Send]:
    """One `Send` per pending sub-question, splitting what's left of the page and token budget."""
    pending = state.get("pending", [])
    read_urls = state.get("read_urls", [])
    pages = [
        min(share, limits.pages_per_question)
        for share in split_evenly(limits.max_pages - len(read_urls), len(pending))
    ]
    tokens = split_evenly(limits.token_budget - state.get("tokens_used", 0), len(pending))
    skip = sorted(set(read_urls))
    return [
        Send(
            "research",
            ResearchTask(sub_question=q, max_pages=p, max_tokens=t, skip_urls=skip),
        )
        for q, p, t in zip(pending, pages, tokens, strict=True)
        if p > 0 and t > 0
    ]


def split_evenly(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    base, extra = divmod(max(total, 0), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def limit_reached(state: ResearchState, rounds: int, limits: Limits) -> str | None:
    if rounds >= limits.max_rounds:
        return f"hit the round limit ({limits.max_rounds})"
    if len(state.get("read_urls", [])) >= limits.max_pages:
        return f"hit the page limit ({limits.max_pages})"
    if state.get("tokens_used", 0) >= limits.token_budget:
        return f"hit the token budget ({limits.token_budget})"
    return None


def parse_plan(text: str, limit: int) -> list[str]:
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("sub_questions")
    if isinstance(data, list):
        return clean_questions(data, limit)
    # Small models sometimes ignore the JSON instruction and answer with a plain list.
    items = [m.group(1) for line in text.splitlines() if (m := _LIST_ITEM.match(line))]
    return clean_questions(items, limit)


def parse_reflection(text: str) -> Reflection:
    data = extract_json(text)
    if not isinstance(data, dict):
        return Reflection(enough=True)
    try:
        return Reflection.model_validate(data)
    except ValidationError:
        return Reflection(enough=True)


def clean_questions(items: Iterable[object], limit: int) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        question = " ".join(item.split())[:MAX_QUESTION_CHARS]
        if question and question.casefold() not in seen:
            seen.add(question.casefold())
            cleaned.append(question)
    return cleaned[:limit]


def is_useful_note(text: str) -> bool:
    stripped = text.strip()
    return len(stripped) > 20 and not stripped.upper().startswith("NOTHING")


def format_notes(notes: Iterable[Note]) -> str:
    lines: list[str] = []
    current = None
    for note in sorted(notes, key=lambda n: n["sub_question"]):
        if note["sub_question"] != current:
            current = note["sub_question"]
            lines.append(f"\n## {current}")
        lines.append(f"From {note['title']}:\n{note['content']}")
    return "\n".join(lines).strip() or "(no notes yet)"
