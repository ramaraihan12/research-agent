from operator import add
from typing import Annotated, NotRequired, TypedDict

from pydantic import BaseModel, Field


class Note(TypedDict):
    sub_question: str
    url: str
    title: str
    content: str


class ResearchState(TypedDict):
    question: str
    plan: NotRequired[list[str]]
    approved: NotRequired[bool]
    # Sub-questions queued for the next research round.
    pending: NotRequired[list[str]]
    searched: NotRequired[Annotated[list[str], add]]
    notes: NotRequired[Annotated[list[Note], add]]
    read_urls: NotRequired[Annotated[list[str], add]]
    tokens_used: NotRequired[Annotated[int, add]]
    rounds: NotRequired[int]
    stop_reason: NotRequired[str]
    report: NotRequired[str]


class ResearchTask(TypedDict):
    """Payload for one parallel research branch, sent with `Send`."""

    sub_question: str
    max_pages: int
    max_tokens: int
    skip_urls: list[str]


class PlanReview(BaseModel):
    """What a human sends back when the run pauses on the plan."""

    approved: bool = True
    plan: list[str] | None = Field(
        default=None, description="Edited sub-questions. Leave out to keep the drafted plan."
    )
