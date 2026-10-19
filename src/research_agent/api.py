import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.types import Command
from pydantic import BaseModel, Field

from research_agent.config import get_settings
from research_agent.runs import Graph, RunState, RunStatus, get_status, stream_run
from research_agent.runtime import open_graph
from research_agent.state import PlanReview

log = logging.getLogger(__name__)


class StartRun(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(graph: Graph | None = None) -> FastAPI:
    """Build the API. Tests pass a ready graph; otherwise one is built from settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if graph is not None:
            app.state.graph = graph
            yield
            return
        async with open_graph(get_settings()) as built:
            app.state.graph = built
            yield

    app = FastAPI(title="research-agent", lifespan=lifespan)
    # Thread ids with a request driving them right now. The research runs inside that
    # request, so a second one on the same thread would run the graph twice in parallel.
    app.state.active = set()

    @app.post("/runs", response_class=EventSourceResponse)
    async def start_run(
        body: StartRun, request: Request, graph: Annotated[Graph, Depends(get_graph)]
    ) -> AsyncIterable[ServerSentEvent]:
        """Draft a plan for a new question. The stream ends with a `plan` event."""
        thread_id = uuid4().hex
        active: set[str] = request.app.state.active
        active.add(thread_id)
        try:
            yield ServerSentEvent(event="run", data={"thread_id": thread_id})
            async for event in sse_events(graph, {"question": body.question}, thread_id):
                yield event
        finally:
            active.discard(thread_id)

    @app.post("/runs/{thread_id}/plan", response_class=EventSourceResponse)
    async def review_plan(
        review: PlanReview,
        thread_id: Annotated[str, Depends(claim_run("awaiting_approval"))],
        graph: Annotated[Graph, Depends(get_graph)],
    ) -> AsyncIterable[ServerSentEvent]:
        """Approve, edit or reject the plan. Streams research progress, then `done`."""
        async for event in sse_events(graph, Command(resume=review.model_dump()), thread_id):
            yield event

    @app.post("/runs/{thread_id}/resume", response_class=EventSourceResponse)
    async def resume_run(
        thread_id: Annotated[str, Depends(claim_run("incomplete"))],
        graph: Annotated[Graph, Depends(get_graph)],
    ) -> AsyncIterable[ServerSentEvent]:
        """Continue a run that stopped partway, after an error or a dropped connection."""
        async for event in sse_events(graph, None, thread_id):
            yield event

    @app.get("/runs/{thread_id}")
    async def get_run(thread_id: str, graph: Annotated[Graph, Depends(get_graph)]) -> RunStatus:
        status = await get_status(graph, thread_id)
        if status is None:
            raise HTTPException(404, "no run with that id")
        return status

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def get_graph(request: Request) -> Graph:
    return request.app.state.graph


def claim_run(expected: RunState) -> Callable[..., AsyncIterator[str]]:
    """Dependency that checks the run's state and holds it until the stream ends.

    Doing this in a dependency means a bad id gets a real 404/409 before the stream starts.
    """

    async def dependency(
        thread_id: str, request: Request, graph: Annotated[Graph, Depends(get_graph)]
    ) -> AsyncIterator[str]:
        active: set[str] = request.app.state.active
        # Claimed before the first await, so two requests can't both get past this.
        if thread_id in active:
            raise HTTPException(409, "run is already in progress")
        active.add(thread_id)
        try:
            status = await get_status(graph, thread_id)
            if status is None:
                raise HTTPException(404, "no run with that id")
            if status.status != expected:
                raise HTTPException(409, f"run is {status.status}, expected {expected}")
            yield thread_id
        finally:
            active.discard(thread_id)

    return dependency


async def sse_events(
    graph: Graph, payload: dict[str, Any] | Command | None, thread_id: str
) -> AsyncIterator[ServerSentEvent]:
    try:
        async for event in stream_run(graph, payload, thread_id):
            data = event.data | ({"message": event.message} if event.message else {})
            yield ServerSentEvent(event=event.type, data=data)
    except Exception as exc:
        # Headers are already sent, so the only way to report this is in the stream.
        log.exception("run %s failed", thread_id)
        yield ServerSentEvent(event="error", data={"message": f"{type(exc).__name__}: {exc}"})
