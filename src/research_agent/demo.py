"""The public demo: the same graph as `api`, made safe to leave open on the internet.

`api` has no auth and is meant for localhost. This app is what gets deployed instead, so none
of those routes are reachable from outside. It differs in four ways: a visitor only ever sees
their own runs, starting a run is limited per address and per day, every run has a budget, and
a run moves forward in time slices because a serverless request can't outlive its time limit.
"""

import hashlib
import ipaddress
import logging
import re
import secrets
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, StringConstraints
from starlette.types import ASGIApp, Receive, Scope, Send

from research_agent.config import Settings, get_settings
from research_agent.db import open_pool
from research_agent.guard import Guard

if TYPE_CHECKING:
    from research_agent.demo_run import RunSession

log = logging.getLogger(__name__)

COOKIE = "visitor"
COOKIE_DAYS = 7
TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
DAILY_KEY = "all:day"
MAX_BODY_BYTES = 8 * 1024
MAX_PLAN_ITEMS = 10
# A request can't live past the platform's 300 seconds, so after this long its holder is gone.
LEASE_SECONDS = 300
NOT_FOUND = "no run with that id"

PlanItem = Annotated[str, StringConstraints(min_length=1, max_length=300)]


class StartRun(BaseModel):
    question: str = Field(min_length=3, max_length=300)


class Review(BaseModel):
    approved: bool = True
    plan: list[PlanItem] | None = Field(default=None, max_length=MAX_PLAN_ITEMS)


class BodyLimit:
    """Refuses a large body from its declared length, before anything reads it."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            length = headers.get(b"content-length", b"0")
            if (
                b"transfer-encoding" in headers
                or not length.isdigit()
                or int(length) > MAX_BODY_BYTES
            ):
                response = JSONResponse({"detail": "request is too large"}, status_code=413)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def client_ip(request: Request) -> str:
    """Vercel overwrites these headers, so a client can't forge them there. Behind anything
    else the proxy in front has to do the same."""
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return request.headers.get("x-real-ip") or forwarded or "unknown"


def client_bucket(ip: str) -> str:
    """One IPv6 customer usually holds a whole /64, so that's the unit to count."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return ip[:64]
    return str(ipaddress.ip_network(f"{ip}/64", strict=False)) if address.version == 6 else ip


def owner_key(token: str) -> str:
    """Stored instead of the cookie, so the database alone can't be used to open a run."""
    return hashlib.sha256(token.encode()).hexdigest()


def seconds_to_midnight() -> int:
    now = datetime.now(UTC)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))


def refused(reason: str, message: str, retry_after: int) -> HTTPException:
    return HTTPException(
        429,
        detail={"reason": reason, "message": message, "retry_after": retry_after},
        headers={"Retry-After": str(retry_after)},
    )


def session_tools() -> Any:
    from research_agent import demo_run

    return demo_run


async def charge_run(guard: Guard, settings: Settings, client: str) -> bool:
    """Count one new run against the limits. True if it's the first of the day."""
    midnight = seconds_to_midnight()
    hour = await guard.count_hit(f"ip:{client}:hour", 3600)
    if hour.hits > settings.demo_runs_per_ip_hour:
        raise refused("ip_limit", "You've used your runs for now.", hour.retry_after)
    day = await guard.count_hit(f"ip:{client}:day", midnight)
    if day.hits > settings.demo_runs_per_ip_day:
        raise refused("ip_limit", "You've used your runs for today.", day.retry_after)
    # Counted last, so one client hammering the route can't spend what everyone shares.
    total = await guard.count_hit(DAILY_KEY, midnight)
    if total.hits > settings.demo_runs_per_day:
        raise refused("daily_cap", "The demo has used all of its runs for today.", midnight)
    return total.hits == 1


def app_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_guard(
    settings: Annotated[Settings, Depends(app_settings)],
) -> AsyncIterator[Guard]:
    """A pool per request, closed afterwards: a connection left open would keep Neon awake,
    and a frozen serverless instance can't keep one healthy anyway."""
    assert settings.database_url
    async with open_pool(settings.database_url) as pool:
        yield Guard(pool)


async def throttle(request: Request, guard: Annotated[Guard, Depends(get_guard)]) -> str:
    client = client_bucket(client_ip(request))
    minute = await guard.count_hit(f"ip:{client}:minute", 60)
    if minute.hits > request.app.state.settings.demo_requests_per_ip_minute:
        raise refused("slow_down", "Too many requests, wait a minute.", minute.retry_after)
    return client


def visitor(request: Request) -> str:
    """The visitor's owner key. No cookie means no runs, so that's a 404 like any other."""
    token = request.cookies.get(COOKIE, "")
    if not TOKEN.fullmatch(token):
        raise HTTPException(404, NOT_FOUND)
    return owner_key(token)


@dataclass(frozen=True)
class Claim:
    session: "RunSession"


def claim_run(expected: str) -> Callable[..., AsyncIterator[Claim]]:
    """Dependency that holds a run's lease until the stream ends.

    Doing this in a dependency means a bad id gets a real 404/409 before the stream starts.
    """

    async def dependency(
        run_id: str,
        owner: Annotated[str, Depends(visitor)],
        guard: Annotated[Guard, Depends(get_guard)],
        _: Annotated[str, Depends(throttle)],
        settings: Annotated[Settings, Depends(app_settings)],
    ) -> AsyncIterator[Claim]:
        row = await guard.get_run(run_id, owner)
        if row is None:
            raise HTTPException(404, NOT_FOUND)
        if not await guard.take_lease(run_id, owner, LEASE_SECONDS):
            raise HTTPException(409, "run is already in progress")
        try:
            async with session_tools().open_session(settings, guard, run_id, owner) as session:
                status = await session.status()
                if status is None or status.status != expected:
                    raise HTTPException(409, f"run is not {expected}")
                if row.slices >= settings.demo_max_slices:
                    raise HTTPException(409, "run has used up its time")
                await guard.count_slice(run_id)
                yield Claim(session)
        finally:
            await guard.release_lease(run_id)

    return dependency


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if not settings.database_url:
        raise RuntimeError("the demo keeps its limits in Postgres, so it needs DATABASE_URL")

    app = FastAPI(title="research-agent demo", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.add_middleware(BodyLimit)

    @app.get("/api/limits")
    async def limits(
        response: Response, guard: Annotated[Guard, Depends(get_guard)]
    ) -> dict[str, Any]:
        """Runs left today, for the page. Cached so page views don't reach the database."""
        response.headers["Cache-Control"] = "public, s-maxage=30"
        used = await guard.hits(DAILY_KEY)
        return {
            "runs_left": max(0, settings.demo_runs_per_day - used),
            "runs_per_day": settings.demo_runs_per_day,
            "resets_in": seconds_to_midnight(),
            "caps": {
                "sub_questions": settings.max_sub_questions,
                "rounds": settings.max_rounds,
                "pages": settings.max_pages,
            },
        }

    @app.post("/api/runs")
    async def start_run(
        body: StartRun,
        request: Request,
        response: Response,
        guard: Annotated[Guard, Depends(get_guard)],
        client: Annotated[str, Depends(throttle)],
    ) -> dict[str, Any]:
        """Draft a plan for a new question and pause on it."""
        first_of_the_day = await charge_run(guard, settings, client)
        tools = session_tools()
        if first_of_the_day:
            await tools.forget_old_runs(guard)

        token = request.cookies.get(COOKIE, "")
        if not TOKEN.fullmatch(token):
            token = secrets.token_urlsafe(32)
        run_id = secrets.token_urlsafe(32)
        owner = owner_key(token)
        await guard.create_run(run_id, owner)
        try:
            async with tools.open_session(settings, guard, run_id, owner) as session:
                view = await session.start(body.question)
        except Exception as exc:
            log.exception("couldn't start a run")
            raise HTTPException(503, "The model is busy, try again in a minute.") from exc

        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(
            COOKIE,
            token,
            max_age=COOKIE_DAYS * 86_400,
            path="/api",
            httponly=True,
            secure=True,
            samesite="strict",
        )
        return view

    @app.get("/api/runs/{run_id}")
    async def read_run(
        run_id: str,
        response: Response,
        owner: Annotated[str, Depends(visitor)],
        guard: Annotated[Guard, Depends(get_guard)],
        _: Annotated[str, Depends(throttle)],
    ) -> dict[str, Any]:
        if await guard.get_run(run_id, owner) is None:
            raise HTTPException(404, NOT_FOUND)
        response.headers["Cache-Control"] = "no-store"
        async with session_tools().open_session(settings, guard, run_id, owner) as session:
            return await session.view()

    @app.post("/api/runs/{run_id}/plan", response_class=EventSourceResponse)
    async def review_plan(
        review: Review, claim: Annotated[Claim, Depends(claim_run("awaiting_approval"))]
    ) -> AsyncIterable[ServerSentEvent]:
        """Approve, edit or reject the plan. Ends in `done`, or `paused` if time ran out."""
        async for event in claim.session.review(review.approved, review.plan):
            yield event

    @app.post("/api/runs/{run_id}/resume", response_class=EventSourceResponse)
    async def resume_run(
        claim: Annotated[Claim, Depends(claim_run("incomplete"))],
    ) -> AsyncIterable[ServerSentEvent]:
        """Take the next time slice of a run that stopped partway."""
        async for event in claim.session.resume():
            yield event

    return app
