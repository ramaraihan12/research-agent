"""What keeps the public demo cheap, in Postgres because serverless instances share no memory."""

from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb

from research_agent.db import Pool

SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_hits (
    key text PRIMARY KEY,
    hits integer NOT NULL,
    expires_at timestamptz NOT NULL
);
CREATE TABLE IF NOT EXISTS demo_runs (
    id text PRIMARY KEY,
    owner text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    lease_until timestamptz,
    slices integer NOT NULL DEFAULT 0,
    search_credits integer NOT NULL DEFAULT 0,
    model_calls integer NOT NULL DEFAULT 0,
    searches jsonb NOT NULL DEFAULT '{}'
);
"""

COUNT_HIT = """
INSERT INTO demo_hits AS h (key, hits, expires_at)
VALUES (%(key)s, 1, now() + make_interval(secs => %(seconds)s))
ON CONFLICT (key) DO UPDATE SET
    hits = CASE WHEN h.expires_at <= now() THEN 1 ELSE h.hits + 1 END,
    expires_at = CASE WHEN h.expires_at <= now() THEN EXCLUDED.expires_at ELSE h.expires_at END
RETURNING hits, GREATEST(1, CEIL(EXTRACT(EPOCH FROM h.expires_at - now())))::int AS retry_after
"""


class RunBudgetExceeded(RuntimeError):
    """A RuntimeError because LangGraph doesn't retry those, and retrying can't help here."""


@dataclass(frozen=True)
class Hit:
    hits: int
    retry_after: int


@dataclass(frozen=True)
class RunRow:
    id: str
    slices: int
    search_credits: int
    model_calls: int


class Guard:
    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(SCHEMA)  # pyright: ignore[reportArgumentType]

    async def count_hit(self, key: str, seconds: float) -> Hit:
        """Count one hit against a window that starts at the first hit and lasts `seconds`."""
        row = await self._one(COUNT_HIT, {"key": key, "seconds": seconds})
        assert row is not None
        return Hit(row["hits"], row["retry_after"])

    async def hits(self, key: str) -> int:
        row = await self._one(
            "SELECT hits FROM demo_hits WHERE key = %s AND expires_at > now()", [key]
        )
        return row["hits"] if row else 0

    async def create_run(self, run_id: str, owner: str) -> None:
        await self._run("INSERT INTO demo_runs (id, owner) VALUES (%s, %s)", [run_id, owner])

    async def get_run(self, run_id: str, owner: str) -> RunRow | None:
        """Only ever by id and owner together, so one visitor can't reach another's run."""
        row = await self._one(
            "SELECT id, slices, search_credits, model_calls FROM demo_runs "
            "WHERE id = %s AND owner = %s",
            [run_id, owner],
        )
        return RunRow(**row) if row else None

    async def take_lease(self, run_id: str, owner: str, seconds: float) -> bool:
        """One request drives a run at a time. A lease runs out, so a killed request can't
        hold a run forever."""
        row = await self._one(
            "UPDATE demo_runs SET lease_until = now() + make_interval(secs => %s) "
            "WHERE id = %s AND owner = %s AND (lease_until IS NULL OR lease_until <= now()) "
            "RETURNING id",
            [seconds, run_id, owner],
        )
        return row is not None

    async def release_lease(self, run_id: str) -> None:
        await self._run("UPDATE demo_runs SET lease_until = NULL WHERE id = %s", [run_id])

    async def count_slice(self, run_id: str) -> None:
        await self._run("UPDATE demo_runs SET slices = slices + 1 WHERE id = %s", [run_id])

    async def spend_search_credit(self, run_id: str, cap: int) -> bool:
        """Spent before the call, so a request that dies halfway still counts it."""
        return await self._spend("search_credits", run_id, cap)

    async def spend_model_call(self, run_id: str, cap: int) -> bool:
        return await self._spend("model_calls", run_id, cap)

    async def _spend(self, column: str, run_id: str, cap: int) -> bool:
        row = await self._one(
            f"UPDATE demo_runs SET {column} = {column} + 1 "  # pyright: ignore[reportArgumentType]
            f"WHERE id = %s AND {column} < %s RETURNING id",
            [run_id, cap],
        )
        return row is not None

    async def cached_search(self, run_id: str, query: str) -> list[dict[str, str]] | None:
        row = await self._one(
            "SELECT searches -> %s AS rows FROM demo_runs WHERE id = %s", [query, run_id]
        )
        return row["rows"] if row else None

    async def save_search(self, run_id: str, query: str, rows: list[dict[str, str]]) -> None:
        await self._run(
            "UPDATE demo_runs SET searches = searches || %s WHERE id = %s",
            [Jsonb({query: rows}), run_id],
        )

    async def expire(self, days: int) -> list[str]:
        """Delete used-up counters and old runs, returning the ids of the runs."""
        await self._run("DELETE FROM demo_hits WHERE expires_at <= now()", [])
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM demo_runs WHERE created_at < now() - make_interval(days => %s) "
                "RETURNING id",
                [days],
            )
            return [row["id"] for row in await cursor.fetchall()]

    async def _one(self, query: Any, params: Any) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(query, params)
            return await cursor.fetchone()

    async def _run(self, query: Any, params: Any) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(query, params)
