from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

type Pool = AsyncConnectionPool[AsyncConnection[DictRow]]


@asynccontextmanager
async def open_pool(database_url: str) -> AsyncIterator[Pool]:
    # prepare_threshold=None turns server-side prepared statements off. They don't survive a
    # transaction-mode pooler like Neon's, where each query can land on another connection.
    async with AsyncConnectionPool[AsyncConnection[DictRow]](
        database_url,
        min_size=1,
        max_size=5,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
    ) as pool:
        yield pool
