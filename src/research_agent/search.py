import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import httpx
from ddgs import DDGS
from ddgs.exceptions import DDGSException

from research_agent.config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchClient(Protocol):
    async def search(self, query: str, max_results: int) -> list[SearchResult]: ...


class DuckDuckGoSearch:
    """Free search through the ddgs metasearch package. No key needed."""

    def __init__(self, timeout: int = 10) -> None:
        self._timeout = timeout

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            rows = await asyncio.to_thread(self._text, query, max_results)
        except DDGSException as exc:
            # ddgs raises when nothing comes back, as well as on rate limits.
            log.warning("search failed for %r: %s", query, exc)
            return []
        return [
            SearchResult(title=row.get("title", ""), url=row["href"], snippet=row.get("body", ""))
            for row in rows
            if row.get("href")
        ]

    def _text(self, query: str, max_results: int) -> list[dict[str, str]]:
        return DDGS(timeout=self._timeout).text(query, max_results=max_results)


class TavilySearch:
    """Optional, used when TAVILY_API_KEY is set (pip extra `tavily`)."""

    def __init__(self, api_key: str) -> None:
        from tavily import AsyncTavilyClient, errors

        self._client = AsyncTavilyClient(api_key=api_key)
        # Bad keys and forbidden requests are left to raise: those need fixing, not skipping.
        self._soft_errors = (
            errors.BadRequestError,
            errors.TimeoutError,
            errors.UsageLimitExceededError,
            httpx.HTTPError,
        )

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            data = await self._client.search(query, max_results=max_results)
        except self._soft_errors as exc:
            log.warning("tavily search failed for %r: %s", query, exc)
            return []
        return [
            SearchResult(
                title=item.get("title", ""), url=item["url"], snippet=item.get("content", "")
            )
            for item in data.get("results", [])
            if item.get("url")
        ]


def make_search(settings: Settings) -> SearchClient:
    if settings.tavily_api_key:
        return TavilySearch(settings.tavily_api_key.get_secret_value())
    return DuckDuckGoSearch()
