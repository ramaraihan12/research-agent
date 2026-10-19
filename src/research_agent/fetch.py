import asyncio
import ipaddress
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

import httpx
import trafilatura

from research_agent.search import SearchResult

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; research-agent/0.1; +https://github.com/ramaraihan12/research-agent)"
)
MIN_TEXT_CHARS = 200


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str


class Fetcher(Protocol):
    async def fetch(self, result: SearchResult) -> Page | None: ...


class PageFetcher:
    """Downloads a page with a hard time and size limit and keeps only the main text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: float,
        max_bytes: int,
        max_chars: int,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_chars = max_chars

    async def fetch(self, result: SearchResult) -> Page | None:
        if not is_public_http_url(result.url):
            return None
        try:
            async with asyncio.timeout(self._timeout):
                raw, content_type, encoding = await self._download(result.url)
        except (httpx.HTTPError, TimeoutError) as exc:
            log.info("skipping %s: %s", result.url, exc or type(exc).__name__)
            return None
        if raw is None:
            return None

        body = raw.decode(encoding or "utf-8", errors="replace")
        if "html" in content_type:
            title, text = await asyncio.to_thread(extract_main_text, body, result.url)
        else:
            title, text = "", body.strip()
        if len(text) < MIN_TEXT_CHARS:
            return None
        return Page(
            url=result.url, title=result.title or title or result.url, text=text[: self._max_chars]
        )

    async def _download(self, url: str) -> tuple[bytes | None, str, str | None]:
        async with self._client.stream("GET", url) as response:
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code >= 400 or not (
                "html" in content_type or content_type.startswith("text/plain")
            ):
                return None, content_type, None
            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) >= self._max_bytes:
                    break
            return bytes(body[: self._max_bytes]), content_type, response.encoding


def extract_main_text(html: str, url: str) -> tuple[str, str]:
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,
        include_comments=False,
        include_tables=True,
        include_images=False,
    )
    if doc is None:
        return "", ""
    return doc.title or "", doc.text or ""


def is_public_http_url(url: str) -> bool:
    """Reject non-http schemes and obvious internal hosts before fetching."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    host = parts.hostname
    if host == "localhost" or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return address.is_global


async def read_pages(fetcher: Fetcher, results: Sequence[SearchResult], limit: int) -> list[Page]:
    """Fetch results in rank order until `limit` pages worked or the results run out."""
    pages: list[Page] = []
    queue = list(results)
    while queue and len(pages) < limit:
        need = limit - len(pages)
        batch, queue = queue[:need], queue[need:]
        fetched = await asyncio.gather(*(fetcher.fetch(result) for result in batch))
        pages.extend(page for page in fetched if page is not None)
    return pages


def make_http_client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        max_redirects=5,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
    )
