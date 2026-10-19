import asyncio
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable, Sequence
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
MAX_REDIRECTS = 5

type Resolver = Callable[[str], Awaitable[list[str]]]


@dataclass(frozen=True)
class Page:
    url: str
    title: str
    text: str


class Fetcher(Protocol):
    async def fetch(self, result: SearchResult) -> Page | None: ...


async def resolve_host(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


class PageFetcher:
    """Downloads a page with a hard time and size limit and keeps only the main text."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: float,
        max_bytes: int,
        max_chars: int,
        resolve: Resolver = resolve_host,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._resolve = resolve

    async def fetch(self, result: SearchResult) -> Page | None:
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
        # Redirects are followed by hand so every hop gets the same address check.
        for _ in range(MAX_REDIRECTS + 1):
            if not await self._is_public(url):
                log.info("not fetching %s: not a public address", url)
                return None, "", None
            async with self._client.stream("GET", url, follow_redirects=False) as response:
                if response.next_request is not None:
                    url = str(response.next_request.url)
                    continue
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
        log.info("not fetching %s: more than %d redirects", url, MAX_REDIRECTS)
        return None, "", None

    async def _is_public(self, url: str) -> bool:
        """Also resolve the host, so a public-looking name that points inside is rejected.

        httpx looks the name up again when it connects, so this doesn't stop DNS rebinding.
        """
        if not is_public_http_url(url):
            return False
        try:
            addresses = await self._resolve(urlsplit(url).hostname or "")
        except OSError:
            return False
        return bool(addresses) and all(ipaddress.ip_address(a).is_global for a in addresses)


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
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
    )
