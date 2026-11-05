import asyncio
import contextlib
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import urlsplit

import httpcore
import httpx
import trafilatura
from trafilatura.settings import Document

from research_agent.search import SearchResult

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (compatible; research-agent/0.1; +https://github.com/ramaraihan12/research-agent)"
)
MIN_TEXT_CHARS = 200
MAX_REDIRECTS = 5
ALLOWED_PORTS = {None, 80, 443}
INTERNAL_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home.arpa")
METADATA_HOSTS = {"metadata.goog"}
# Azure's wire server has a public address but only answers from inside Azure.
METADATA_ADDRESSES = {ipaddress.ip_address("168.63.129.16")}
NAT64 = ipaddress.ip_network("64:ff9b::/96")

type Resolver = Callable[[str], Awaitable[list[str]]]
type Address = ipaddress.IPv4Address | ipaddress.IPv6Address


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


class PublicOnlyBackend(httpcore.AsyncNetworkBackend):
    """Opens sockets only to addresses it has just checked.

    Checking a name and then letting httpx connect means two lookups, and a DNS server can
    answer them differently. Here the address that was checked is the one connected to. TLS
    still verifies the real hostname, because httpcore takes that from the URL.
    """

    def __init__(
        self, resolve: Resolver = resolve_host, inner: httpcore.AsyncNetworkBackend | None = None
    ) -> None:
        self._resolve = resolve
        self._inner = inner or cast(httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend())

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's signature
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = list(dict.fromkeys(await self._resolve(host)))
        except OSError as exc:
            raise httpcore.ConnectError(f"can't resolve {host}") from exc
        if not addresses or not all(is_public_address(a) for a in addresses):
            raise httpcore.ConnectError(f"{host} is not a public address")

        async def connect(address: str) -> httpcore.AsyncNetworkStream:
            return await self._inner.connect_tcp(
                address,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )

        *others, last = addresses
        for address in others:
            with contextlib.suppress(httpcore.ConnectError, httpcore.ConnectTimeout):
                return await connect(address)
        return await connect(last)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


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
        except (httpx.HTTPError, httpx.InvalidURL, TimeoutError) as exc:
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
        """Refuse early, before anything is sent.

        `PublicOnlyBackend` is what enforces this at connect time, so a name that changes its
        answer after this check still can't get in.
        """
        if not is_public_http_url(url):
            return False
        try:
            addresses = await self._resolve(urlsplit(url).hostname or "")
        except OSError:
            return False
        return bool(addresses) and all(is_public_address(a) for a in addresses)


def extract_main_text(html: str, url: str) -> tuple[str, str]:
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,
        include_comments=False,
        include_tables=True,
        include_images=False,
    )
    if not isinstance(doc, Document):
        return "", ""
    return doc.title or "", doc.text or ""


def is_public_http_url(url: str) -> bool:
    """Reject anything but plain web URLs on public-looking hosts, before resolving them."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return False
    if port not in ALLOWED_PORTS or parts.username or parts.password:
        return False
    host = parts.hostname.rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # A single label is either a local service or an integer spelling of an address.
        return "." in host and not host.endswith(INTERNAL_SUFFIXES) and host not in METADATA_HOSTS
    return is_public_address(host)


def is_public_address(address: str) -> bool:
    """True for addresses on the public internet, looking inside IPv6 forms that wrap IPv4."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return all(is_global(a) for a in [ip, *wrapped_ipv4(ip)])


def is_global(ip: Address) -> bool:
    # is_global alone lets multicast through.
    return (
        ip.is_global and not ip.is_multicast and not ip.is_reserved and ip not in METADATA_ADDRESSES
    )


def wrapped_ipv4(ip: Address) -> list[ipaddress.IPv4Address]:
    if isinstance(ip, ipaddress.IPv4Address):
        return []
    wrapped = [ip.ipv4_mapped, ip.sixtofour, *(ip.teredo or ())]
    if ip in NAT64:
        wrapped.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return [a for a in wrapped if a is not None]


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


def make_http_client(timeout: float, resolve: Resolver = resolve_host) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport()
    # httpx has no option for a network backend, so its pool is replaced with one that has it.
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=httpx.create_ssl_context(),
        network_backend=PublicOnlyBackend(resolve),
    )
    return httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
    )
