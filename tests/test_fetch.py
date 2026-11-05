import asyncio
import ipaddress

import httpcore
import httpx
import pytest

from research_agent.fetch import (
    MAX_REDIRECTS,
    Page,
    PageFetcher,
    PublicOnlyBackend,
    is_public_address,
    is_public_http_url,
    make_http_client,
    read_pages,
    resolve_host,
)
from research_agent.search import SearchResult

ARTICLE = """
<html><head><title>Tide pools</title></head>
<body>
  <nav>Home | About | Shop</nav>
  <article>
    <h1>How tide pools work</h1>
    <p>{body}</p>
  </article>
  <footer>Copyright footer text</footer>
</body></html>
""".format(
    body="Tide pools form when the sea goes out and water stays behind in rock hollows. " * 8
)


RAW_RESPONSE = b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\ncontent-length: 2\r\n\r\nhi"


async def public_dns(host: str) -> list[str]:
    return ["93.184.215.14"]


def make_fetcher(handler, **overrides) -> PageFetcher:
    # follow_redirects=True on purpose: the fetcher has to check each hop itself anyway.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    options = {"timeout": 2.0, "max_bytes": 100_000, "max_chars": 5_000, "resolve": public_dns}
    return PageFetcher(client, **(options | overrides))


def result(url: str, title: str = "") -> SearchResult:
    return SearchResult(title=title, url=url, snippet="")


async def test_extracts_main_text_from_html():
    fetcher = make_fetcher(
        lambda request: httpx.Response(200, html=ARTICLE),
    )

    page = await fetcher.fetch(result("https://example.com/tides", "Tide pools explained"))

    assert page is not None
    assert page.title == "Tide pools explained"
    assert "rock hollows" in page.text
    assert "Home | About" not in page.text


async def test_truncates_text_to_max_chars():
    fetcher = make_fetcher(lambda request: httpx.Response(200, html=ARTICLE), max_chars=300)

    page = await fetcher.fetch(result("https://example.com/tides"))

    assert page is not None
    assert len(page.text) == 300


async def test_stops_reading_at_max_bytes():
    big = "x" * 50_000
    fetcher = make_fetcher(
        lambda request: httpx.Response(200, text=big, headers={"content-type": "text/plain"}),
        max_bytes=10_000,
        max_chars=100_000,
    )

    page = await fetcher.fetch(result("https://example.com/big.txt"))

    assert page is not None
    assert len(page.text) == 10_000


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, html=ARTICLE),
        httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}),
        httpx.Response(200, html="<html><body><p>too short</p></body></html>"),
    ],
    ids=["not-found", "pdf", "too-short"],
)
async def test_skips_unusable_responses(response):
    fetcher = make_fetcher(lambda request: response)

    assert await fetcher.fetch(result("https://example.com/x")) is None


async def test_gives_up_on_slow_pages():
    async def slow(request):
        await asyncio.sleep(5)
        return httpx.Response(200, html=ARTICLE)

    fetcher = make_fetcher(slow, timeout=0.05)

    assert await fetcher.fetch(result("https://example.com/slow")) is None


async def test_never_requests_internal_addresses():
    def handler(request):
        raise AssertionError(f"should not fetch {request.url}")

    fetcher = make_fetcher(handler)

    for url in ["http://localhost:8000/", "http://127.0.0.1/", "http://10.0.0.5/", "file:///etc"]:
        assert await fetcher.fetch(result(url)) is None


async def test_follows_redirects_between_public_hosts():
    def handler(request):
        if request.url.host == "old.example.com":
            return httpx.Response(301, headers={"location": "https://new.example.com/tides"})
        return httpx.Response(200, html=ARTICLE)

    fetcher = make_fetcher(handler)

    page = await fetcher.fetch(result("https://old.example.com/tides"))

    assert page is not None
    assert "rock hollows" in page.text


@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/", "file:///etc"]
)
async def test_never_follows_a_redirect_to_an_internal_address(target):
    requested: list[str] = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": target})

    fetcher = make_fetcher(handler)

    assert await fetcher.fetch(result("https://example.com/go")) is None
    assert requested == ["https://example.com/go"]


async def test_gives_up_after_too_many_redirects():
    requested: list[str] = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": f"/hop{len(requested)}"})

    fetcher = make_fetcher(handler)

    assert await fetcher.fetch(result("https://example.com/loop")) is None
    assert len(requested) == MAX_REDIRECTS + 1


async def test_skips_names_that_resolve_to_internal_addresses():
    async def internal_dns(host: str) -> list[str]:
        return ["93.184.215.14", "10.0.0.5"] if host == "sneaky.example.com" else ["93.184.215.14"]

    def handler(request):
        if request.url.host == "sneaky.example.com":
            raise AssertionError(f"should not fetch {request.url}")
        return httpx.Response(302, headers={"location": "http://sneaky.example.com/"})

    fetcher = make_fetcher(handler, resolve=internal_dns)

    assert await fetcher.fetch(result("http://sneaky.example.com/")) is None
    assert await fetcher.fetch(result("https://example.com/go")) is None


async def test_skips_names_that_dont_resolve():
    async def no_dns(host: str) -> list[str]:
        raise OSError("name not known")

    fetcher = make_fetcher(lambda request: httpx.Response(200, html=ARTICLE), resolve=no_dns)

    assert await fetcher.fetch(result("https://example.com/tides")) is None


async def test_resolve_host_uses_the_system_resolver():
    addresses = await resolve_host("localhost")

    assert addresses
    assert all(ipaddress.ip_address(a).is_loopback for a in addresses)


def test_public_url_check():
    assert is_public_http_url("https://docs.python.org/3/")
    assert is_public_http_url("http://8.8.8.8/")
    assert not is_public_http_url("http://192.168.1.1/admin")
    assert not is_public_http_url("ftp://example.com/")


@pytest.mark.parametrize(
    "url",
    [
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://metadata.goog/",
        "http://instance-data/latest/",
        "http://service.internal/",
        "http://localhost./",
        "http://foo.localhost/",
        "http://printer.local./",
        "http://redis/",
        "http://user:hunter2@example.com/",
        "http://example.com:6379/",
        "http://example.com:99999/",
        "http://[::1/",
    ],
)
def test_public_url_check_refuses_internal_names_and_odd_urls(url):
    assert not is_public_http_url(url)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # AWS, GCP and Azure metadata
        "169.254.170.2",  # ECS task credentials
        "100.100.100.200",  # Alibaba metadata, inside CGNAT space
        "168.63.129.16",  # Azure wire server, a public address on paper
        "0.0.0.0",
        "224.0.0.1",  # multicast passes ipaddress' is_global
        "239.255.255.250",
        "::1",
        "::",
        "fe80::1",
        "fd00:ec2::254",  # AWS IPv6 metadata
        "::ffff:169.254.169.254",  # IPv4-mapped
        "64:ff9b::a9fe:a9fe",  # NAT64 wrapping 169.254.169.254, also passes is_global
        "64:ff9b::7f00:1",
        "2002:7f00:1::",  # 6to4 wrapping 127.0.0.1
        "2001:0:4136:e378:8000:63bf:80ff:fffe",  # Teredo wrapping 127.0.0.1
        "ff02::1",
        "not an address",
    ],
)
def test_refuses_addresses_that_are_not_public(address):
    assert not is_public_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "93.184.215.14", "2606:4700:4700::1111"])
def test_accepts_public_addresses(address):
    assert is_public_address(address)


async def test_a_bad_url_in_the_results_is_skipped_not_raised():
    fetcher = make_fetcher(lambda request: httpx.Response(200, html=ARTICLE))

    assert await fetcher.fetch(result("http://[::1/")) is None


class RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, fail: set[str] | None = None) -> None:
        self.hosts: list[str] = []
        self.fail = fail or set()

    async def connect_tcp(self, host, port, **_):
        self.hosts.append(host)
        if host in self.fail:
            raise httpcore.ConnectError(f"{host} unreachable")
        return httpcore.AsyncMockStream([])


async def test_connects_to_the_address_it_checked_not_the_name():
    inner = RecordingBackend()
    backend = PublicOnlyBackend(public_dns, inner=inner)

    await backend.connect_tcp("example.com", 443)

    assert inner.hosts == ["93.184.215.14"]


async def test_tries_the_next_checked_address_when_one_is_unreachable():
    async def two_addresses(host: str) -> list[str]:
        return ["2606:4700:4700::1111", "93.184.215.14"]

    inner = RecordingBackend(fail={"2606:4700:4700::1111"})
    backend = PublicOnlyBackend(two_addresses, inner=inner)

    await backend.connect_tcp("example.com", 443)

    assert inner.hosts == ["2606:4700:4700::1111", "93.184.215.14"]


async def test_refuses_at_connect_time_if_any_address_is_internal():
    async def mixed(host: str) -> list[str]:
        return ["93.184.215.14", "10.0.0.5"]

    inner = RecordingBackend()
    backend = PublicOnlyBackend(mixed, inner=inner)

    with pytest.raises(httpcore.ConnectError):
        await backend.connect_tcp("example.com", 443)
    assert inner.hosts == []


async def test_a_name_that_stops_resolving_is_a_connect_error():
    async def no_dns(host: str) -> list[str]:
        raise OSError("name not known")

    with pytest.raises(httpcore.ConnectError):
        await PublicOnlyBackend(no_dns, inner=RecordingBackend()).connect_tcp("example.com", 443)


async def test_dns_rebinding_never_reaches_the_internal_address():
    """The name looks public when it's checked and points at loopback when it's connected to.

    Uses a real listening socket and the real client, so a bypass would show up as a connection.
    """
    connections: list[object] = []

    async def on_connect(reader, writer):
        connections.append(writer)
        writer.write(RAW_RESPONSE)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    answers = iter([["93.184.215.14"], ["127.0.0.1"]])

    async def rebinding_dns(host: str) -> list[str]:
        return next(answers)

    # First answer: what a check before fetching would have seen.
    assert all(is_public_address(a) for a in await rebinding_dns("rebind.example.com"))

    async with server, make_http_client(1.0, resolve=rebinding_dns) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get(f"http://rebind.example.com:{port}/")

    assert connections == []


async def test_the_real_client_still_connects_through_the_guard(monkeypatch):
    # Without this the rebinding test above would also pass if nothing could connect at all.
    monkeypatch.setattr("research_agent.fetch.is_public_address", lambda _: True)

    async def on_connect(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(RAW_RESPONSE)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def loopback_dns(host: str) -> list[str]:
        return ["127.0.0.1"]

    async with server, make_http_client(1.0, resolve=loopback_dns) as client:
        response = await client.get(f"http://anything.example.com:{port}/")

    assert response.text == "hi"


class ListFetcher:
    def __init__(self, broken: set[str]) -> None:
        self.broken = broken
        self.calls: list[str] = []

    async def fetch(self, result: SearchResult) -> Page | None:
        self.calls.append(result.url)
        if result.url in self.broken:
            return None
        return Page(url=result.url, title=result.title, text="text")


async def test_read_pages_replaces_failures_and_respects_limit():
    results = [result(f"https://site{i}.com") for i in range(6)]
    fetcher = ListFetcher(broken={"https://site0.com", "https://site2.com"})

    pages = await read_pages(fetcher, results, limit=3)

    assert [p.url for p in pages] == ["https://site1.com", "https://site3.com", "https://site4.com"]
    assert "https://site5.com" not in fetcher.calls
