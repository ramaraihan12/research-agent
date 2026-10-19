import asyncio

import httpx
import pytest

from research_agent.fetch import MAX_REDIRECTS, Page, PageFetcher, is_public_http_url, read_pages
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


def make_fetcher(handler, **overrides) -> PageFetcher:
    # follow_redirects=True on purpose: the fetcher has to check each hop itself anyway.
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    options = {"timeout": 2.0, "max_bytes": 100_000, "max_chars": 5_000} | overrides
    return PageFetcher(client, **options)


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


def test_public_url_check():
    assert is_public_http_url("https://docs.python.org/3/")
    assert is_public_http_url("http://8.8.8.8/")
    assert not is_public_http_url("http://192.168.1.1/admin")
    assert not is_public_http_url("ftp://example.com/")


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
