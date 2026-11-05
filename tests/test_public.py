import json
import runpy
from pathlib import Path

from research_agent.config import get_settings

ROOT = Path(__file__).parents[1]


def test_the_page_shows_a_report_from_the_examples_folder():
    shown = (ROOT / "public" / "example.md").read_bytes()

    assert shown == (ROOT / "examples" / "postgres-port.md").read_bytes()


def test_the_deployed_app_is_the_demo_not_the_open_api(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://not-connected-to")
    get_settings.cache_clear()
    try:
        app = runpy.run_path(str(ROOT / "app.py"))["app"]
    finally:
        get_settings.cache_clear()

    paths = {route.path for route in app.routes}
    assert paths
    assert all(path.startswith("/api/") for path in paths)


def test_the_function_can_outlast_a_time_slice():
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert config["regions"] == ["sin1"]
    assert config["functions"]["app.py"]["maxDuration"] > get_settings().demo_slice_seconds
