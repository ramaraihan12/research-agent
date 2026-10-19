from contextlib import asynccontextmanager

import pytest
from typer.testing import CliRunner

from research_agent import cli

runner = CliRunner()


@pytest.fixture
def harness(make_harness, monkeypatch):
    h = make_harness()

    @asynccontextmanager
    async def fake_open_graph(settings):
        yield h.graph

    monkeypatch.setattr(cli, "open_graph", fake_open_graph)
    return h


def test_ask_with_yes_prints_the_report(harness, tmp_path):
    out = tmp_path / "report.md"

    result = runner.invoke(cli.app, ["ask", "How do tide pools work?", "--yes", "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# How do tide pools work?")
    assert "drafted a plan with 2 sub-questions" in result.stderr
    assert out.read_text(encoding="utf-8") == result.stdout.rstrip("\n") + "\n"


def test_editing_the_plan_in_the_terminal(harness):
    result = runner.invoke(
        cli.app, ["ask", "How do tide pools work?"], input="e\ntide pool zones\n\n"
    )

    assert result.exit_code == 0, result.output
    assert harness.search.queries == ["tide pool zones"]


def test_rejecting_the_plan_in_the_terminal(harness):
    result = runner.invoke(cli.app, ["ask", "How do tide pools work?"], input="r\n")

    assert result.exit_code == 0, result.output
    assert "no report (plan rejected)" in result.stderr
    assert harness.search.queries == []
