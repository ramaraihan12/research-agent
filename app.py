"""Vercel's entry point. It serves the public demo, never the open API in research_agent.api."""

from research_agent.demo import create_app

app = create_app()
