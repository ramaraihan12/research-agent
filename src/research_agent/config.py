from dataclasses import dataclass
from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class Limits:
    """Hard caps that keep a run from going on forever."""

    max_sub_questions: int = 3
    max_rounds: int = 2
    max_pages: int = 10
    pages_per_question: int = 2
    token_budget: int = 60_000
    results_per_query: int = 5


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: str = "ollama"
    llm_model: str = "qwen3.5:9b"
    llm_api_key: SecretStr | None = None
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = Field(default=1500, ge=64)
    # Hosted models only. Left unset, each provider uses its own defaults.
    llm_timeout: float | None = Field(default=None, gt=0)
    llm_max_retries: int | None = Field(default=None, ge=1)
    ollama_base_url: str = "http://localhost:11434"

    database_url: str | None = None
    tavily_api_key: SecretStr | None = None

    max_sub_questions: int = Field(default=Limits.max_sub_questions, ge=1, le=10)
    max_rounds: int = Field(default=Limits.max_rounds, ge=1, le=5)
    max_pages: int = Field(default=Limits.max_pages, ge=1, le=40)
    pages_per_question: int = Field(default=Limits.pages_per_question, ge=1, le=10)
    token_budget: int = Field(default=Limits.token_budget, ge=1_000)
    results_per_query: int = Field(default=Limits.results_per_query, ge=1, le=20)

    search_timeout: float = Field(default=15.0, gt=0)
    fetch_timeout: float = Field(default=10.0, gt=0)
    max_page_bytes: int = Field(default=2_000_000, ge=10_000)
    max_page_chars: int = Field(default=6_000, ge=500)

    # The public demo (research_agent.demo). The CLI and the local API ignore these.
    demo_runs_per_day: int = Field(default=10, ge=1)
    demo_runs_per_ip_hour: int = Field(default=2, ge=1)
    demo_runs_per_ip_day: int = Field(default=3, ge=1)
    demo_requests_per_ip_minute: int = Field(default=30, ge=1)
    demo_max_search_credits: int = Field(default=3, ge=1)
    demo_max_model_calls: int = Field(default=12, ge=1)
    demo_slice_seconds: float = Field(default=240.0, gt=0)
    demo_max_slices: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _every_sub_question_gets_a_page(self) -> Self:
        if self.max_pages < self.max_sub_questions:
            raise ValueError("MAX_PAGES must be at least MAX_SUB_QUESTIONS")
        return self

    @property
    def limits(self) -> Limits:
        return Limits(
            max_sub_questions=self.max_sub_questions,
            max_rounds=self.max_rounds,
            max_pages=self.max_pages,
            pages_per_question=self.pages_per_question,
            token_budget=self.token_budget,
            results_per_query=self.results_per_query,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
