import json
import re
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.config import Settings

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_chat_model(settings: Settings) -> BaseChatModel:
    kwargs: dict[str, Any] = {"temperature": settings.llm_temperature}
    if settings.llm_provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url
        kwargs["num_predict"] = settings.llm_max_output_tokens
        # Thinking models spend most of their budget on hidden reasoning,
        # which is slow on a laptop GPU and not needed for these short prompts.
        kwargs["reasoning"] = False
    else:
        kwargs["max_tokens"] = settings.llm_max_output_tokens
        if settings.llm_api_key:
            kwargs["api_key"] = settings.llm_api_key.get_secret_value()
    return init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **kwargs)


@dataclass(frozen=True)
class Completion:
    text: str
    tokens: int


async def complete(model: BaseChatModel, system: str, user: str) -> Completion:
    message = await model.ainvoke([SystemMessage(system), HumanMessage(user)])
    text = strip_think(message.text)
    usage = message.usage_metadata
    if usage and usage.get("total_tokens"):
        tokens = usage["total_tokens"]
    else:
        tokens = estimate_tokens(system + user + text)
    return Completion(text=text, tokens=tokens)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def strip_think(text: str) -> str:
    """Drop inline <think> blocks some reasoning models emit in plain content."""
    return _THINK_BLOCK.sub("", text).strip()


def extract_json(text: str) -> Any | None:
    """Return the first JSON object or array found in a model reply, or None."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return value
    return None
