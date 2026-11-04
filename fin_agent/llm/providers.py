"""Concrete LLM providers.

- ``ClaudeSubscriptionProvider`` (default): the Claude Agent SDK, which drives the
  locally installed Claude Code CLI and uses its login, so a Claude subscription
  works without an API key. Personal, local use only, within plan rate limits.
- ``AnthropicAPIProvider``: the Anthropic API with ``ANTHROPIC_API_KEY``; use for
  large eval sweeps.
- ``FakeProvider``: deterministic, for tests and offline demos.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from fin_agent.llm.base import LLMError, LLMProvider, LLMResult, extract_json, json_instruction


def _run(coro):
    """Run an async coroutine from sync code, also when a loop is already running (Streamlit)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class ClaudeSubscriptionProvider(LLMProvider):
    name = "claude_subscription"

    def __init__(self, model: str | None = None, max_turns: int = 3):
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as exc:
            raise LLMError("pip install claude-agent-sdk, and sign in once with `claude` (Claude Code CLI)") from exc
        self.model = model or os.getenv("FIN_AGENT_MODEL") or None
        self.max_turns = max_turns

    async def _complete(self, system: str, prompt: str, json_schema: dict | None) -> LLMResult:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

        options = ClaudeAgentOptions(
            system_prompt=system,
            tools=[],  # no built-in tools: this is pure reasoning; data access stays in our graph
            setting_sources=[],  # do not load the user's CLAUDE.md or settings into the analyst
            max_turns=self.max_turns,
            model=self.model,
            output_format={"type": "json_schema", "schema": json_schema} if json_schema else None,
        )
        text_parts: list[str] = []
        result: Any = None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                text_parts += [b.text for b in msg.content if isinstance(b, TextBlock)]
            elif isinstance(msg, ResultMessage):
                result = msg
        if result is None or result.is_error:
            raise LLMError(f"Claude subscription call failed: {getattr(result, 'errors', None) or getattr(result, 'subtype', 'no result')}")
        text = result.result or "".join(text_parts)
        data = None
        if json_schema:
            data = result.structured_output if result.structured_output is not None else extract_json(text)
        return LLMResult(text=text, data=data, usage={"cost_usd": result.total_cost_usd, **(result.usage or {})})

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        return _run(self._complete(system, prompt, json_schema))


class AnthropicAPIProvider(LLMProvider):
    name = "anthropic_api"

    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("pip install anthropic") from exc
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("FIN_AGENT_MODEL") or "claude-sonnet-5-5"
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        content = prompt + (json_instruction(json_schema) if json_schema else "")
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(getattr(b, "text", "") for b in msg.content)
        data = extract_json(text) if json_schema else None
        return LLMResult(text=text, data=data, usage={"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens})


class FakeProvider(LLMProvider):
    """Calls ``handler(system, prompt, schema)``; the handler returns a dict (JSON) or a string."""

    name = "fake"

    def __init__(self, handler: Callable[[str, str, dict | None], Any]):
        self.handler = handler
        self.calls: list[dict] = []

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt, "schema": json_schema})
        out = self.handler(system, prompt, json_schema)
        if isinstance(out, str):
            return LLMResult(text=out, data=extract_json(out) if json_schema else None)
        import json

        return LLMResult(text=json.dumps(out, ensure_ascii=False), data=out)


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.getenv("FIN_AGENT_LLM", "claude_subscription")).lower()
    if name == "claude_subscription":
        return ClaudeSubscriptionProvider()
    if name == "anthropic_api":
        return AnthropicAPIProvider()
    raise LLMError(f"Unknown LLM provider {name!r} (claude_subscription | anthropic_api)")
