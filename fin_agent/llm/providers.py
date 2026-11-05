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
import time
from typing import Any, Callable

from fin_agent.llm.base import LLMError, LLMProvider, LLMResult, extract_json, json_instruction
from fin_agent.llm.usage import normalize


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
        model_seen: str | None = None
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                model_seen = getattr(msg, "model", None) or model_seen
                text_parts += [b.text for b in msg.content if isinstance(b, TextBlock)]
            elif isinstance(msg, ResultMessage):
                result = msg
        usage = self._usage(result, model_seen)
        if result is None or result.is_error:
            raise LLMError(f"Claude subscription call failed: {getattr(result, 'errors', None) or getattr(result, 'subtype', 'no result')}",
                           usage=usage)
        text = result.result or "".join(text_parts)
        data = None
        if json_schema:
            try:
                data = result.structured_output if result.structured_output is not None else extract_json(text)
            except LLMError as exc:
                raise LLMError(str(exc), usage=usage) from exc
        return LLMResult(text=text, data=data, usage=usage)

    def _usage(self, result: Any, model_seen: str | None) -> dict | None:
        if result is None:
            return None
        model_usage = getattr(result, "model_usage", None) or {}
        model = next(iter(model_usage), None) if len(model_usage) == 1 else (model_seen or self.model)
        cost = getattr(result, "total_cost_usd", None)
        return normalize(getattr(result, "usage", None), provider=self.name, model=model,
                         duration_ms=getattr(result, "duration_ms", None),
                         cost_usd=cost, cost_source="sdk_estimate" if cost is not None else None)

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
        t0 = time.perf_counter()
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
        )
        usage = normalize(msg.usage, provider=self.name, model=getattr(msg, "model", None) or self.model,
                          duration_ms=(time.perf_counter() - t0) * 1000)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        try:
            data = extract_json(text) if json_schema else None
        except LLMError as exc:
            raise LLMError(str(exc), usage=usage) from exc
        return LLMResult(text=text, data=data, usage=usage)


class FakeProvider(LLMProvider):
    """Calls ``handler(system, prompt, schema)``; the handler returns a dict (JSON) or a string."""

    name = "fake"

    def __init__(self, handler: Callable[[str, str, dict | None], Any], model: str | None = None):
        self.handler = handler
        self.model = model  # set a priced model id to exercise cost accounting in tests
        self.calls: list[dict] = []

    def _usage(self, system: str, prompt: str, text: str) -> dict:
        # deterministic stand-in: ~4 characters per token
        return normalize({"input_tokens": (len(system) + len(prompt)) // 4, "output_tokens": len(text) // 4},
                         provider=self.name, model=self.model, duration_ms=0)

    def complete(self, system: str, prompt: str, json_schema: dict | None = None) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt, "schema": json_schema})
        out = self.handler(system, prompt, json_schema)
        if isinstance(out, str):
            usage = self._usage(system, prompt, out)
            try:
                data = extract_json(out) if json_schema else None
            except LLMError as exc:
                raise LLMError(str(exc), usage=usage) from exc
            return LLMResult(text=out, data=data, usage=usage)
        import json

        text = json.dumps(out, ensure_ascii=False)
        return LLMResult(text=text, data=out, usage=self._usage(system, prompt, text))


def get_provider(name: str | None = None) -> LLMProvider:
    name = (name or os.getenv("FIN_AGENT_LLM", "claude_subscription")).lower()
    if name == "claude_subscription":
        return ClaudeSubscriptionProvider()
    if name == "anthropic_api":
        return AnthropicAPIProvider()
    raise LLMError(f"Unknown LLM provider {name!r} (claude_subscription | anthropic_api)")
