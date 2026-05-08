"""Per-task trace buffer + serializers shared by both backends.

The eval runner installs a fresh list via ``set_trace_buffer`` for each
item when ``--trace`` is on. Both backends append serialized messages
into the same buffer so a unified ``<run>.trace.jsonl`` covers either
``agent-sdk:`` runs (claude-agent-sdk message stream) or pydantic-ai
runs (alternating ``ModelRequest`` / ``ModelResponse``).

Why a ``contextvars.ContextVar`` rather than a kwarg? Threading a
``trace`` parameter through every ``run_*()`` and ``run_with_iteration_cap``
call site (plus the ``TaskAgent`` protocol) would touch ~10 functions
and the dispatcher; a per-asyncio-task contextvar keeps the change
localized to where messages actually stream past.

For pydantic-ai, ``run_pydantic_ai_with_trace`` is the iter-based
substitute for ``agent.run(...)``. It's needed because the
``submit_*`` tools raise ``SubmitTerminated`` to short-circuit
``agent.run()`` (so ``result`` is never assigned and the final
``ModelResponse`` is lost). Iterating via ``agent.iter()`` lets us call
``agent_run.new_messages()`` from inside the ``except`` block and
preserve the model's last text/thinking/tool_call before re-raising.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.agent import AgentRunResult


_TRACE_BUFFER: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("pcfbench_trace_buffer", default=None)
)


def set_trace_buffer(
    buf: list[dict[str, Any]] | None,
) -> contextvars.Token[list[dict[str, Any]] | None]:
    return _TRACE_BUFFER.set(buf)


def reset_trace_buffer(
    token: contextvars.Token[list[dict[str, Any]] | None],
) -> None:
    _TRACE_BUFFER.reset(token)


def get_trace_buffer() -> list[dict[str, Any]] | None:
    return _TRACE_BUFFER.get()


# ---- claude-agent-sdk serializers ---------------------------------


def serialize_agent_sdk_message(message: Any) -> dict[str, Any]:
    """Best-effort JSON-safe view of one streamed agent-sdk message.

    Captures the fields useful for post-hoc trace inspection (text,
    thinking, tool I/O, result + usage). Unknown message / block types
    fall back to ``repr()`` rather than raising; callers also pass
    ``default=str`` to ``json.dumps`` as a backstop."""
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        UserMessage,
    )

    cls = type(message).__name__
    if isinstance(message, SystemMessage):
        return {
            "type": cls,
            "subtype": getattr(message, "subtype", None),
            "data": getattr(message, "data", None),
        }
    if isinstance(message, AssistantMessage):
        return {
            "type": cls,
            "model": getattr(message, "model", None),
            "content": [_serialize_agent_sdk_block(b) for b in message.content],
        }
    if isinstance(message, UserMessage):
        content = message.content
        body = (
            content
            if isinstance(content, str)
            else [_serialize_agent_sdk_block(b) for b in content]
        )
        return {"type": cls, "content": body}
    if isinstance(message, ResultMessage):
        return {
            "type": cls,
            "subtype": message.subtype,
            "is_error": message.is_error,
            "duration_ms": message.duration_ms,
            "num_turns": message.num_turns,
            "total_cost_usd": message.total_cost_usd,
            "result": message.result,
            "usage": message.usage,
        }
    return {"type": cls, "repr": repr(message)}


def _serialize_agent_sdk_block(block: Any) -> dict[str, Any]:
    from claude_agent_sdk import (
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )

    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        return {"type": "thinking", "thinking": block.thinking}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    return {"type": type(block).__name__, "repr": repr(block)}


# ---- pydantic-ai serializers --------------------------------------


def serialize_pydantic_ai_message(message: Any) -> dict[str, Any]:
    """Serialize a ``ModelRequest`` / ``ModelResponse`` to a flat dict.

    Unknown message types fall back to ``repr()``; the eval runner
    writes with ``default=str`` so partially-typed fields (e.g.
    ``ToolCallPart.args`` may arrive as a dict OR a JSON string
    depending on provider) still serialize."""
    from pydantic_ai.messages import ModelRequest, ModelResponse

    cls = type(message).__name__
    if isinstance(message, ModelRequest):
        return {
            "type": cls,
            "parts": [_serialize_pydantic_ai_part(p) for p in message.parts],
        }
    if isinstance(message, ModelResponse):
        return {
            "type": cls,
            "model_name": getattr(message, "model_name", None),
            "finish_reason": getattr(message, "finish_reason", None),
            "parts": [_serialize_pydantic_ai_part(p) for p in message.parts],
        }
    return {"type": cls, "repr": repr(message)}


def _serialize_pydantic_ai_part(part: Any) -> dict[str, Any]:
    from pydantic_ai.messages import (
        RetryPromptPart,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    if isinstance(part, SystemPromptPart):
        return {"type": "system_prompt", "content": part.content}
    if isinstance(part, UserPromptPart):
        return {"type": "user_prompt", "content": part.content}
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.content}
    if isinstance(part, ThinkingPart):
        return {"type": "thinking", "thinking": part.content}
    if isinstance(part, ToolCallPart):
        return {
            "type": "tool_call",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "args": part.args,
        }
    if isinstance(part, ToolReturnPart):
        return {
            "type": "tool_return",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": part.content,
        }
    if isinstance(part, RetryPromptPart):
        return {
            "type": "retry_prompt",
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": part.content,
        }
    return {"type": type(part).__name__, "repr": repr(part)}


# ---- pydantic-ai run wrapper --------------------------------------


async def run_pydantic_ai_with_trace(
    agent: "Agent",
    user_prompt: str,
    **iter_kwargs: Any,
) -> "AgentRunResult":
    """Iter-based replacement for ``agent.run(...)`` with trace capture.

    Behaves like ``agent.run()`` for the caller: returns an
    ``AgentRunResult`` on success, re-raises whatever the agent raised.

    The reason we don't just wrap ``agent.run()`` is that the
    ``submit_*`` tools used across PCFBench raise ``SubmitTerminated``
    to short-circuit the run. When that happens inside ``agent.run()``,
    pydantic-ai's graph queues the failure and re-raises it from the
    ``__aexit__`` of its internal context manager — by which point
    ``agent.run()`` has already discarded the partial state, so the
    final ``ModelResponse`` (the one containing the model's text /
    thinking / tool_call) is lost.

    Iterating via ``agent.iter()`` lets us snapshot
    ``agent_run.all_messages()`` in a ``finally`` *inside* the async-
    with, before the graph's ``__aexit__`` re-raises. We slice off the
    prior ``message_history`` (if any) so the trace buffer accumulates
    only the new turn's messages.

    No-op for the buffer when no buffer is set; the iter-vs-run
    semantics are otherwise identical because ``Agent.run()`` is itself
    implemented on top of ``Agent.iter()``."""
    trace_buf = _TRACE_BUFFER.get()
    prior_count = len(iter_kwargs.get("message_history") or [])
    snapshot: list[Any] = []
    try:
        async with agent.iter(user_prompt, **iter_kwargs) as agent_run:
            try:
                async for _ in agent_run:
                    pass
            finally:
                # Captured here, inside the with-body's finally, because
                # ``__aexit__`` is what re-raises ``SubmitTerminated``
                # and we'd lose the messages otherwise. ``new_messages``
                # returns [] on the tool-raise path; ``all_messages``
                # has them, so we keep the latter and trim the prior
                # message_history below.
                try:
                    snapshot = list(agent_run.all_messages())
                except Exception:  # noqa: BLE001
                    snapshot = []
            return agent_run.result
    finally:
        if trace_buf is not None:
            for msg in snapshot[prior_count:]:
                trace_buf.append(serialize_pydantic_ai_message(msg))
