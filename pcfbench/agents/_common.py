"""Shared single-shot agent runner.

Each task module (triage / decomposition / extraction / epd / mapping)
defines its own typed ``submit_*`` function plus a ``deps`` dataclass
holding the captured output. This module provides the shared
``run_singleshot`` driver: re-prompt loop, retry budget, exception
handling. Mirrors PSWAgent's max_iterations=1 single-shot semantics
plus a "you must call the submit tool" nudge if the model emits text
only.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import Usage, UsageLimits

from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)


async def run_singleshot(
    *,
    agent: Agent,
    user_prompt: str,
    deps: Any,
    output_recovered: Callable[[Any], bool],
    max_outer_turns: int = 4,
    usage: Usage | None = None,
) -> Any:
    """Run a single-shot agent until ``output_recovered(deps)`` is True
    (typically: submit_* fired) or we've used up the outer-turn budget.

    Each outer turn is one ``agent.run`` with up to 4 inner model
    requests. PSWAgent's max_iterations=1 single-shot is replicated by
    the model being forced (via the system prompt) to call submit_* on
    its first or second turn; the nudge re-prompt covers cases where
    the model produces a chain-of-thought text without the tool call.
    Returns the deps object so callers can read out their structured
    result.
    """
    nudge = (
        "Invalid response. You must call the submit tool with the "
        "required structured fields. Call it now and do not respond "
        "with text only."
    )
    history: list[Any] = []
    next_prompt = user_prompt
    for _ in range(max(1, max_outer_turns)):
        # Fresh per-turn Usage so ``usage_limits.request_limit`` applies
        # to this single ``agent.run`` rather than the cumulative count
        # across turns or callers (the latter would trip the limit on
        # the second call when ``usage`` is shared across many runs).
        turn_usage = Usage()
        try:
            kwargs: dict[str, Any] = {
                "deps": deps,
                "usage_limits": UsageLimits(request_limit=4),
                "usage": turn_usage,
            }
            if history:
                kwargs["message_history"] = history
            result = await agent.run(next_prompt, **kwargs)
            history = result.all_messages()
        except SubmitTerminated:
            if usage is not None:
                usage.incr(turn_usage)
            return deps
        except UsageLimitExceeded:
            if usage is not None:
                usage.incr(turn_usage)
            history = []
        except Exception as exc:  # noqa: BLE001
            if usage is not None:
                usage.incr(turn_usage)
            if "unprocessed tool calls" not in str(exc):
                raise
            history = []
        else:
            if usage is not None:
                usage.incr(turn_usage)
        if output_recovered(deps):
            return deps
        next_prompt = nudge
    return deps
