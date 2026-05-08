"""Iteration-cap + force-submit runner.

Mirrors three PSWAgent semantics on top of Pydantic AI:

1. **Text-only is invalid.** If the model produces a final text response
   without calling the submit_* tool, PSWAgent appends a user message
   ``"Invalid response. You must use one of the terminating tools..."``
   and forces another iteration. Pydantic AI's default is to *terminate*
   on text output, so we manually re-prompt.

2. **Iteration cap.** Up to ``max_iterations`` model turns. PSWAgent
   counts every LLM call; we approximate with our outer-loop turn count
   (one outer turn = one model call after a text-only result; tool-call
   chains inside Pydantic AI's internal loop also count toward the same
   per-run usage limit).

3. **Final iteration: submit-only tools.** On the last turn we swap the
   agent for ``submit_only_agent`` (which exposes only submit_*) so the
   model has no choice but to submit. PSWAgent's
   ``_get_tools_available_for_iteration`` does the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from pcfbench.agents._trace import run_pydantic_ai_with_trace
from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)

D = TypeVar("D")


@dataclass
class AgentRunResult:
    """Result of running an agent through ``run_with_iteration_cap``."""

    output_text: str
    deps: Any
    usage_limit_hit: bool


async def run_with_iteration_cap(
    *,
    agent: Agent,
    submit_only_agent: Agent,
    user_prompt: str,
    deps: Any,
    submit_only_user_prompt_suffix: str,
    max_iterations: int = 20,
    output_recovered: Callable[[Any], bool],
    submit_tool_name: str = "submit_mapping",
) -> AgentRunResult:
    """Run *agent* with an iteration cap. If the cap is hit before
    ``output_recovered(deps)`` is True, fall back to *submit_only_agent*
    (which exposes only the terminating submit_* tool) so the agent must
    submit its best-so-far answer.

    Args:
        agent: full-tools agent (search/inspect/deep_thought/submit_*).
        submit_only_agent: stripped agent exposing only submit_*.
        user_prompt: the per-item prompt.
        deps: the per-call state object (e.g. ``MappingDeps``); both
            agents share it so the search-tracker state carries over to
            the submit-only pass.
        submit_only_user_prompt_suffix: extra text appended to the prompt
            on the submit-only retry.
        max_iterations: PSWAgent's max_iterations equivalent (default
            20, matching ``AGENTIC_MAX_ITERATIONS``).
        output_recovered: predicate over deps; if True after the main
            run, we don't fall back to the submit-only pass.
        submit_tool_name: name of the terminating tool the model must
            call (e.g. ``"submit_mapping"`` for mapping,
            ``"submit_triage"`` for triage). Used in the text-only
            nudge so we don't tell the triage agent to call a tool it
            doesn't have.
    """
    output_text = ""
    invalid_text_nudge = (
        f"Invalid response. You must call the {submit_tool_name} tool "
        f"with the required structured fields. Do not respond with "
        f"text only — call {submit_tool_name} now."
    )
    submit_budget = 3
    main_budget = max(1, max_iterations - submit_budget)
    history: list = []

    # Main pass — single agent.run with PSWAgent's full budget. The model
    # keeps context across all tool calls instead of being chopped into
    # fragmented outer turns (which lose conversation continuity and
    # cause the model to give up with hallucinated "I cannot find a
    # match" submissions on hard items like ed51 catalytic-converter).
    try:
        result = await run_pydantic_ai_with_trace(
            agent,
            user_prompt,
            deps=deps,
            usage_limits=UsageLimits(request_limit=main_budget),
        )
        output_text = result.output or ""
        history = result.all_messages()
    except SubmitTerminated:
        return AgentRunResult(output_text="", deps=deps, usage_limit_hit=False)
    except UsageLimitExceeded:
        # Cap hit; Pydantic AI discards in-flight history on this
        # exception, so we go to submit-only with empty history. The
        # submit-only agent will at least force a picklist-name guess.
        history = []
    except Exception as exc:  # noqa: BLE001
        if "unprocessed tool calls" not in str(exc):
            raise
        history = []

    if output_recovered(deps):
        return AgentRunResult(output_text=output_text, deps=deps, usage_limit_hit=False)

    # Re-prompt with the nudge if the main pass produced text only.
    if history:
        try:
            kwargs: dict[str, Any] = {
                "deps": deps,
                "usage_limits": UsageLimits(request_limit=submit_budget),
                "message_history": history,
            }
            result = await run_pydantic_ai_with_trace(
                agent, invalid_text_nudge, **kwargs
            )
            output_text = result.output or output_text
            history = result.all_messages()
        except SubmitTerminated:
            return AgentRunResult(output_text="", deps=deps, usage_limit_hit=False)
        except UsageLimitExceeded:
            pass
        except Exception as exc:  # noqa: BLE001
            if "unprocessed tool calls" not in str(exc):
                raise
            history = []

        if output_recovered(deps):
            return AgentRunResult(
                output_text=output_text, deps=deps, usage_limit_hit=False
            )

    # Submit-only fallback.
    try:
        kwargs = {
            "deps": deps,
            "usage_limits": UsageLimits(request_limit=submit_budget),
        }
        if history:
            kwargs["message_history"] = history
        final_result = await run_pydantic_ai_with_trace(
            submit_only_agent,
            invalid_text_nudge + "\n\n" + submit_only_user_prompt_suffix,
            **kwargs,
        )
        output_text = final_result.output or output_text
    except SubmitTerminated:
        pass
    except UsageLimitExceeded:
        pass
    except Exception:  # noqa: BLE001
        pass

    return AgentRunResult(
        output_text=output_text,
        deps=deps,
        usage_limit_hit=not output_recovered(deps),
    )
