"""Agentic and single-shot mapping_with_context agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pydantic as pyd
from pydantic_ai import Agent
from pydantic_ai import Tool as PydanticAITool
from pydantic_ai.usage import Usage

from pcfbench.agents.runner import (
    AgentRunResult,
    run_with_iteration_cap,
)
from pcfbench.models.factory import build_agent
from pcfbench.tools.ecoinvent_tools import (
    INSPECT_DESCRIPTION,
    SEARCH_DESCRIPTION,
    MappingDeps,
    SubmitTerminated,
    inspect_ecoinvent,
    new_mapping_deps,
    search_ecoinvent,
    submit_mapping,
)
from pcfbench.tools.material_library import (
    MaterialLibrary,
)

MAPPING_SYSTEM_PROMPT = """\
You are an LCA expert performing material-to-ecoinvent mapping. Given a \
material name (and optionally additional context), select the best matching \
ecoinvent reference product from the provided list.

Your output must be an exact string from the ecoinvent_products list. \
Select the product that an LCA practitioner would choose for this material."""


MAPPING_SYSTEM_PROMPT_AGENTIC = """\
You are an LCA expert performing material-to-ecoinvent mapping. Given a \
material name (and optionally additional context), select the single best \
matching ecoinvent reference product from the PCFBench picklist \
(reference products of single-geography ``kg`` market activities in the \
ecoinvent v3.11 cut-off "Cut-Off AO" sheet).

You have access to two tools over the picklist:

  - search_ecoinvent: search by keyword and/or vector queries. Returns \
reference-product names only.
  - inspect_ecoinvent: return name and product information for \
specific reference products.

Use these tools to find and compare candidates. You should typically issue \
at least one search and inspect multiple candidates before deciding.

When you are confident, call submit_mapping with the exact reference \
product name you have chosen. The name must come from the picklist — do \
not invent names. Select the product that an LCA practitioner would choose \
for this material."""


_AGENTIC_SUBMIT_ONLY_SUFFIX = (
    "You have used your full search/inspect budget. Submit your best "
    "current answer now via submit_mapping with the exact reference "
    "product name from the picklist."
)


class MappingOutput(pyd.BaseModel):
    """Output schema for the single-shot submit_mapping tool call."""

    reference_product: str
    confidence: float | None = None


@dataclass
class MappingInput:
    material_name: str
    description: str | None = None
    supplier: str | None = None
    purchaser_context: dict[str, Any] | None = None


def _build_mapping_user_prompt(
    inp: MappingInput, *, picklist_names: list[str] | None
) -> str:
    """Build the user prompt for the mapping task."""
    prompt = f"Material: {inp.material_name}"
    if inp.description:
        prompt += f"\nDescription: {inp.description}"
    if inp.supplier:
        prompt += f"\nSupplier: {inp.supplier}"
    if inp.purchaser_context:
        prompt += f"\nPurchaser context: {json.dumps(inp.purchaser_context)}"
    if picklist_names is not None:
        products_str = "\n".join(picklist_names)
        prompt += (
            f"\n\nEcoinvent reference products:\n{products_str}"
            f"\n\nSelect the best matching product."
        )
    else:
        prompt += (
            "\n\nUse search_ecoinvent and inspect_ecoinvent to find "
            "candidates, compare them, and submit the single best match "
            "via submit_mapping."
        )
    return prompt


def build_mapping_agent_singleshot(*, model_id: str) -> Agent:
    """Single-shot mapping agent: no agentic tools; full picklist in the
    prompt; submit_mapping registered as a *regular tool* (not Pydantic
    AI's ``ToolOutput``).

    Why not ``ToolOutput``: that mode sets ``allow_text_output=False``
    which makes Pydantic AI emit ``tool_choice: any`` to Anthropic,
    forcing an immediate tool call and suppressing the model's
    chain-of-thought reasoning. The existing PSWAgent harness uses
    Anthropic's default ``tool_choice: auto`` (no override), so the
    model produces text-block reasoning then a tool_use block in the
    same turn. Replicating that requires a regular tool + ``output_type
    =str`` + the ``SubmitTerminated`` capture pattern (same as agentic).

    The ``submit_mapping`` tool also captures a calibrated confidence
    in [0, 1] so the same run produces both accuracy and calibration
    metrics; per Stage G paired testing the ask is neutral-to-slightly-
    positive on accuracy.
    """
    submit_tool = PydanticAITool(
        submit_mapping,
        name="submit_mapping",
        description="Submit the best matching ecoinvent activity.",
    )
    return build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=MAPPING_SYSTEM_PROMPT,
        tools=[submit_tool],
        deps_type=MappingDeps,
    )


def build_mapping_agent_agentic(*, model_id: str) -> tuple[Agent, Agent]:
    """Agentic mapping agent + a submit-only sibling for the
    iteration-cap fallback path."""
    # deep_thought tool removed: provider-side reasoning already covers
    # the deliberation channel; exposing an explicit scratchpad caused
    # tool-loop blowup on agentic mapping with thinking-on models.
    full_tools = [
        PydanticAITool(
            search_ecoinvent, name="search_ecoinvent", description=SEARCH_DESCRIPTION
        ),
        PydanticAITool(
            inspect_ecoinvent, name="inspect_ecoinvent", description=INSPECT_DESCRIPTION
        ),
        PydanticAITool(
            submit_mapping,
            name="submit_mapping",
            description=(
                "Submit the chosen ecoinvent reference product name. "
                "Must be an exact name from the picklist."
            ),
        ),
    ]
    submit_only_tools = [
        PydanticAITool(
            submit_mapping,
            name="submit_mapping",
            description=(
                "Submit the chosen ecoinvent reference product name. "
                "Must be an exact name from the picklist."
            ),
        ),
    ]
    full_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=MAPPING_SYSTEM_PROMPT_AGENTIC,
        tools=full_tools,
        deps_type=MappingDeps,
    )
    submit_only_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=MAPPING_SYSTEM_PROMPT_AGENTIC,
        tools=submit_only_tools,
        deps_type=MappingDeps,
    )
    return full_agent, submit_only_agent


async def run_mapping_singleshot(
    *,
    agent: Agent,
    inp: MappingInput,
    picklist_names: list[str],
    library: MaterialLibrary | None = None,
    usage: Usage | None = None,
) -> MappingOutput | None:
    """Run the single-shot mapping. Submission lands in
    ``deps.submitted_reference_product`` via the ``SubmitTerminated``
    pattern; we expect it to fire on the model's first turn (PSWAgent's
    ``max_iterations=1`` equivalent) but tolerate a re-prompt outer
    turn if the model produces text-only.

    ``library`` is required to construct ``MappingDeps``; pass the same
    instance the runner uses to construct the agent's tools."""
    if library is None:
        library = MaterialLibrary.load_default()
    user_prompt = _build_mapping_user_prompt(inp, picklist_names=picklist_names)
    deps = new_mapping_deps(library)
    invalid_text_nudge = (
        "Invalid response. You must call the submit_mapping tool with "
        "the exact reference product name from the picklist."
    )
    history: list[Any] = []
    for outer_turn in range(2):  # initial + at most one nudge retry
        prompt = user_prompt if outer_turn == 0 else invalid_text_nudge
        # Per-turn local Usage so we can roll tokens into the caller's
        # accumulator without polluting any shared usage_limits checks.
        turn_usage = Usage()
        try:
            kwargs: dict[str, Any] = {"deps": deps, "usage": turn_usage}
            if history:
                kwargs["message_history"] = history
            result = await agent.run(prompt, **kwargs)
            history = result.all_messages()
        except SubmitTerminated:
            if usage is not None:
                usage.incr(turn_usage)
            ref = deps.submitted_reference_product
            return (
                MappingOutput(
                    reference_product=ref.strip(),
                    confidence=deps.submitted_confidence,
                )
                if ref
                else None
            )
        if usage is not None:
            usage.incr(turn_usage)
        if deps.submitted_reference_product is not None:
            return MappingOutput(
                reference_product=deps.submitted_reference_product.strip(),
                confidence=deps.submitted_confidence,
            )
    ref = deps.submitted_reference_product
    return (
        MappingOutput(
            reference_product=ref.strip(),
            confidence=deps.submitted_confidence,
        )
        if ref
        else None
    )


async def run_mapping_agentic(
    *,
    agent: Agent,
    submit_only_agent: Agent,
    inp: MappingInput,
    library: MaterialLibrary,
    max_iterations: int = 20,
) -> AgentRunResult:
    deps = new_mapping_deps(library)
    user_prompt = _build_mapping_user_prompt(inp, picklist_names=None)
    return await run_with_iteration_cap(
        agent=agent,
        submit_only_agent=submit_only_agent,
        user_prompt=user_prompt,
        deps=deps,
        submit_tool_name="submit_mapping",
        submit_only_user_prompt_suffix=_AGENTIC_SUBMIT_ONLY_SUFFIX,
        max_iterations=max_iterations,
        output_recovered=lambda d: d.submitted_reference_product is not None,
    )
