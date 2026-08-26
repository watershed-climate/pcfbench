"""Single-shot and agentic triage agents."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any

import pydantic as pyd
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai import Tool as PydanticAITool
from pydantic_ai.usage import Usage

from pcfbench.agents._common import run_singleshot
from pcfbench.agents.runner import (
    AgentRunResult,
    run_with_iteration_cap,
)
from pcfbench.models.factory import build_agent
from pcfbench.tools.ecoinvent_tools import (
    INSPECT_DESCRIPTION,
    SEARCH_DESCRIPTION,
    SubmitTerminated,
    TriageDeps,
    inspect_ecoinvent,
    new_triage_deps,
    search_ecoinvent,
    submit_triage,
)
from pcfbench.tools.material_library import (
    MaterialLibrary,
)

TRIAGE_SYSTEM_PROMPT = """\
You are an LCA expert performing a routing decision during the \
bill-of-materials (BOM) expansion of a product.

You will receive two things:

  1. A MARKET node --- a specific input within a product's BOM. \
THIS is the thing you must triage.
  2. A ROOT MATERIAL CONTEXT --- the original material whose BOM \
expansion this market sits inside. Use this only as context to \
interpret the market; DO NOT triage the root material itself.

For the MARKET (not the root material context), decide:
  - should_map=true  if the MARKET can reasonably be matched directly to \
one of the ecoinvent activities listed in the user message.
  - should_map=false if the MARKET is too complex or composite and needs \
to be decomposed further into sub-components first."""


TRIAGE_SYSTEM_PROMPT_AGENTIC = """\
You are an LCA expert performing a routing decision during the \
bill-of-materials (BOM) expansion of a product.

You will receive two things:

  1. A MARKET node --- a specific input within a product's BOM. \
THIS is the thing you must triage.
  2. A ROOT MATERIAL CONTEXT --- the original material whose BOM \
expansion this market sits inside. Use this only as context to \
interpret the market; DO NOT triage the root material itself.

You have access to the PCFBench ecoinvent picklist (single-geography \
``kg`` market activities from the ecoinvent v3.11 cut-off "Cut-Off AO" \
sheet) via two tools:

  - search_ecoinvent: search the picklist by keyword and/or vector \
queries. Returns ecoinvent activity names only.
  - inspect_ecoinvent: return activity name and product information \
for specific ecoinvent activities.

Use these tools to investigate whether the MARKET can be matched directly \
to one or more ecoinvent activities at the right granularity. You should \
typically issue at least one search and inspect a few candidates before \
deciding.

When you are confident, call submit_triage with should_map:
  - should_map=true  if the MARKET can reasonably be matched directly to \
an ecoinvent activity you have found.
  - should_map=false if the MARKET is too complex or composite and needs \
to be decomposed further into sub-components first."""


# Ablation: same agentic harness, but the MARKET is triaged without the
# ROOT MATERIAL CONTEXT block in the user prompt.  Tests how much of the
# triage signal comes from knowing what BOM expansion the market sits
# inside vs. the market name/description alone.
TRIAGE_SYSTEM_PROMPT_AGENTIC_NO_CONTEXT = """\
You are an LCA expert performing a routing decision during the \
bill-of-materials (BOM) expansion of a product.

You will receive a MARKET node --- a specific input within a product's \
BOM.  This is the thing you must triage.

You have access to the PCFBench ecoinvent picklist (single-geography \
``kg`` market activities from the ecoinvent v3.11 cut-off "Cut-Off AO" \
sheet) via two tools:

  - search_ecoinvent: search the picklist by keyword and/or vector \
queries. Returns ecoinvent activity names only.
  - inspect_ecoinvent: return activity name and product information \
for specific ecoinvent activities.

Use these tools to investigate whether the MARKET can be matched directly \
to one or more ecoinvent activities at the right granularity. You should \
typically issue at least one search and inspect a few candidates before \
deciding.

When you are confident, call submit_triage with should_map:
  - should_map=true  if the MARKET can reasonably be matched directly to \
an ecoinvent activity you have found.
  - should_map=false if the MARKET is too complex or composite and needs \
to be decomposed further into sub-components first."""


@dataclasses.dataclass
class TriageMarket:
    name: str
    description: str


@dataclasses.dataclass
class TriageMaterialContext:
    name: str | None
    description: str | None
    material_name: str


@dataclasses.dataclass
class TriageInput:
    market: TriageMarket
    material_context_data: TriageMaterialContext


class TriageOutput(pyd.BaseModel):
    should_map: bool
    confidence: float | None = None


@dataclasses.dataclass
class _SingleShotTriageDeps:
    submitted: TriageOutput | None = None


def _build_triage_user_prompt(
    inp: TriageInput,
    *,
    picklist_names: list[str] | None,
    include_material_context: bool = True,
) -> str:
    """Render the triage user prompt.

    ``include_material_context=False`` drops the ROOT MATERIAL CONTEXT
    block; the model sees only the MARKET node.  This is the user-prompt
    side of the ``pcfbench_triage_agentic_no_context`` ablation.
    """
    market = inp.market
    body = (
        f"MARKET (this is the thing to triage):\n"
        f"  name: {market.name}\n"
        f"  description: {market.description}"
    )
    if include_material_context:
        ctx = inp.material_context_data
        body += (
            "\n\n"
            f"ROOT MATERIAL CONTEXT (the original material whose BOM expansion "
            f"contains the market above; DO NOT triage this):\n"
            f"  material_name: {ctx.material_name}\n"
            f"  name: {ctx.name or 'Not provided'}\n"
            f"  description: {ctx.description or 'Not provided'}"
        )
    if picklist_names is not None:
        products_str = "\n".join(picklist_names)
        return body + f"\n\nEcoinvent activities:\n{products_str}"
    return body + (
        "\n\nUse search_ecoinvent and inspect_ecoinvent to investigate, "
        "then call submit_triage with your decision."
    )


# ---- single-shot ----


_TRIAGE_CONFIDENCE_FIELD = Field(
    ge=0.0,
    le=1.0,
    description=(
        "Calibrated probability in [0, 1] that the submitted "
        "should_map decision is correct. Be calibrated: across many "
        "decisions at confidence X, you should be right X fraction of "
        "the time."
    ),
)


async def _submit_triage_singleshot(
    ctx: RunContext[_SingleShotTriageDeps],
    should_map: bool,
    confidence: Annotated[float, _TRIAGE_CONFIDENCE_FIELD],
) -> dict[str, Any]:
    ctx.deps.submitted = TriageOutput(should_map=should_map, confidence=confidence)
    raise SubmitTerminated()


def build_triage_agent_singleshot(*, model_id: str) -> Agent:
    submit_tool = PydanticAITool(
        _submit_triage_singleshot,
        name="submit_triage",
        description="Submit your triage decision.",
    )
    return build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        tools=[submit_tool],
        deps_type=_SingleShotTriageDeps,
    )


async def run_triage_singleshot(
    *,
    agent: Agent,
    inp: TriageInput,
    picklist_names: list[str],
    usage: Usage | None = None,
) -> TriageOutput | None:
    deps = _SingleShotTriageDeps()
    user_prompt = _build_triage_user_prompt(inp, picklist_names=picklist_names)
    await run_singleshot(
        agent=agent,
        user_prompt=user_prompt,
        deps=deps,
        output_recovered=lambda d: d.submitted is not None,
        usage=usage,
    )
    return deps.submitted


# ---- agentic ----


def build_triage_agent_agentic(*, model_id: str) -> tuple[Agent, Agent]:
    # deep_thought tool removed: provider-side reasoning (thinking_budget)
    # already gives the model a private deliberation channel, and exposing
    # an extra tool-call scratchpad caused observable tool-loop blowup
    # (the model would alternate between deep_thought and search/inspect
    # without converging on a submission).
    full_tools = [
        PydanticAITool(
            search_ecoinvent, name="search_ecoinvent", description=SEARCH_DESCRIPTION
        ),
        PydanticAITool(
            inspect_ecoinvent, name="inspect_ecoinvent", description=INSPECT_DESCRIPTION
        ),
        PydanticAITool(
            submit_triage,
            name="submit_triage",
            description="Submit your triage decision (should_map: bool).",
        ),
    ]
    submit_only_tools = [
        PydanticAITool(
            submit_triage,
            name="submit_triage",
            description="Submit your triage decision (should_map: bool).",
        ),
    ]
    full_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=TRIAGE_SYSTEM_PROMPT_AGENTIC,
        tools=full_tools,
        deps_type=TriageDeps,
    )
    submit_only_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=TRIAGE_SYSTEM_PROMPT_AGENTIC,
        tools=submit_only_tools,
        deps_type=TriageDeps,
    )
    return full_agent, submit_only_agent


async def run_triage_agentic(
    *,
    agent: Agent,
    submit_only_agent: Agent,
    inp: TriageInput,
    library: MaterialLibrary,
    max_iterations: int = 20,
) -> AgentRunResult:
    deps = new_triage_deps(library)
    user_prompt = _build_triage_user_prompt(inp, picklist_names=None)
    return await run_with_iteration_cap(
        agent=agent,
        submit_only_agent=submit_only_agent,
        user_prompt=user_prompt,
        deps=deps,
        submit_tool_name="submit_triage",
        submit_only_user_prompt_suffix=(
            "You have used your full search/inspect budget. Submit your "
            "current best decision now via submit_triage."
        ),
        max_iterations=max_iterations,
        output_recovered=lambda d: d.submitted_should_map is not None,
    )


# ---- agentic, no-material-context ablation ----


def build_triage_agent_agentic_no_context(*, model_id: str) -> tuple[Agent, Agent]:
    """Same agentic harness as ``build_triage_agent_agentic`` but without
    the ROOT MATERIAL CONTEXT in the system prompt -- ablates how much
    of the triage signal comes from knowing the parent BOM context."""
    full_tools = [
        PydanticAITool(
            search_ecoinvent, name="search_ecoinvent", description=SEARCH_DESCRIPTION
        ),
        PydanticAITool(
            inspect_ecoinvent, name="inspect_ecoinvent", description=INSPECT_DESCRIPTION
        ),
        PydanticAITool(
            submit_triage,
            name="submit_triage",
            description="Submit your triage decision (should_map: bool).",
        ),
    ]
    submit_only_tools = [
        PydanticAITool(
            submit_triage,
            name="submit_triage",
            description="Submit your triage decision (should_map: bool).",
        ),
    ]
    full_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=TRIAGE_SYSTEM_PROMPT_AGENTIC_NO_CONTEXT,
        tools=full_tools,
        deps_type=TriageDeps,
    )
    submit_only_agent = build_agent(
        model_id=model_id,
        agentic=True,
        output_type=str,
        system_prompt=TRIAGE_SYSTEM_PROMPT_AGENTIC_NO_CONTEXT,
        tools=submit_only_tools,
        deps_type=TriageDeps,
    )
    return full_agent, submit_only_agent


async def run_triage_agentic_no_context(
    *,
    agent: Agent,
    submit_only_agent: Agent,
    inp: TriageInput,
    library: MaterialLibrary,
    max_iterations: int = 20,
) -> AgentRunResult:
    deps = new_triage_deps(library)
    user_prompt = _build_triage_user_prompt(
        inp, picklist_names=None, include_material_context=False
    )
    return await run_with_iteration_cap(
        agent=agent,
        submit_only_agent=submit_only_agent,
        user_prompt=user_prompt,
        deps=deps,
        submit_tool_name="submit_triage",
        submit_only_user_prompt_suffix=(
            "You have used your full search/inspect budget. Submit your "
            "current best decision now via submit_triage."
        ),
        max_iterations=max_iterations,
        output_recovered=lambda d: d.submitted_should_map is not None,
    )
