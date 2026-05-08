"""Single-shot Task 7 EPD agents — four progressive-disclosure settings.

Each setting reveals a different field set to the model:

- ``name_only``: just product_name + declared unit
- ``with_description``: + description
- ``with_composition``: + material composition breakdown
- ``with_region``: + manufacturing region (most disclosed)

This is the "tops-down" baseline: the model produces a single kgCO2e
estimate without any internal decomposition. The compositional
counterpart lives in ``stepwise.py``."""

from __future__ import annotations

import dataclasses
from typing import Literal

import pydantic as pyd
from pydantic_ai import RunContext
from pydantic_ai import Tool as PydanticAITool

from pcfbench.agents._task_agent import (
    AGENT_SDK_PREFIX,
    AgentSDKSingleshotAgent,
    PydanticAISingleshotAgent,
    TaskAgent,
)
from pcfbench.models.factory import build_agent
from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)

Disclosure = Literal["name_only", "with_description", "with_composition", "with_region"]

EPD_NAME_ONLY_SYSTEM_PROMPT = """\
You are an LCA expert estimating product carbon footprints. Given only a \
product name, estimate the cradle-to-gate greenhouse gas emissions in \
kg CO2 equivalent per declared unit.

Provide your best estimate as a single number. Use your knowledge of \
typical emission intensities for this product category."""

EPD_WITH_DESCRIPTION_SYSTEM_PROMPT = """\
You are an LCA expert estimating product carbon footprints. Given a product \
name and description, estimate the cradle-to-gate greenhouse gas emissions \
in kg CO2 equivalent per declared unit.

Provide your best estimate as a single number. Consider the product's \
materials, manufacturing processes, and typical emission intensities for \
this product category."""

EPD_WITH_COMPOSITION_SYSTEM_PROMPT = """\
You are an LCA expert estimating product carbon footprints. Given a product \
name, description, and material composition breakdown, estimate the \
cradle-to-gate greenhouse gas emissions in kg CO2 equivalent per declared unit.

Use the composition percentages to weight emission factors for each \
constituent material. Provide your best estimate as a single number."""

EPD_WITH_REGION_SYSTEM_PROMPT = """\
You are an LCA expert estimating product carbon footprints. Given a product \
name, description, material composition, and manufacturing region, estimate \
the cradle-to-gate greenhouse gas emissions in kg CO2 equivalent per \
declared unit.

Use regional emission factors (especially grid carbon intensity) and the \
composition breakdown to refine your estimate. Provide your best estimate \
as a single number."""

_PROMPTS_BY_DISCLOSURE: dict[Disclosure, str] = {
    "name_only": EPD_NAME_ONLY_SYSTEM_PROMPT,
    "with_description": EPD_WITH_DESCRIPTION_SYSTEM_PROMPT,
    "with_composition": EPD_WITH_COMPOSITION_SYSTEM_PROMPT,
    "with_region": EPD_WITH_REGION_SYSTEM_PROMPT,
}


@dataclasses.dataclass
class EPDInput:
    product_name: str
    description: str
    quantity_unit: str
    country_of_origin: str | None = None
    composition: str | None = None
    geography: str | None = None
    recycled_content: str | None = None
    source_url: str | None = None


class EPDOutput(pyd.BaseModel):
    kgco2e: float


@dataclasses.dataclass
class _Deps:
    submitted: EPDOutput | None = None


async def _submit_epd_estimate(
    ctx: RunContext[_Deps],
    kgco2e: float,
) -> dict:
    ctx.deps.submitted = EPDOutput(kgco2e=kgco2e)
    raise SubmitTerminated()


_SUBMIT_TOOL_NAME = "submit_epd_estimate"
_SUBMIT_TOOL_DESCRIPTION = "Submit your kgCO2e estimate."


def build_epd_agent(*, model_id: str, disclosure: Disclosure) -> TaskAgent:
    """Build a Task 7 EPD agent for the given disclosure setting."""
    if model_id.startswith(AGENT_SDK_PREFIX):
        return AgentSDKSingleshotAgent(
            model_id=model_id,
            system_prompt=_PROMPTS_BY_DISCLOSURE[disclosure],
            output_type=EPDOutput,
            submit_tool_name=_SUBMIT_TOOL_NAME,
            submit_tool_description=_SUBMIT_TOOL_DESCRIPTION,
        )
    submit_tool = PydanticAITool(
        _submit_epd_estimate,
        name=_SUBMIT_TOOL_NAME,
        description=_SUBMIT_TOOL_DESCRIPTION,
    )
    agent = build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=_PROMPTS_BY_DISCLOSURE[disclosure],
        tools=[submit_tool],
        deps_type=_Deps,
    )
    return PydanticAISingleshotAgent(
        agent=agent,
        make_deps=_Deps,
        get_output=lambda d: d.submitted,
        output_recovered=lambda d: d.submitted is not None,
    )


# Back-compat alias used by ``evals/runner.py``'s composition spec.
def build_epd_agent_with_composition(*, model_id: str) -> TaskAgent:
    return build_epd_agent(model_id=model_id, disclosure="with_composition")


def _build_user_prompt(inp: EPDInput, *, disclosure: Disclosure) -> str:
    """Emit the user-prompt body for one disclosure setting.

    Field inclusion mirrors the system-prompt's promise: a more-disclosed
    setting reveals strictly more fields than a less-disclosed one. The
    declared unit is always shown — without it the kgCO2e/unit answer
    is meaningless."""
    parts = [
        f"Product: {inp.product_name}",
        f"Declared unit: {inp.quantity_unit}",
    ]
    if disclosure in ("with_description", "with_composition", "with_region"):
        if inp.description:
            parts.append(f"Description: {inp.description}")
    if disclosure in ("with_composition", "with_region"):
        if inp.composition:
            parts.append(f"Material composition: {inp.composition}")
        if inp.recycled_content:
            parts.append(f"Recycled content: {inp.recycled_content}")
    if disclosure == "with_region":
        # Geography is the canonical "manufacturing region" field; fall
        # back to country_of_origin if geography is unset.
        region = inp.geography or inp.country_of_origin
        if region:
            parts.append(f"Manufacturing region: {region}")
    return "\n".join(parts)


async def run_epd(
    *, agent: TaskAgent, inp: EPDInput, disclosure: Disclosure
) -> EPDOutput | None:
    return await agent.run_task(_build_user_prompt(inp, disclosure=disclosure))


async def run_epd_with_composition(
    *, agent: TaskAgent, inp: EPDInput
) -> EPDOutput | None:
    return await run_epd(agent=agent, inp=inp, disclosure="with_composition")


async def run_epd_name_only(*, agent: TaskAgent, inp: EPDInput) -> EPDOutput | None:
    return await run_epd(agent=agent, inp=inp, disclosure="name_only")


async def run_epd_with_description(
    *, agent: TaskAgent, inp: EPDInput
) -> EPDOutput | None:
    return await run_epd(agent=agent, inp=inp, disclosure="with_description")


async def run_epd_with_region(*, agent: TaskAgent, inp: EPDInput) -> EPDOutput | None:
    return await run_epd(agent=agent, inp=inp, disclosure="with_region")
