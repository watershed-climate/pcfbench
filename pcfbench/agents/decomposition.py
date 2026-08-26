"""Single-shot decomposition agent."""

from __future__ import annotations

import dataclasses
from typing import Any

import pydantic as pyd
from pydantic_ai import Agent, RunContext
from pydantic_ai import Tool as PydanticAITool
from pydantic_ai.usage import Usage

from pcfbench.agents._common import run_singleshot
from pcfbench.models.factory import build_agent
from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)

DECOMPOSITION_SYSTEM_PROMPT = """\
You are an LCA expert decomposing a finished product into its bill of \
materials (BOM). Given a product name and description, list the input \
materials that flow into the facility making the product -- i.e., the \
inputs one hop down the supply chain.

Do NOT list:
  - process or energy inputs (e.g. "electricity", "natural gas combustion")
  - generic descriptors (e.g. "raw materials", "ingredients")

Output rules:
  - List up to 8 components (fewer is fine for simple products).
  - Order from highest to lowest mass contribution to the finished product.
  - Use the most generic name a practitioner would use; only retain a brand \
or trade name if no generic equivalent exists.
  - Do NOT include percentages, quantities, or units.
  - Do NOT include duplicates."""


@dataclasses.dataclass
class DecompositionInput:
    product_name: str
    description: str
    quantity_unit: str


class DecompositionOutput(pyd.BaseModel):
    components: list[str]


@dataclasses.dataclass
class _Deps:
    submitted: DecompositionOutput | None = None


async def _submit_decomposition(
    ctx: RunContext[_Deps],
    components: list[str],
) -> dict[str, Any]:
    ctx.deps.submitted = DecompositionOutput(components=components)
    raise SubmitTerminated()


def build_decomposition_agent(*, model_id: str) -> Agent:
    submit_tool = PydanticAITool(
        _submit_decomposition,
        name="submit_decomposition",
        description="Submit the bill-of-materials components.",
    )
    return build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
        tools=[submit_tool],
        deps_type=_Deps,
    )


def _build_user_prompt(inp: DecompositionInput) -> str:
    return (
        f"Product: {inp.product_name}\n"
        f"Description: {inp.description}\n"
        f"Functional unit: {inp.quantity_unit}\n\n"
        f"List the BOM components."
    )


async def run_decomposition(
    *,
    agent: Agent,
    inp: DecompositionInput,
    usage: Usage | None = None,
) -> DecompositionOutput | None:
    deps = _Deps()
    await run_singleshot(
        agent=agent,
        user_prompt=_build_user_prompt(inp),
        deps=deps,
        output_recovered=lambda d: d.submitted is not None,
        usage=usage,
    )
    return deps.submitted
