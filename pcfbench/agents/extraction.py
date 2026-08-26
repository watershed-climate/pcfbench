"""Single-shot extraction agent (Tasks 4 + 5)."""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any

import pydantic as pyd
from pydantic import Field
from pydantic_ai import Agent, RunContext
from pydantic_ai import Tool as PydanticAITool

from pcfbench.agents._common import run_singleshot
from pcfbench.models.factory import build_agent
from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)

EXTRACTION_SYSTEM_PROMPT = """\
You are an LCA expert extracting physical parameters from technical \
documents. Given a query about a specific parameter and the full text of a \
source document, extract all relevant numerical claims.

For each claim, provide the numerical value and its unit exactly as \
expressed in the document. The unit field must be JUST the unit symbol \
or short canonical name (e.g. ``%``, ``h``, ``kg/kg``, ``MJ/kg``, \
``mass%``) — do NOT include descriptive prose, qualifiers, or context \
in the unit field. Put any contextual notes in the value or omit them. \
If the document states a range, use the midpoint. Only extract values \
that are directly stated in or clearly derivable from the document \
text. Do not estimate or hallucinate values."""


# Query-only ablation: no source document, model emits its prior over the
# parameter from training-data knowledge.  Establishes an F1 floor that
# reflects how good the model's priors are when explicitly elicited (vs.
# being suppressed by a no-hallucinate instruction).
EXTRACTION_QUERY_ONLY_SYSTEM_PROMPT = """\
You are an LCA expert. You will receive a query about a physical parameter \
(e.g. a material input rate or an energy intensity), but no source \
document. Your task is to provide your best numerical estimate(s) of the \
queried parameter using only your general knowledge of the domain --- \
typical values, industry conventions, textbook ranges, etc.

For each claim, provide the numerical value and its unit, matching the \
unit families requested in the query. The unit field must be JUST the \
unit symbol or short canonical name (e.g. ``%``, ``h``, ``kg/kg``, \
``MJ/kg``, ``mass%``) — do NOT include descriptive prose, qualifiers, \
or context. If you would naturally cite a range, report the midpoint. \
It is acceptable to emit multiple claims when the parameter has \
different typical values in different contexts (e.g. hydraulic vs. \
all-electric injection moulding) --- treat each as a separate claim. \
Do not refuse on the grounds that no document was provided; the goal \
is precisely to elicit your priors."""


@dataclasses.dataclass
class ExtractionInput:
    query: str
    document_text: str
    source_url: str | None = None


class ExtractionClaim(pyd.BaseModel):
    value: float
    unit: Annotated[
        str,
        Field(
            description=(
                "Just the unit symbol or short canonical form (e.g. '%', "
                "'h', 'kg/kg', 'MJ/kg', 'mass%'). Do NOT include "
                "descriptive prose, qualifiers, or context — those "
                "belong in surrounding metadata, not in the unit field."
            )
        ),
    ]


class ExtractionOutput(pyd.BaseModel):
    claims: list[ExtractionClaim]


@dataclasses.dataclass
class _Deps:
    submitted: ExtractionOutput | None = None


async def _submit_extraction(
    ctx: RunContext[_Deps],
    claims: list[ExtractionClaim],
) -> dict[str, Any]:
    ctx.deps.submitted = ExtractionOutput(claims=claims)
    raise SubmitTerminated()


def build_extraction_agent(
    *, model_id: str, system_prompt: str = EXTRACTION_SYSTEM_PROMPT
) -> Agent:
    """Build the extraction agent.

    The default ``system_prompt`` is the no-hallucinate, document-grounded
    prompt for the headline ``pcfbench_extraction`` eval.  Pass
    ``EXTRACTION_QUERY_ONLY_SYSTEM_PROMPT`` for the
    ``pcfbench_extraction_query_only_estimate`` ablation.
    """
    submit_tool = PydanticAITool(
        _submit_extraction,
        name="submit_extraction",
        description="Submit all extracted numerical claims.",
    )
    return build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=system_prompt,
        tools=[submit_tool],
        deps_type=_Deps,
    )


def _build_user_prompt(inp: ExtractionInput, *, include_document: bool) -> str:
    body = f"Query: {inp.query}"
    if include_document:
        body += f"\n\nDocument text:\n{inp.document_text}"
    return body


async def run_extraction(
    *,
    agent: Agent,
    inp: ExtractionInput,
    include_document: bool = True,
) -> ExtractionOutput | None:
    """Run the extraction agent.

    ``include_document=False`` drops the document text from the user
    prompt -- the model only sees the query, used by the
    ``pcfbench_extraction_query_only_estimate`` ablation.
    """
    deps = _Deps()
    await run_singleshot(
        agent=agent,
        user_prompt=_build_user_prompt(inp, include_document=include_document),
        deps=deps,
        output_recovered=lambda d: d.submitted is not None,
    )
    return deps.submitted
