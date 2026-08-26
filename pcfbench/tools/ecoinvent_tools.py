"""Pydantic AI tool definitions for the agentic ecoinvent path.

Each tool reads its per-call state from ``RunContext[MappingDeps]`` (or
similar deps types), so PCFBench agents can be built once at import
time and only deps change per dataset item.

Submit tools raise ``_SubmitTerminated`` to short-circuit the
Pydantic AI loop the moment the model commits. The runner in
``pcfbench.agents.runner`` catches this and reads the chosen answer
back from deps.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any

from pydantic import Field
from pydantic_ai import RunContext

from pcfbench.tools.material_library import (
    MaterialLibrary,
)
from pcfbench.tools.search_tracker import (
    SearchMaterialTracker,
)


class SubmitTerminated(Exception):
    """Raised inside submit_* tools to end the agent loop. Pydantic AI
    propagates this out of ``agent.run()``; the runner catches it and
    the caller reads the structured answer from deps."""


@dataclasses.dataclass
class MappingDeps:
    """Per-call state threaded through the agentic mapping loop."""

    tracker: SearchMaterialTracker
    submitted_reference_product: str | None = None
    submitted_confidence: float | None = None


@dataclasses.dataclass
class TriageDeps:
    """Per-call state threaded through the agentic triage loop."""

    tracker: SearchMaterialTracker
    submitted_should_map: bool | None = None
    submitted_confidence: float | None = None


SEARCH_DESCRIPTION = (
    "Search the PCFBench ecoinvent picklist for candidate activities. "
    "Submit one or more keyword queries (substring match on activity "
    "name) and/or vector queries (semantic search) at once. Internal "
    "memory ensures duplicate queries and previously-surfaced results "
    "are not returned again. Results are ecoinvent activity names "
    "only; use inspect_ecoinvent for the product description."
)

INSPECT_DESCRIPTION = (
    "Inspect specific ecoinvent activities in detail. Returns the "
    "activity name and product information for each requested "
    "activity. Use after search_ecoinvent to compare candidates."
)

DEEP_THOUGHT_DESCRIPTION = (
    "Scratchpad to think through the routing decision. Use this "
    "after a search/inspect step to deliberate before submitting."
)


async def search_ecoinvent(
    ctx: RunContext[Any],
    keyword_queries: Annotated[
        list[str],
        Field(
            description=(
                "List of keyword queries to search for activities (substring "
                "match on activity name)."
            )
        ),
    ],
    vector_queries: Annotated[
        list[str],
        Field(
            description=("List of vector queries for semantic search of activities.")
        ),
    ],
    max_keyword_search_results: Annotated[
        int,
        Field(description="Maximum keyword search results to return."),
    ] = 10,
    max_vector_search_results: Annotated[
        int,
        Field(description="Maximum vector search results to return."),
    ] = 10,
) -> dict[str, Any]:
    """search_ecoinvent — exposed as a Pydantic AI tool."""
    tracker: SearchMaterialTracker = ctx.deps.tracker
    return tracker.search_materials(
        keyword_queries=keyword_queries,
        vector_queries=vector_queries,
        max_keyword_search_results=max_keyword_search_results,
        max_vector_search_results=max_vector_search_results,
    )


async def inspect_ecoinvent(
    ctx: RunContext[Any],
    material_names: Annotated[
        list[str],
        Field(
            description=(
                "List of activity names to inspect in detail (must match "
                "the picklist's ``activity_name`` field exactly)."
            )
        ),
    ],
) -> list[dict[str, Any]]:
    """inspect_ecoinvent — exposed as a Pydantic AI tool."""
    tracker: SearchMaterialTracker = ctx.deps.tracker
    return tracker.inspect_materials(material_names)


async def deep_thought(
    ctx: RunContext[Any],  # noqa: ARG001
    thought: str,  # noqa: ARG001
) -> dict[str, Any]:
    """No-op scratchpad. Returns an empty dict so the agent gets a
    response and continues. Mirrors PSWAgent's deep-thought tool."""
    return {}


_CONFIDENCE_FIELD = Field(
    ge=0.0,
    le=1.0,
    description=(
        "Calibrated probability in [0, 1] that the submission is "
        "correct. Be calibrated: across many decisions at confidence "
        "X, you should be right X fraction of the time."
    ),
)


async def submit_mapping(
    ctx: RunContext[MappingDeps],
    reference_product: str,
    confidence: Annotated[float, _CONFIDENCE_FIELD],
) -> dict[str, Any]:
    """Terminating tool: stores the chosen reference product and the
    model's calibrated self-confidence into deps and raises
    ``SubmitTerminated`` to end the agent loop. The runner catches it
    and recovers the structured output from deps. (Anthropic + extended
    thinking + Pydantic AI ``output_type=ToolOutput`` is forbidden by
    the API, so we cannot use Pydantic AI's structured-output
    termination here — see Stage A notes.)

    Adding the confidence ask is neutral-to-slightly-positive for
    accuracy on Haiku per paired-item Stage G testing (3a config), so
    headline runs include it and we get the calibration data for
    free."""
    ctx.deps.submitted_reference_product = reference_product
    ctx.deps.submitted_confidence = confidence
    raise SubmitTerminated()


async def submit_triage(
    ctx: RunContext[TriageDeps],
    should_map: bool,
    confidence: Annotated[float, _CONFIDENCE_FIELD],
) -> dict[str, Any]:
    """Terminating tool for triage. See ``submit_mapping`` re: the
    confidence parameter."""
    ctx.deps.submitted_should_map = should_map
    ctx.deps.submitted_confidence = confidence
    raise SubmitTerminated()


def new_mapping_deps(library: MaterialLibrary) -> MappingDeps:
    return MappingDeps(tracker=SearchMaterialTracker(library))


def new_triage_deps(library: MaterialLibrary) -> TriageDeps:
    return TriageDeps(tracker=SearchMaterialTracker(library))
