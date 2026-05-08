"""Bottom-up compositional Task 7 pipeline (decompose → triage → map →
rate-estimate → sum) built on the pcfbench agents.

This is the publishable counterpart of the legacy ``analysis.baselines.
stepwise_agent`` PSWAgent pipeline. The stages here use the same
Pydantic AI agents as the rest of pcfbench (decomposition,
triage, mapping, plus a small in-module rate-estimate agent), so the
"whole vs. sum of parts" comparison against monolithic Task 7 isn't
confounded by harness differences.

The pipeline stays harness-clean by delegating EF lookup to a pluggable
``ef_resolver`` callable. Plug in your own ecoinvent-licensed lookup.
Energy EFs are tiny
public constants from ecoinvent v3.10.

Recursion is bounded by ``max_decomp_depth`` (default 2 = top-level
plus one re-decompose). ``max_triage_depth`` independently bounds the
deepest layer at which the triage step is invoked: setting to 1
triages only the top-level components and sends sub-components from
one re-decompose straight to mapping (skipping the wasted depth-1
triage call that almost always ends in mapping anyway).

Per-item provenance: ``StepwiseResult.max_depth_reached`` records the
deepest recursion layer the pipeline actually touched, so we can
quantify how often models recurse vs. stay top-level.
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, Any, Callable, Literal

import pydantic as pyd
from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai import Tool as PydanticAITool
from pydantic_ai.usage import Usage

from pcfbench.agents._task_agent import (
    AGENT_SDK_PREFIX,
    AgentSDKSingleshotAgent,
    PydanticAISingleshotAgent,
    TaskAgent,
)
from pcfbench.agents.decomposition import (
    DecompositionInput,
    build_decomposition_agent,
    run_decomposition,
)
from pcfbench.agents.epd import EPDInput
from pcfbench.agents.mapping import (
    MappingInput,
    build_mapping_agent_singleshot,
    run_mapping_singleshot,
)
from pcfbench.agents.triage import (
    TriageInput,
    TriageMarket,
    TriageMaterialContext,
    build_triage_agent_singleshot,
    run_triage_singleshot,
)
from pcfbench.models.factory import build_agent
from pcfbench.models.pricing import cost_usd
from pcfbench.tools.ecoinvent_tools import (
    SubmitTerminated,
)
from pcfbench.tools.material_library import (
    MaterialLibrary,
)

# ---------------------------------------------------------------------------
# Hardcoded energy EFs. Sourced from ecoinvent v3.10:
#   - electricity: market group for electricity, low voltage, GLO
#   - natural gas: heat production, natural gas, at industrial furnace >100kW, RoW
# These values are public-facing constants we can ship with pcfbench
# without redistributing the licensed ecoinvent dataset.
# ---------------------------------------------------------------------------
EF_ELECTRICITY_KGCO2E_PER_KWH = 0.485
EF_NATURAL_GAS_KGCO2E_PER_MJ = 0.0671


# ---------------------------------------------------------------------------
# Output / breakdown schemas. Names match the legacy StepwiseResult so
# stepwise_scoring's invariants (mass_fraction_sum, ghost_components,
# zero_ef_components, etc.) work without changes.
# ---------------------------------------------------------------------------


class ComponentContribution(pyd.BaseModel):
    kind: Literal["material", "energy"]
    name: str
    reference_product: str | None = None
    rate_value: float
    rate_unit: str
    ef_kgco2e_per_unit: float
    contribution_kgco2e_per_kg_product: float
    depth: int = 0


class StageRecord(pyd.BaseModel):
    stage: str
    success: bool
    detail: dict[str, Any]


class StepwiseResult(pyd.BaseModel):
    kgco2e_predicted: float | None
    success: bool
    model_name: str
    max_depth_reached: int
    n_components_at_depth: dict[int, int]
    stages: list[StageRecord]
    breakdown: list[ComponentContribution]
    notes: list[str]
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    cost_usd: float | None = None


# ---------------------------------------------------------------------------
# Rate-estimate agent — query-only extraction for "kg of X per kg of
# product Y" / "kWh per kg" / "MJ per kg" style asks. Stays in this
# module rather than agents/extraction.py because the task differs
# (no document, model is asked for a best-guess prior) and we don't
# want to muddy the headline extraction agent with a query-only mode.
# ---------------------------------------------------------------------------


_RATE_ESTIMATE_SYSTEM_PROMPT = """\
You are an LCA expert estimating typical input rates for a product. \
Given a question like "kg of X per kg of product Y" or "kWh per kg of \
product Y", give your single best-guess numerical value with the unit \
exactly as the question phrased it.

If the input is plausibly zero (e.g. a finished product that doesn't \
consume that input), submit value 0. Otherwise give a positive value \
based on training-data priors. Always submit exactly one claim. The \
unit field must be JUST the unit symbol (e.g. ``kg/kg``, ``kWh/kg``, \
``MJ/kg``)."""


class RateEstimateClaim(pyd.BaseModel):
    value: float
    unit: Annotated[
        str,
        Field(
            description=(
                "Just the unit symbol (e.g. 'kg/kg', 'kWh/kg', 'MJ/kg'). "
                "No descriptive prose."
            )
        ),
    ]


@dataclasses.dataclass
class _RateDeps:
    submitted: RateEstimateClaim | None = None


async def _submit_rate_estimate(
    ctx: RunContext[_RateDeps],
    value: float,
    unit: Annotated[
        str,
        Field(description="Unit symbol — no descriptive prose."),
    ],
) -> dict:
    ctx.deps.submitted = RateEstimateClaim(value=value, unit=unit)
    raise SubmitTerminated()


_RATE_SUBMIT_TOOL_NAME = "submit_rate_estimate"
_RATE_SUBMIT_TOOL_DESCRIPTION = "Submit a single best-guess input rate."


def build_rate_estimate_agent(*, model_id: str) -> TaskAgent:
    if model_id.startswith(AGENT_SDK_PREFIX):
        return AgentSDKSingleshotAgent(
            model_id=model_id,
            system_prompt=_RATE_ESTIMATE_SYSTEM_PROMPT,
            output_type=RateEstimateClaim,
            submit_tool_name=_RATE_SUBMIT_TOOL_NAME,
            submit_tool_description=_RATE_SUBMIT_TOOL_DESCRIPTION,
        )
    submit_tool = PydanticAITool(
        _submit_rate_estimate,
        name=_RATE_SUBMIT_TOOL_NAME,
        description=_RATE_SUBMIT_TOOL_DESCRIPTION,
    )
    agent = build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=_RATE_ESTIMATE_SYSTEM_PROMPT,
        tools=[submit_tool],
        deps_type=_RateDeps,
    )
    return PydanticAISingleshotAgent(
        agent=agent,
        make_deps=_RateDeps,
        get_output=lambda d: d.submitted,
        output_recovered=lambda d: d.submitted is not None,
    )


async def _run_rate_estimate(
    *, agent: TaskAgent, query: str, usage: Usage | None = None
) -> RateEstimateClaim | None:
    return await agent.run_task(query, usage=usage)


# ---------------------------------------------------------------------------
# Unit normalisation for the rate-estimate output. Mirrors the
# legacy stepwise's _to_kg_per_kg / _to_kwh_per_kg / _to_mj_per_kg.
# ---------------------------------------------------------------------------


_KG_PER_KG_ALIASES = frozenset(
    {"kg/kg", "kg per kg", "kg/kg product", "kg / kg", "kgkg"}
)


def _to_kg_per_kg(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u in _KG_PER_KG_ALIASES:
        return value
    if u in {"%", "percent", "mass%", "mass percent", "wt%", "weight%"}:
        return value / 100.0
    if u in {"g/kg", "grams per kg"}:
        return value / 1000.0
    if u in {"mg/kg"}:
        return value / 1_000_000.0
    return None


def _to_kwh_per_kg(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u in {"kwh/kg", "kwh per kg"}:
        return value
    if u in {"wh/kg"}:
        return value / 1000.0
    if u in {"mj/kg"}:
        return value * 0.2778  # 1 MJ = 0.2778 kWh
    return None


def _to_mj_per_kg(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u in {"mj/kg", "mj per kg"}:
        return value
    if u in {"kj/kg"}:
        return value / 1000.0
    if u in {"kwh/kg"}:
        return value * 3.6
    return None


# ---------------------------------------------------------------------------
# Pipeline config + EF resolver type.
# ---------------------------------------------------------------------------

# Resolver: takes a picklist reference-product / activity name, returns
# kgCO2e/kg, or None if the resolver doesn't have an EF for it. Plug in
# your own ecoinvent-licensed lookup.
EFResolver = Callable[[str], float | None]


@dataclasses.dataclass
class StepwiseConfig:
    model_id: str
    library: MaterialLibrary
    ef_resolver: EFResolver
    max_decomp_depth: int = 2
    max_triage_depth: int | None = None
    # Precompiled agent handles. If None, ``build_pipeline_agents`` does
    # the construction. Reusing built agents across many EPDs is cheap
    # and avoids re-running the model factory.
    decomp_agent: TaskAgent | None = None
    triage_agent: TaskAgent | None = None
    mapping_agent: TaskAgent | None = None
    rate_agent: TaskAgent | None = None


def build_pipeline_agents(*, model_id: str) -> dict[str, TaskAgent]:
    """Pre-build the four agent handles a stepwise run uses. Costs one
    model-factory call each, then reused across all EPDs in a sweep cell."""
    return {
        "decomp": build_decomposition_agent(model_id=model_id),
        "triage": build_triage_agent_singleshot(model_id=model_id),
        "mapping": build_mapping_agent_singleshot(model_id=model_id),
        "rate": build_rate_estimate_agent(model_id=model_id),
    }


# ---------------------------------------------------------------------------
# The pipeline.
# ---------------------------------------------------------------------------


async def _decompose_step(
    *,
    config: StepwiseConfig,
    product_name: str,
    description: str,
    quantity_unit: str,
    usage: Usage | None = None,
) -> tuple[list[str], StageRecord]:
    agent = config.decomp_agent or build_decomposition_agent(model_id=config.model_id)
    inp = DecompositionInput(
        product_name=product_name,
        description=description,
        quantity_unit=quantity_unit,
    )
    out = await run_decomposition(agent=agent, inp=inp, usage=usage)
    components = out.components if out is not None else []
    return components, StageRecord(
        stage=f"decompose:{product_name[:40]}",
        success=out is not None,
        detail={"components": components},
    )


async def _triage_step(
    *,
    config: StepwiseConfig,
    component: str,
    product_name: str,
    description: str,
    picklist_names: list[str],
    usage: Usage | None = None,
) -> tuple[bool | None, StageRecord]:
    agent = config.triage_agent or build_triage_agent_singleshot(
        model_id=config.model_id
    )
    inp = TriageInput(
        market=TriageMarket(
            name=component,
            description=f"input flowing into the production of {product_name}",
        ),
        material_context_data=TriageMaterialContext(
            name=product_name,
            description=description,
            material_name=product_name,
        ),
    )
    out = await run_triage_singleshot(
        agent=agent, inp=inp, picklist_names=picklist_names, usage=usage
    )
    should_map = out.should_map if out is not None else None
    return should_map, StageRecord(
        stage=f"triage:{component[:40]}",
        success=out is not None,
        detail={"should_map": should_map, "component": component},
    )


async def _map_step(
    *,
    config: StepwiseConfig,
    component: str,
    picklist_names: list[str],
    usage: Usage | None = None,
) -> tuple[str | None, StageRecord]:
    agent = config.mapping_agent or build_mapping_agent_singleshot(
        model_id=config.model_id
    )
    inp = MappingInput(material_name=component)
    out = await run_mapping_singleshot(
        agent=agent,
        inp=inp,
        picklist_names=picklist_names,
        library=config.library,
        usage=usage,
    )
    ref_product = out.reference_product if out is not None else None
    return ref_product, StageRecord(
        stage=f"map:{component[:40]}",
        success=out is not None,
        detail={"component": component, "reference_product": ref_product},
    )


async def _rate_step(
    *,
    config: StepwiseConfig,
    query: str,
    label: str,
    usage: Usage | None = None,
) -> tuple[RateEstimateClaim | None, StageRecord]:
    agent = config.rate_agent or build_rate_estimate_agent(model_id=config.model_id)
    out = await _run_rate_estimate(agent=agent, query=query, usage=usage)
    return out, StageRecord(
        stage=f"rate:{label}",
        success=out is not None,
        detail={
            "query": query,
            "value": out.value if out else None,
            "unit": out.unit if out else None,
        },
    )


async def _estimate_energy(
    *, config: StepwiseConfig, product_name: str, usage: Usage | None = None
) -> tuple[float, float, list[StageRecord]]:
    elec_query = (
        f"How many kWh of electricity per kg of finished {product_name}? "
        f"Submit a single value with unit kWh/kg. If electricity is not "
        f"a meaningful input (e.g. a raw agricultural product), submit 0."
    )
    gas_query = (
        f"How many MJ of natural gas combustion per kg of finished "
        f"{product_name}? Submit a single value with unit MJ/kg. If "
        f"natural gas is not a meaningful input, submit 0."
    )
    elec_claim, elec_stage = await _rate_step(
        config=config, query=elec_query, label="electricity", usage=usage
    )
    gas_claim, gas_stage = await _rate_step(
        config=config, query=gas_query, label="natural_gas", usage=usage
    )

    elec_kwh = 0.0
    if elec_claim is not None:
        v = _to_kwh_per_kg(elec_claim.value, elec_claim.unit)
        if v is not None:
            elec_kwh = v
    gas_mj = 0.0
    if gas_claim is not None:
        v = _to_mj_per_kg(gas_claim.value, gas_claim.unit)
        if v is not None:
            gas_mj = v
    return elec_kwh, gas_mj, [elec_stage, gas_stage]


async def run_stepwise_compositional(
    epd: EPDInput, *, config: StepwiseConfig
) -> StepwiseResult:
    """Run the stepwise pipeline on one EPD using pcfbench agents.

    Note on input scope: this pipeline reads only ``epd.product_name``,
    ``epd.description``, and ``epd.quantity_unit``.  ``epd.composition``
    is deliberately ignored — the pipeline must re-derive the BOM via
    the decomposition agent so the comparison against the single-shot
    ``+description`` disclosure setting is apples-to-apples.  The eval
    name ``pcfbench_stepwise_epd_with_composition`` (and its
    ``has_composition`` item tag) refers to the dataset filter — run
    only on items whose ground truth provides composition so the
    stepwise output can be paired against the single-shot
    ``+composition`` baseline — not to what the pipeline itself sees.
    """
    stages: list[StageRecord] = []
    breakdown: list[ComponentContribution] = []
    notes: list[str] = []
    n_at_depth: dict[int, int] = {}
    max_depth = 0
    usage = Usage()

    picklist_names = [m.reference_product_name for m in config.library.materials]

    # Top-level decomposition.
    components, decomp_stage = await _decompose_step(
        config=config,
        product_name=epd.product_name,
        description=epd.description or "",
        quantity_unit=epd.quantity_unit or "1 kg",
        usage=usage,
    )
    stages.append(decomp_stage)
    if not components:
        notes.append("top-level decomposition failed")
        return StepwiseResult(
            kgco2e_predicted=None,
            success=False,
            model_name=config.model_id,
            max_depth_reached=0,
            n_components_at_depth={},
            stages=stages,
            breakdown=breakdown,
            notes=notes,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            requests=usage.requests,
            cost_usd=cost_usd(config.model_id, usage),
        )

    # Triage + bounded recursion. Leaves get mapped + rate-estimated.
    leaves: list[tuple[str, int]] = []
    queue: list[tuple[str, int]] = [(c, 0) for c in components]
    while queue:
        comp, depth = queue.pop(0)
        n_at_depth[depth] = n_at_depth.get(depth, 0) + 1
        max_depth = max(max_depth, depth)

        # Skip triage entirely past the configured depth — component
        # becomes a leaf and goes straight to mapping.
        skip_triage = (
            config.max_triage_depth is not None and depth >= config.max_triage_depth
        )
        if skip_triage:
            stages.append(
                StageRecord(
                    stage=f"triage_skipped:{comp[:40]}",
                    success=True,
                    detail={
                        "reason": "depth >= max_triage_depth",
                        "component": comp,
                        "depth": depth,
                    },
                )
            )
            leaves.append((comp, depth))
            continue

        should_map, triage_stage = await _triage_step(
            config=config,
            component=comp,
            product_name=epd.product_name,
            description=epd.description or "",
            picklist_names=picklist_names,
            usage=usage,
        )
        stages.append(triage_stage)
        if should_map is None:
            notes.append(f"triage failed for {comp!r}; dropped")
            continue
        if should_map:
            leaves.append((comp, depth))
            continue
        if depth >= config.max_decomp_depth - 1:
            notes.append(
                f"triage said decompose-further for {comp!r} but max depth "
                f"reached; forcing map at this level"
            )
            leaves.append((comp, depth))
            continue
        sub_components, sub_stage = await _decompose_step(
            config=config,
            product_name=comp,
            description="",
            quantity_unit="1 kg",
            usage=usage,
        )
        stages.append(sub_stage)
        if sub_components:
            queue.extend((sc, depth + 1) for sc in sub_components)

    # Map leaves to ecoinvent + extract per-component rates.
    total_kgco2e = 0.0
    for comp, depth in leaves:
        ref_product, map_stage = await _map_step(
            config=config, component=comp, picklist_names=picklist_names, usage=usage
        )
        stages.append(map_stage)
        if not ref_product:
            notes.append(f"mapping failed for {comp!r}; dropped")
            continue
        ef = config.ef_resolver(ref_product)
        if ef is None:
            ef = 0.0
            notes.append(f"no EF for {ref_product!r}; using 0")

        rate_query = (
            f"How many kg of {comp} per kg of finished {epd.product_name}? "
            f"Submit a single value with unit kg/kg. If {comp} is not "
            f"actually an input to {epd.product_name}, submit 0."
        )
        rate_claim, rate_stage = await _rate_step(
            config=config, query=rate_query, label=comp[:40], usage=usage
        )
        stages.append(rate_stage)
        if rate_claim is None:
            rate_value = 0.0
            rate_unit = "kg/kg"
        else:
            rate_unit = rate_claim.unit
            normalised = _to_kg_per_kg(rate_claim.value, rate_unit)
            if normalised is None:
                notes.append(
                    f"unrecognised rate unit {rate_unit!r} for {comp!r}; "
                    f"using value as-is"
                )
                rate_value = rate_claim.value
            else:
                rate_value = normalised

        contribution = rate_value * ef
        total_kgco2e += contribution
        breakdown.append(
            ComponentContribution(
                kind="material",
                name=comp,
                reference_product=ref_product,
                rate_value=rate_value,
                rate_unit=rate_unit,
                ef_kgco2e_per_unit=ef,
                contribution_kgco2e_per_kg_product=contribution,
                depth=depth,
            )
        )

    # Energy carriers (parallel sibling to material decomposition).
    elec_kwh, gas_mj, energy_stages = await _estimate_energy(
        config=config, product_name=epd.product_name, usage=usage
    )
    stages.extend(energy_stages)
    elec_contribution = elec_kwh * EF_ELECTRICITY_KGCO2E_PER_KWH
    gas_contribution = gas_mj * EF_NATURAL_GAS_KGCO2E_PER_MJ
    total_kgco2e += elec_contribution + gas_contribution
    breakdown.append(
        ComponentContribution(
            kind="energy",
            name="electricity",
            rate_value=elec_kwh,
            rate_unit="kWh/kg",
            ef_kgco2e_per_unit=EF_ELECTRICITY_KGCO2E_PER_KWH,
            contribution_kgco2e_per_kg_product=elec_contribution,
            depth=0,
        )
    )
    breakdown.append(
        ComponentContribution(
            kind="energy",
            name="natural_gas",
            rate_value=gas_mj,
            rate_unit="MJ/kg",
            ef_kgco2e_per_unit=EF_NATURAL_GAS_KGCO2E_PER_MJ,
            contribution_kgco2e_per_kg_product=gas_contribution,
            depth=0,
        )
    )

    return StepwiseResult(
        kgco2e_predicted=total_kgco2e,
        success=True,
        model_name=config.model_id,
        max_depth_reached=max_depth,
        n_components_at_depth=n_at_depth,
        stages=stages,
        breakdown=breakdown,
        notes=notes,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        requests=usage.requests,
        cost_usd=cost_usd(config.model_id, usage),
    )
