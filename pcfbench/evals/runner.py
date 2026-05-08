"""Generic eval runner for PCFBench's 5 headline tasks.

Loads dataset items from local JSONL files (default
``pcfbench_data_external/``), dispatches each to the right agent,
scores per-item, writes a per-run JSONL.

Each eval entry in ``EVALS`` provides:
    - dataset: PCFBench dataset name (mapped to JSONL file(s) via
      ``DATASET_TO_FILES``)
    - input_parser: dict -> typed input
    - agent_factory: model_id -> Agent (or (Agent, Agent) for agentic)
    - run_fn: (agent, inp, **extras) -> typed output (awaitable)
    - extras_fn: () -> dict of extra kwargs (e.g. picklist_names, library)
    - scorer: (output, expected) -> dict[score_name, value | None]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from pcfbench.agents.decomposition import (
    DecompositionInput,
    build_decomposition_agent,
    run_decomposition,
)
from pcfbench.agents.epd import (
    EPDInput,
    build_epd_agent,
    build_epd_agent_with_composition,
    run_epd_name_only,
    run_epd_with_composition,
    run_epd_with_description,
    run_epd_with_region,
)
from pcfbench.agents.extraction import (
    EXTRACTION_QUERY_ONLY_SYSTEM_PROMPT,
    ExtractionInput,
    build_extraction_agent,
    run_extraction,
)
from pcfbench.agents.mapping import (
    MappingInput,
    MappingOutput,
    build_mapping_agent_agentic,
    build_mapping_agent_singleshot,
    run_mapping_agentic,
    run_mapping_singleshot,
)
from pcfbench.agents.parakeet import (
    ParakeetState,
    build_parakeet_agents,
    load_parakeet_state,
    run_parakeet_mapping,
)
from pcfbench.agents.stepwise import (
    StepwiseConfig,
    StepwiseResult,
    build_pipeline_agents,
    run_stepwise_compositional,
)
from pcfbench.agents.triage import (
    TriageInput,
    TriageMarket,
    TriageMaterialContext,
    TriageOutput,
    build_triage_agent_agentic,
    build_triage_agent_agentic_no_context,
    build_triage_agent_singleshot,
    run_triage_agentic,
    run_triage_agentic_no_context,
    run_triage_singleshot,
)
from pcfbench.scoring import (
    calibration as _score_cal,
)
from pcfbench.scoring import (
    decomposition as _score_decomp,
)
from pcfbench.scoring import epd as _score_epd
from pcfbench.scoring import (
    extraction as _score_ext,
)
from pcfbench.scoring import (
    mapping as _score_map,
)
from pcfbench.scoring import (
    stepwise as _score_stepwise,
)
from pcfbench.scoring import (
    triage as _score_triage,
)
from pcfbench.tools.material_library import (
    MaterialLibrary,
)

_RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# --- Per-item scorers (delegate to pcfbench.scoring.*) -----------
# Each takes the agent's typed output + the dataset's ``expected_output`` dict
# and returns a flat dict[score_name -> value | None] suitable for JSONL +
# the ``_summarize`` aggregator below.


async def score_decomposition(out, expected: dict) -> dict:
    """Gemini-judge-aligned precision/recall/F1/Kendall-tau/exact-set.

    Async because the Gemini judge call goes through ``asyncio.to_thread``
    so we don't recursively block the event loop with ``run_sync``."""
    components = out.components if out is not None else None
    return await _score_decomp.score_item_async(components, expected.get("components"))


def score_triage(out, expected: dict) -> dict:
    pred = out.should_map if out is not None else None
    conf = out.confidence if out is not None else None
    return {
        "correct": _score_triage.score_correct(pred, expected.get("should_map")),
        "confidence": conf,
    }


def score_mapping(out, expected: dict) -> dict:
    pred = out.reference_product if out is not None else None
    conf = out.confidence if out is not None else None
    return {
        "exact_match": _score_map.score_exact_match(pred, expected),
        "relevant_substring": _score_map.score_relevant_substring(pred, expected),
        "banned_substring_absent": _score_map.score_banned_substring_absent(
            pred, expected
        ),
        "confidence": conf,
    }


def score_extraction(out, expected: dict) -> dict:
    if out is None:
        pred = None
    else:
        pred = [(c.value, c.unit) for c in (out.claims or [])]
    truth = [(c["value"], c.get("unit") or "") for c in (expected.get("claims") or [])]
    return {
        "claim_matched": _score_ext.score_claim_matched_count(pred, truth),
        "claim_pred_count": _score_ext.score_claim_pred_count(pred),
        "claim_gt_count": _score_ext.score_claim_gt_count(truth),
        "p90_re_on_matched": _score_ext.score_p90_re_on_matched(pred, truth),
        "unit_correctness": _score_ext.score_unit_correctness(pred, truth),
        "best_relative_error": _score_ext.score_best_relative_error(pred, truth),
    }


def score_epd(out, expected: dict) -> dict:
    pred_kg = out.kgco2e if out is not None else None
    truth_kg = expected.get("kgco2e")
    rel_err = _score_epd.score_relative_error(pred_kg, truth_kg)
    return {"rel_err": rel_err}


def score_stepwise(out, expected: dict) -> dict:
    """Wrapper that the eval registry binds to the stepwise EvalSpec.

    Per-item structural-invariant scores only. We deliberately do NOT
    score ``kgco2e_predicted`` against EPD truth here — the shipping
    default ``ef_resolver`` returns ``None`` for every material, so
    aggregate kgCO2e is not a PCF estimate. License-holders who plug
    in a real resolver can grade the kgCO2e column themselves from
    the stored ``output.kgco2e_predicted``."""
    return _score_stepwise.score_stepwise_violations(out, expected)


# --- Stepwise pipeline factory + runner -----------------------------------


def _no_ef_resolver(_reference_product: str) -> float | None:
    """Shipping default: no material-EF lookup. Energy EFs are public
    constants baked into the pipeline, so the structural invariants
    (mass-balance, ghost components, depth, stage success) are still
    populated; only ``kgco2e_predicted`` becomes a partial sum (energy
    only) that this harness does NOT grade. License-holders should
    pass their own resolver via ``StepwiseConfig.ef_resolver`` to
    produce a full kgCO2e estimate."""
    return None


def build_stepwise_config(*, model_id: str) -> StepwiseConfig:
    """Construct a license-free ``StepwiseConfig`` for the given model.

    Loads the public picklist-name material library, pre-builds the
    four sub-agents the pipeline needs, and wires the no-op
    ``_no_ef_resolver`` so the pipeline runs end-to-end without an
    ecoinvent licence."""
    library = MaterialLibrary.load_default()
    agents = build_pipeline_agents(model_id=model_id)
    return StepwiseConfig(
        model_id=model_id,
        library=library,
        ef_resolver=_no_ef_resolver,
        decomp_agent=agents["decomp"],
        triage_agent=agents["triage"],
        mapping_agent=agents["mapping"],
        rate_agent=agents["rate"],
    )


async def run_stepwise(*, agent: StepwiseConfig, inp: EPDInput) -> StepwiseResult:
    """Adapter so the dispatcher's default branch (``run_fn(agent, inp)``)
    can drive the stepwise pipeline. ``agent`` here is the
    ``StepwiseConfig`` bundle returned by ``build_stepwise_config``."""
    return await run_stepwise_compositional(epd=inp, config=agent)


# --- Eval registry ---


@dataclass
class EvalSpec:
    dataset: str
    parse_input: Callable[[dict], Any]
    build_agent: Callable[..., Any]
    run_fn: Callable[..., Any]
    scorer: Callable[[Any, dict], dict]
    needs_picklist: bool = False
    needs_library: bool = False
    is_parakeet: bool = False
    extra_runner_kwargs: dict = None  # type: ignore[assignment]


def _parse_decomposition(inp: dict) -> DecompositionInput:
    return DecompositionInput(
        product_name=inp.get("product_name") or "",
        description=inp.get("description") or "",
        quantity_unit=inp.get("quantity_unit") or "",
    )


def _parse_triage(inp: dict) -> TriageInput:
    market = inp.get("market") or {}
    ctx = inp.get("material_context_data") or {}
    return TriageInput(
        market=TriageMarket(
            name=market.get("name") or "",
            description=market.get("description") or "",
        ),
        material_context_data=TriageMaterialContext(
            name=ctx.get("name"),
            description=ctx.get("description"),
            material_name=ctx.get("material_name") or "",
        ),
    )


def _parse_mapping(inp: dict) -> MappingInput:
    return MappingInput(
        material_name=inp.get("material_name") or "",
        description=inp.get("description"),
        supplier=inp.get("supplier"),
        purchaser_context=inp.get("purchaser_context"),
    )


def _parse_extraction(inp: dict) -> ExtractionInput:
    return ExtractionInput(
        query=inp.get("query") or "",
        document_text=inp.get("document_text") or "",
        source_url=inp.get("source_url"),
    )


def _parse_epd(inp: dict) -> EPDInput:
    return EPDInput(
        product_name=inp.get("product_name") or "",
        description=inp.get("description") or "",
        quantity_unit=inp.get("quantity_unit") or "",
        country_of_origin=inp.get("country_of_origin"),
        composition=inp.get("composition"),
        geography=inp.get("geography"),
        recycled_content=inp.get("recycled_content"),
        source_url=inp.get("source_url"),
    )


EVALS: dict[str, EvalSpec] = {
    "pcfbench_decomposition": EvalSpec(
        dataset="lca_benchmark_paper_decomposition",
        parse_input=_parse_decomposition,
        build_agent=build_decomposition_agent,
        run_fn=run_decomposition,
        scorer=score_decomposition,
    ),
    "pcfbench_triage": EvalSpec(
        dataset="lca_benchmark_paper_triage",
        parse_input=_parse_triage,
        build_agent=build_triage_agent_singleshot,
        run_fn=run_triage_singleshot,
        scorer=score_triage,
        needs_picklist=True,
    ),
    "pcfbench_triage_agentic": EvalSpec(
        dataset="lca_benchmark_paper_triage",
        parse_input=_parse_triage,
        build_agent=build_triage_agent_agentic,
        run_fn=run_triage_agentic,
        scorer=score_triage,
        needs_library=True,
    ),
    # Ablation: same agentic harness, no ROOT MATERIAL CONTEXT in
    # system or user prompt -- isolates the parent-BOM signal in
    # triage decisions.
    "pcfbench_triage_agentic_no_context": EvalSpec(
        dataset="lca_benchmark_paper_triage",
        parse_input=_parse_triage,
        build_agent=build_triage_agent_agentic_no_context,
        run_fn=run_triage_agentic_no_context,
        scorer=score_triage,
        needs_library=True,
    ),
    "pcfbench_mapping_with_context": EvalSpec(
        dataset="lca_benchmark_paper_mapping",
        parse_input=_parse_mapping,
        build_agent=build_mapping_agent_singleshot,
        run_fn=run_mapping_singleshot,
        scorer=score_mapping,
        needs_picklist=True,
        needs_library=True,
    ),
    "pcfbench_mapping_agentic_with_context": EvalSpec(
        dataset="lca_benchmark_paper_mapping",
        parse_input=_parse_mapping,
        build_agent=build_mapping_agent_agentic,
        run_fn=run_mapping_agentic,
        scorer=score_mapping,
        needs_library=True,
    ),
    "pcfbench_mapping_parakeet_with_context": EvalSpec(
        dataset="lca_benchmark_paper_mapping",
        parse_input=_parse_mapping,
        build_agent=build_parakeet_agents,
        run_fn=run_parakeet_mapping,
        scorer=score_mapping,
        is_parakeet=True,
    ),
    "pcfbench_extraction": EvalSpec(
        dataset="lca_benchmark_paper_extraction",
        parse_input=_parse_extraction,
        build_agent=build_extraction_agent,
        run_fn=run_extraction,
        scorer=score_extraction,
    ),
    # Ablation: same dataset/output schema/scorer, but the model sees
    # only the query (no document text) under a system prompt that
    # explicitly invites best-guess values from training-data priors.
    # Establishes an F1 floor that reflects how good the model's
    # priors are when elicited.
    "pcfbench_extraction_query_only_estimate": EvalSpec(
        dataset="lca_benchmark_paper_extraction",
        parse_input=_parse_extraction,
        build_agent=lambda *, model_id: build_extraction_agent(
            model_id=model_id,
            system_prompt=EXTRACTION_QUERY_ONLY_SYSTEM_PROMPT,
        ),
        run_fn=lambda *, agent, inp: run_extraction(
            agent=agent, inp=inp, include_document=False
        ),
        scorer=score_extraction,
    ),
    "pcfbench_epd_with_composition": EvalSpec(
        dataset="lca_benchmark_paper_epd",
        parse_input=_parse_epd,
        build_agent=build_epd_agent_with_composition,
        run_fn=run_epd_with_composition,
        scorer=score_epd,
    ),
    "pcfbench_epd_name_only": EvalSpec(
        dataset="lca_benchmark_paper_epd",
        parse_input=_parse_epd,
        build_agent=lambda *, model_id: build_epd_agent(
            model_id=model_id, disclosure="name_only"
        ),
        run_fn=run_epd_name_only,
        scorer=score_epd,
    ),
    "pcfbench_epd_with_description": EvalSpec(
        dataset="lca_benchmark_paper_epd",
        parse_input=_parse_epd,
        build_agent=lambda *, model_id: build_epd_agent(
            model_id=model_id, disclosure="with_description"
        ),
        run_fn=run_epd_with_description,
        scorer=score_epd,
    ),
    "pcfbench_epd_with_region": EvalSpec(
        dataset="lca_benchmark_paper_epd",
        parse_input=_parse_epd,
        build_agent=lambda *, model_id: build_epd_agent(
            model_id=model_id, disclosure="with_region"
        ),
        run_fn=run_epd_with_region,
        scorer=score_epd,
    ),
    # Bottom-up compositional Task 7. The pipeline (decompose → triage →
    # map → rate-estimate → sum) reads only ``product_name``,
    # ``description``, and ``quantity_unit`` from each EPD — the same
    # field set the single-shot ``pcfbench_epd_with_description``
    # headline T7 baseline sees, so direct-vs-compositional is an
    # apples-to-apples disclosure-matched comparison. The shipping
    # default ``ef_resolver`` is a no-op, so ``kgco2e_predicted`` is
    # NOT a PCF estimate; we score on structural invariants only (mass
    # balance, ghost components, recursion depth, per-stage success).
    # License-holders can plug a real ecoinvent resolver into
    # ``StepwiseConfig.ef_resolver`` to grade kgCO2e themselves.
    "pcfbench_stepwise_with_description": EvalSpec(
        dataset="lca_benchmark_paper_epd_stepwise",
        parse_input=_parse_epd,
        build_agent=build_stepwise_config,
        run_fn=run_stepwise,
        scorer=score_stepwise,
    ),
}


# --- Runner ---


DATASET_TO_FILES: dict[str, list[str]] = {
    "lca_benchmark_paper_decomposition": ["task1_decomposition.jsonl"],
    "lca_benchmark_paper_triage": ["task2_triage.jsonl"],
    "lca_benchmark_paper_mapping": ["task3_mapping.jsonl"],
    "lca_benchmark_paper_extraction": [
        "task4_extraction_material.jsonl",
        "task5_extraction_energy.jsonl",
    ],
    "lca_benchmark_paper_epd": ["task7_epd.jsonl"],
    # Alias: the stepwise eval reads the same EPD JSONL but routes
    # through a different scorer + summary branch.
    "lca_benchmark_paper_epd_stepwise": ["task7_epd.jsonl"],
}


def _load_items(data_dir: Path, dataset: str) -> list:
    rows: list = []
    for fname in DATASET_TO_FILES[dataset]:
        path = data_dir / fname
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rows.append(SimpleNamespace(**json.loads(line)))
    return rows


async def _dispatch(
    spec: EvalSpec,
    *,
    agent: Any,
    submit_only: Any,
    parsed: Any,
    library: MaterialLibrary | None,
    picklist: list[str] | None,
    parakeet_state: ParakeetState | None,
    is_agentic: bool,
):
    """Call the appropriate run_fn given eval signature."""
    if spec.is_parakeet:
        # ``agent`` here is the (paraphrase, rerank) tuple returned by
        # build_parakeet_agents; unpack and pass through.
        paraphrase_agent, rerank_agent = agent
        return await spec.run_fn(
            paraphrase_agent=paraphrase_agent,
            rerank_agent=rerank_agent,
            inp=parsed,
            state=parakeet_state,
        )
    if is_agentic:
        return await spec.run_fn(
            agent=agent,
            submit_only_agent=submit_only,
            inp=parsed,
            library=library,
        )
    if spec.needs_picklist and spec.needs_library:
        return await spec.run_fn(
            agent=agent, inp=parsed, picklist_names=picklist, library=library
        )
    if spec.needs_picklist:
        return await spec.run_fn(agent=agent, inp=parsed, picklist_names=picklist)
    return await spec.run_fn(agent=agent, inp=parsed)


def _is_agentic_eval(eval_name: str) -> bool:
    return "agentic" in eval_name


async def run_eval(
    *,
    eval_name: str,
    model_id: str,
    output_path: Path,
    data_dir: Path,
    limit: int | None = None,
    concurrency: int = 8,
    max_retries: int = 5,
) -> dict:
    spec = EVALS[eval_name]
    is_agentic = _is_agentic_eval(eval_name)

    library = None
    picklist = None
    if spec.needs_library:
        library = MaterialLibrary.load_default()
    if spec.needs_picklist and library is not None:
        picklist = [m.reference_product_name for m in library.materials]
    elif spec.needs_picklist:
        library = MaterialLibrary.load_default()
        picklist = [m.reference_product_name for m in library.materials]

    parakeet_state: ParakeetState | None = None
    if spec.is_parakeet:
        # Loads picklist + GTE embeddings + the GTE-Large model once.
        # Heavy (~700 MB model + 30s warmup if cold) so it's deliberately
        # outside the per-item loop.
        parakeet_state = load_parakeet_state()

    if spec.is_parakeet:
        # build_parakeet_agents returns (paraphrase, rerank); the
        # dispatcher unpacks the pair, just like the agentic branch.
        agent = spec.build_agent(model_id=model_id)
        submit_only = None
    elif is_agentic:
        agent_pair = spec.build_agent(model_id=model_id)
        agent, submit_only = agent_pair
    else:
        agent = spec.build_agent(model_id=model_id)
        submit_only = None

    raw_items = _load_items(data_dir, spec.dataset)
    if limit is not None:
        raw_items = raw_items[:limit]
    print(
        f"Loaded {len(raw_items)} items from {spec.dataset} for "
        f"{eval_name} ({model_id})",
        flush=True,
    )

    sem = asyncio.Semaphore(concurrency)
    out_rows: list[dict] = []
    completed = 0
    t_start = time.perf_counter()

    async def _one(it):
        nonlocal completed
        async with sem:
            t0 = time.perf_counter()
            inp = it.input or {}
            expected = it.expected_output or {}
            try:
                parsed = spec.parse_input(inp)
            except Exception as e:
                completed += 1
                print(
                    f"  [{completed}/{len(raw_items)}] "
                    f"{it.id[:30]}: PARSE-ERROR {e}",
                    flush=True,
                )
                parse_err_scores = spec.scorer(None, expected)
                if asyncio.iscoroutine(parse_err_scores):
                    parse_err_scores = await parse_err_scores
                out_rows.append(
                    {
                        "item_id": it.id,
                        "input": inp,
                        "expected": expected,
                        "output": None,
                        "scores": parse_err_scores,
                        "elapsed": 0.0,
                        "error": str(e),
                    }
                )
                return

            output = None
            err = None
            attempts = 0
            for attempt in range(max_retries):
                attempts = attempt + 1
                try:
                    result = await _dispatch(
                        spec,
                        agent=agent,
                        submit_only=submit_only,
                        parsed=parsed,
                        library=library,
                        picklist=picklist,
                        parakeet_state=parakeet_state,
                        is_agentic=is_agentic,
                    )
                    if is_agentic:
                        # AgentRunResult wraps deps; pull typed output.
                        deps = result.deps
                        if eval_name == "pcfbench_mapping_agentic_with_context":
                            ref = deps.submitted_reference_product
                            output = (
                                MappingOutput(
                                    reference_product=ref,
                                    confidence=deps.submitted_confidence,
                                )
                                if ref
                                else None
                            )
                        elif eval_name in (
                            "pcfbench_triage_agentic",
                            "pcfbench_triage_agentic_no_context",
                        ):
                            sm = deps.submitted_should_map
                            output = (
                                TriageOutput(
                                    should_map=sm,
                                    confidence=deps.submitted_confidence,
                                )
                                if sm is not None
                                else None
                            )
                    else:
                        output = result
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    output = None
                if output is not None:
                    break

            scores = spec.scorer(output, expected)
            if asyncio.iscoroutine(scores):
                scores = await scores
            out_rows.append(
                {
                    "item_id": it.id,
                    "input": inp,
                    "expected": expected,
                    "output": (
                        output.model_dump() if hasattr(output, "model_dump") else None
                    ),
                    "scores": scores,
                    "elapsed": time.perf_counter() - t0,
                    "attempts": attempts,
                    "error": err,
                }
            )
            completed += 1
            tag = (
                " RETRIED-FAIL"
                if (output is None and err)
                else (f" retried-ok({attempts}x)" if attempts > 1 else "")
            )
            sample_score = next(
                (f"{k}={v}" for k, v in scores.items() if v is not None),
                "no-score",
            )
            print(
                f"  [{completed}/{len(raw_items)}] {it.id[:30]}: "
                f"{sample_score} ({time.perf_counter()-t0:.1f}s){tag}",
                flush=True,
            )

    await asyncio.gather(*(_one(it) for it in raw_items))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, default=str) + "\n")

    summary = _summarize(out_rows, spec)
    summary["elapsed_total_s"] = time.perf_counter() - t_start
    summary["eval_name"] = eval_name
    summary["model_id"] = model_id
    summary["n_items"] = len(out_rows)
    summary["n_with_score"] = sum(
        1 for r in out_rows if any(v is not None for v in r["scores"].values())
    )
    return summary


def _summarize(rows: list[dict], spec: EvalSpec) -> dict:
    """Run-level aggregates. Generic mean for most metrics; per-eval
    overrides apply the headline aggregations the existing harness uses
    (e.g. extraction's aggregate claim-F1 from total counts, EPD's
    within-factor-2/5, decomposition's mean F1 + exact-set-match rate)."""
    summary: dict[str, Any] = {}

    def _vals(name: str):
        return [r["scores"].get(name) for r in rows]

    if spec.dataset == "lca_benchmark_paper_extraction":
        summary["claim_f1"] = _score_ext.run_claim_f1(
            matched_counts=_vals("claim_matched"),
            pred_counts=_vals("claim_pred_count"),
            gt_counts=[v for v in _vals("claim_gt_count") if v is not None],
        )
        summary["p90_re_on_matched"] = _score_ext.run_p90_re_on_matched(
            _vals("p90_re_on_matched")
        )
        summary["mean_unit_correctness"] = _score_ext.run_mean_unit_correctness(
            _vals("unit_correctness")
        )
        summary["median_best_relative_error"] = (
            _score_ext.run_median_best_relative_error(_vals("best_relative_error"))
        )
        return summary

    if spec.dataset == "lca_benchmark_paper_epd":
        rel_errs = _vals("rel_err")
        summary["within_factor_2"] = _score_epd.run_within_factor_2(rel_errs)
        summary["within_factor_5"] = _score_epd.run_within_factor_5(rel_errs)
        summary["median_abs_relative_error"] = _score_epd.run_median_abs_relative_error(
            rel_errs
        )
        summary["mean_relative_error"] = _score_epd.run_mean_relative_error(rel_errs)
        return summary

    if spec.dataset == "lca_benchmark_paper_epd_stepwise":
        return _score_stepwise.summarize_stepwise(rows)

    if spec.dataset == "lca_benchmark_paper_triage":
        # Accuracy = mean(correct).
        correct_vals = [
            r["scores"]["correct"]
            for r in rows
            if r["scores"].get("correct") is not None
        ]
        summary["accuracy"] = (
            sum(1.0 for v in correct_vals if v) / len(correct_vals)
            if correct_vals
            else 0.0
        )
        # Precision/recall/F1 need (pred_should_map, expected_should_map);
        # pred lives in the stored agent output dict.
        pred_pairs: list[tuple[bool | None, bool | None]] = []
        for r in rows:
            output = r.get("output")
            pred = output.get("should_map") if isinstance(output, dict) else None
            exp = r["expected"].get("should_map")
            if pred is None or exp is None:
                continue
            pred_pairs.append((bool(pred), bool(exp)))
        summary["precision"] = _score_triage.run_precision(pred_pairs)
        summary["recall"] = _score_triage.run_recall(pred_pairs)
        summary["f1"] = _score_triage.run_f1(pred_pairs)
        # Calibration on (confidence, correct).
        cal_pairs = list(zip(_vals("confidence"), _vals("correct")))
        summary["mean_confidence"] = _score_cal.run_mean_confidence(cal_pairs)
        summary["brier"] = _score_cal.run_brier(cal_pairs)
        summary["ece"] = _score_cal.run_ece(cal_pairs)
        return summary

    if spec.dataset == "lca_benchmark_paper_mapping":
        summary["mean_exact_match"] = _score_map.run_mean_bool(_vals("exact_match"))
        summary["mean_relevant_substring"] = _score_map.run_mean_bool(
            _vals("relevant_substring")
        )
        summary["mean_banned_substring_absent"] = _score_map.run_mean_bool(
            _vals("banned_substring_absent")
        )
        # Calibration on (confidence, exact_match).
        cal_pairs = list(zip(_vals("confidence"), _vals("exact_match")))
        summary["mean_confidence"] = _score_cal.run_mean_confidence(cal_pairs)
        summary["brier"] = _score_cal.run_brier(cal_pairs)
        summary["ece"] = _score_cal.run_ece(cal_pairs)
        return summary

    if spec.dataset == "lca_benchmark_paper_decomposition":
        summary["mean_precision"] = _score_decomp.run_mean_float(_vals("precision"))
        summary["mean_recall"] = _score_decomp.run_mean_float(_vals("recall"))
        summary["mean_f1"] = _score_decomp.run_mean_float(_vals("f1"))
        summary["mean_kendall_tau"] = _score_decomp.run_mean_float(_vals("kendall_tau"))
        summary["exact_set_match_rate"] = _score_decomp.run_mean_bool(
            _vals("exact_set_match")
        )
        return summary

    # Fallback — generic mean over numeric / bool.
    score_keys = list({k for r in rows for k in r["scores"]})
    for k in score_keys:
        vals = [r["scores"][k] for r in rows if r["scores"][k] is not None]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            summary[f"mean_{k}"] = sum(1.0 if v else 0.0 for v in vals) / len(vals)
        elif isinstance(vals[0], (int, float)):
            summary[f"mean_{k}"] = sum(float(v) for v in vals) / len(vals)
    return summary


_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "pcfbench_data_external"


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_name", choices=list(EVALS.keys()))
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Directory with pcfbench task*.jsonl files.",
    )
    args = parser.parse_args()

    out = (
        args.output
        or _RUNS_DIR
        / f"{args.eval_name}__{args.model.replace('/', '_').replace('@', '_')}.jsonl"
    )
    summary = await run_eval(
        eval_name=args.eval_name,
        model_id=args.model,
        output_path=out,
        data_dir=args.data_dir,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
