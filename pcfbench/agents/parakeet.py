"""Parakeet 3-stage mapping baseline (Balaji et al. 2025).

Pipeline (per material):

    1. Paraphrase  — LLM rewrites the raw material name in plain language.
    2. Retrieve    — GTE-Large cosine top-K over the picklist.
    3. Rerank      — LLM picks the best match from the K candidates.

The picklist + GTE embeddings are loaded once per process and shared
across items via ``ParakeetState``. Both LLM stages run through the same
Pydantic-AI factory ``build_agent`` the rest of pcfbench uses,
so reasoning configs / 429 retries / Vertex auth are inherited for free.

The published Parakeet config uses Claude Sonnet 4.6 as the LLM backbone;
we reuse that as the default model for the headline cell. Any model_id
the factory accepts will work here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pydantic as pyd
from pydantic_ai import Agent

from pcfbench.agents.mapping import (
    MappingInput,
    MappingOutput,
)
from pcfbench.models.factory import build_agent

TOP_K = 10

PARAPHRASE_SYSTEM_PROMPT = """\
Given a raw material or ingredient description from an industrial bill of \
materials, rewrite it as a clear, plain-language description. Remove \
abbreviations, ERP codes, supplier-specific notation, and foreign language. \
Keep the essential meaning including material type, grade, form, and \
processing state. Return only the paraphrased text, nothing else."""

RERANK_SYSTEM_PROMPT_TEMPLATE = """\
You are an LCA practitioner selecting the best ecoinvent process match \
for a material.

Material: {paraphrased_material_name}
Original input: {raw_material_name}

Candidate ecoinvent processes (ranked by semantic similarity):
{candidate_list}

Select the single best matching ecoinvent process. Prefer "market for ..." \
activities over production activities. Consider geography, processing state, \
and material grade. If none of the candidates are a good match, select the \
closest one anyway (do not respond with NO_MATCH)."""


class RerankOutput(pyd.BaseModel):
    """Structured rerank result. Pydantic AI's structured-output mode
    forces a single tool call returning these two fields."""

    selected_activity: str
    justification: str


@dataclass
class ParakeetState:
    """Loaded once per process. Carries the GTE model + picklist
    embeddings + the row-aligned ``reference_product_name`` list so
    retrieval can return both the candidate name and its similarity in
    one pass."""

    gte_model: Any
    picklist_names: list[str]
    embeddings: npt.NDArray[np.float32]


def _picklist_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "picklist"


def load_parakeet_state() -> ParakeetState:
    """Load ecoinvent_picklist.jsonl + embeddings_gte.npy and instantiate
    the GTE model. Embeddings + picklist row order are cross-checked
    via the UUID sidecar to fail loudly if either was rebuilt without
    the other.

    The sentence_transformers import is kept local so importing this
    module doesn't pull a 670 MB model into memory unless someone
    actually constructs the state."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    pdir = _picklist_dir()
    with (pdir / "ecoinvent_picklist.jsonl").open() as f:
        materials = [json.loads(line) for line in f if line.strip()]
    npy_path = pdir / "embeddings_gte.npy"
    uuids_path = pdir / "embeddings_gte_uuids.json"
    model_txt = pdir / "embedding_model_gte.txt"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"GTE embeddings not found at {npy_path}. Build them with "
            "`python -m pcfbench.picklist.embed_gte`."
        )
    embeddings = np.load(npy_path).astype(np.float32)
    if uuids_path.exists():
        uuids = json.loads(uuids_path.read_text())
        current = [m["activity_uuid_product_uuid"] for m in materials]
        if uuids != current:
            raise RuntimeError(
                "embeddings_gte_uuids.json row order does not match "
                "ecoinvent_picklist.jsonl — re-run picklist.embed_gte "
                "after any picklist rebuild."
            )
    if embeddings.shape[0] != len(materials):
        raise RuntimeError(
            f"GTE embeddings rows ({embeddings.shape[0]}) != picklist "
            f"items ({len(materials)})."
        )
    model_id = (
        model_txt.read_text().strip() if model_txt.exists() else "thenlper/gte-large"
    )
    gte_model = SentenceTransformer(model_id)
    picklist_names = [m["reference_product_name"] for m in materials]
    return ParakeetState(
        gte_model=gte_model, picklist_names=picklist_names, embeddings=embeddings
    )


def build_parakeet_agents(*, model_id: str) -> tuple[Agent, Agent]:
    """Return ``(paraphrase_agent, rerank_agent)``.

    Mirrors the agentic mapping convention of returning a tuple so the
    runner can unpack two agents from one factory call. The paraphrase
    agent is plain text (``output_type=str``); the rerank agent is
    structured-output via ``RerankOutput``.
    """
    paraphrase_agent = build_agent(
        model_id=model_id,
        agentic=False,
        output_type=str,
        system_prompt=PARAPHRASE_SYSTEM_PROMPT,
        tools=[],
        deps_type=None,
    )
    rerank_agent = build_agent(
        model_id=model_id,
        agentic=False,
        output_type=RerankOutput,
        system_prompt="",  # filled per-call via the user prompt template
        tools=[],
        deps_type=None,
    )
    return paraphrase_agent, rerank_agent


def _build_paraphrase_input(inp: MappingInput) -> str:
    """Build the paraphrase agent's input text: raw material name, then
    optional description / supplier / purchaser context. Empty fields
    are skipped."""
    text = inp.material_name
    if inp.description:
        text += f"\nDescription: {inp.description}"
    if inp.supplier:
        text += f"\nSupplier: {inp.supplier}"
    if inp.purchaser_context:
        text += f"\nPurchaser context: {json.dumps(inp.purchaser_context)}"
    return text


def _retrieve_top_k(
    state: ParakeetState, query: str, top_k: int
) -> list[tuple[str, float]]:
    """L2-normalised query embedding · L2-normalised picklist embeddings
    is cosine similarity. Returns the top-K (name, similarity) sorted
    descending."""
    query_emb = state.gte_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    query_emb = np.asarray(query_emb, dtype=np.float32)
    sims = (state.embeddings @ query_emb.T).squeeze()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(state.picklist_names[int(i)], float(sims[int(i)])) for i in top_idx]


async def run_parakeet_mapping(
    *,
    paraphrase_agent: Agent,
    rerank_agent: Agent,
    inp: MappingInput,
    state: ParakeetState,
    top_k: int = TOP_K,
) -> MappingOutput | None:
    """Run the 3-stage Parakeet pipeline for one material.

    Returns the selected reference product wrapped in MappingOutput so
    the existing ``score_mapping`` scorer applies as-is. Returns None
    only if every stage fails — individual-stage retries are handled by
    Pydantic AI's tenacity transport on the underlying http_client."""
    paraphrase_input = _build_paraphrase_input(inp)

    paraphrase_result = await paraphrase_agent.run(paraphrase_input)
    paraphrased = (paraphrase_result.output or "").strip()
    if not paraphrased:
        # Stage 1 produced nothing usable; fall through to retrieval on
        # the raw material name (same fallback the legacy harness used
        # implicitly when paraphrase calls failed).
        paraphrased = inp.material_name

    candidates = _retrieve_top_k(state, paraphrased, top_k=top_k)

    candidate_list = "\n".join(
        f"{i + 1}. {name} (similarity: {score:.3f})"
        for i, (name, score) in enumerate(candidates)
    )
    rerank_user_prompt = RERANK_SYSTEM_PROMPT_TEMPLATE.format(
        paraphrased_material_name=paraphrased,
        raw_material_name=inp.material_name,
        candidate_list=candidate_list,
    )
    rerank_result = await rerank_agent.run(rerank_user_prompt)
    rerank_output = cast(RerankOutput, rerank_result.output)
    selected = (rerank_output.selected_activity or "").strip()
    if not selected:
        # Pydantic AI guarantees a typed RerankOutput, but the activity
        # name itself can come back blank from the LLM. Mirror the
        # legacy harness's last-resort: take the highest-similarity
        # candidate.
        selected = candidates[0][0] if candidates else ""
    if not selected:
        return None
    return MappingOutput(reference_product=selected, confidence=None)
