"""Picklist material library — activity-name + vector search over the
ecoinvent activities in the PCFBench picklist.

Self-contained Gemini-embedding-backed implementation. The picklist
data ships as JSONL; the embeddings ship as a .npy file built by
``pcfbench.picklist.embed_gemini``.

Each picklist row carries exactly three fields, taken from the public
ecoinvent v3.11 "Database Overview" workbook:
``activity_uuid_product_uuid`` / ``activity_name`` / ``product_information``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types as genai_types

from pcfbench.models.vertex_auth import (
    vertex_project_id,
)
from pcfbench.tools.picklist_material import (
    PicklistMaterial,
)

_PICKLIST_DIR = Path(__file__).resolve().parent.parent / "picklist"
_PICKLIST_JSONL = _PICKLIST_DIR / "ecoinvent_picklist.jsonl"

# Gemini-001 is the only embedding backend; matches the pinned baselines.
_GEMINI_EMBEDDINGS_NPY = _PICKLIST_DIR / "embeddings_gemini.npy"
_GEMINI_EMBEDDING_MODEL_TXT = _PICKLIST_DIR / "embedding_model_gemini.txt"
_GEMINI_EMBEDDINGS_UUIDS = _PICKLIST_DIR / "embeddings_gemini_uuids.json"


def _embedding_paths() -> tuple[Path, Path, Path]:
    """Return the (npy, model.txt, uuids.json) triple for the Gemini
    embedding backend."""
    return (
        _GEMINI_EMBEDDINGS_NPY,
        _GEMINI_EMBEDDING_MODEL_TXT,
        _GEMINI_EMBEDDINGS_UUIDS,
    )


def _load_embedding_model_name() -> str:
    _, model_txt, _ = _embedding_paths()
    if model_txt.exists():
        return model_txt.read_text().strip()
    return "gemini-embedding-001"


@cache
def _embedder() -> "_GeminiEmbedder":
    return _GeminiEmbedder(_load_embedding_model_name())


_GEMINI_OUTPUT_DIMENSIONALITY = 768
_GEMINI_LOCATION = "us-central1"


class _GeminiEmbedder:
    """Wrap ``google.genai`` Vertex client to match the embedding the
    PCFBench picklist embeddings were built with: model
    ``gemini-embedding-001`` at output_dimensionality=768, newlines
    stripped from input. Output is L2-normalised so cosine == dot."""

    def __init__(self, name: str) -> None:
        self._model = name  # "gemini-embedding-001"
        self._config = genai_types.EmbedContentConfig(
            output_dimensionality=_GEMINI_OUTPUT_DIMENSIONALITY
        )
        self._client = genai.Client(
            vertexai=True,
            project=vertex_project_id(),
            location=_GEMINI_LOCATION,
        )

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        cleaned = text.replace("\n", " ")
        response = self._client.models.embed_content(
            model=self._model,
            contents=cleaned,
            config=self._config,
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini returned an empty embedding")
        arr = np.asarray(response.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr


def encode_query(text: str) -> np.ndarray[Any, Any]:
    """Embed a single query string with the active embedding model."""
    return _embedder().embed(text)


class MaterialLibrary:
    """In-memory picklist library: activity-name keyword substring +
    cosine vector search."""

    __slots__ = ("_materials", "_index_by_id", "_embeddings", "_ids_in_order")

    def __init__(
        self,
        *,
        materials: list[PicklistMaterial],
        embeddings: np.ndarray[Any, Any] | None,
        ids_in_order: list[str],
    ) -> None:
        if embeddings is not None and embeddings.shape[0] != len(ids_in_order):
            raise ValueError(
                "embeddings rows must match ids_in_order length: "
                f"{embeddings.shape[0]} vs {len(ids_in_order)}"
            )
        self._materials = {m.activity_uuid_product_uuid: m for m in materials}
        self._index_by_id = {mid: i for i, mid in enumerate(ids_in_order)}
        self._embeddings = (
            None if embeddings is None else embeddings.astype(np.float32, copy=False)
        )
        self._ids_in_order = ids_in_order

    @classmethod
    def load_default(cls) -> "MaterialLibrary":
        """Load picklist + embeddings from the package's bundled assets.

        Embeddings are optional. Vector search needs them, but the
        single-shot evals only need the picklist itself (names inlined
        in the prompt) plus keyword lookup, and the embeddings ``.npy``
        is a large generated artifact that is not distributed. When it
        is absent the library still loads and ``search_vector`` is the
        only thing that fails, with an actionable message.

        When embeddings are present, cross-checks the row-order UUID
        sidecar (written by the embed scripts) against the current
        ``ecoinvent_picklist.jsonl`` order so a stale ``embeddings.npy``
        can never silently misalign with a freshly regenerated picklist.
        """
        with open(_PICKLIST_JSONL) as f:
            raw_materials = [json.loads(line) for line in f if line.strip()]
        materials = [PicklistMaterial.model_validate(m) for m in raw_materials]
        picklist_uuids = [m.activity_uuid_product_uuid for m in materials]

        embeddings_path, _, uuids_path = _embedding_paths()
        if not embeddings_path.exists():
            return cls(
                materials=materials,
                embeddings=None,
                ids_in_order=picklist_uuids,
            )
        embeddings = np.load(embeddings_path)

        if not uuids_path.exists():
            raise FileNotFoundError(
                f"Embedding row-order UUID sidecar not found at {uuids_path}. "
                "Re-run pcfbench.picklist.embed_gemini after regenerating "
                "ecoinvent_picklist.jsonl."
            )
        sidecar_uuids = json.loads(uuids_path.read_text())
        if sidecar_uuids != picklist_uuids:
            raise RuntimeError(
                "Picklist row order does not match the embedding sidecar at "
                f"{uuids_path}. ecoinvent_picklist.jsonl has "
                f"{len(picklist_uuids)} rows, sidecar has {len(sidecar_uuids)}. "
                "Re-run the embed script to rebuild the .npy against the "
                "current ecoinvent_picklist.jsonl."
            )

        return cls(
            materials=materials,
            embeddings=embeddings,
            ids_in_order=picklist_uuids,
        )

    @property
    def materials(self) -> list[PicklistMaterial]:
        return list(self._materials.values())

    def get_material_by_activity_name(self, name: str) -> PicklistMaterial | None:
        for material in self._materials.values():
            if material.activity_name == name:
                return material
        return None

    def get_materials_by_activity_name(self, name: str) -> list[PicklistMaterial]:
        return [m for m in self._materials.values() if m.activity_name == name]

    def get_materials_by_reference_product_name(
        self, name: str
    ) -> list[PicklistMaterial]:
        """Lookup by ``reference_product_name``. Models see this form in
        search/inspect outputs and submit lookups in the same form."""
        return [m for m in self._materials.values() if m.reference_product_name == name]

    def search_keyword(
        self, query: str, *, max_results: int = 10
    ) -> list[PicklistMaterial]:
        """Substring match on ``reference_product_name`` (case-insensitive).
        The model sees this form in every other surface, so keyword
        queries are evaluated against the same form."""
        q = query.lower()
        results: list[PicklistMaterial] = []
        for material in self._materials.values():
            if q in material.reference_product_name.lower():
                results.append(material)
                if len(results) >= max_results:
                    break
        return results

    def search_vector(self, query: str, *, k: int = 10) -> list[PicklistMaterial]:
        """Cosine top-k by embedding the query and ranking against the
        precomputed picklist embeddings (which are L2-normalised)."""
        if self._embeddings is None:
            embeddings_path, _, _ = _embedding_paths()
            raise RuntimeError(
                f"Vector search needs picklist embeddings at {embeddings_path}, "
                "which are not present. Build them with "
                "`python -m pcfbench.picklist.embed_gemini` (requires Vertex "
                "credentials). Single-shot evals do not need them."
            )
        query_vec = encode_query(query)
        # cosine == dot when both sides are L2-normalised
        scores = self._embeddings @ query_vec
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            top_unsorted = np.argpartition(-scores, k)[:k]
            order = top_unsorted[np.argsort(-scores[top_unsorted])]
        out: list[PicklistMaterial] = []
        for idx in order:
            mid = self._ids_in_order[int(idx)]
            material = self._materials.get(mid)
            if material is not None:
                out.append(material)
        return out

    def search_combined(
        self,
        *,
        keyword_queries: list[str],
        vector_queries: list[str],
        max_keyword_search_results: int = 10,
        max_vector_search_results: int = 10,
    ) -> list[PicklistMaterial]:
        """Union of keyword + vector hits, deduped by
        ``activity_uuid_product_uuid``, preserving the insertion order so
        the agent sees keyword hits first."""
        unique: dict[str, PicklistMaterial] = {}
        for query in keyword_queries:
            for material in self.search_keyword(
                query, max_results=max_keyword_search_results
            ):
                unique.setdefault(material.activity_uuid_product_uuid, material)
        for query in vector_queries:
            for material in self.search_vector(query, k=max_vector_search_results):
                unique.setdefault(material.activity_uuid_product_uuid, material)
        return list(unique.values())
