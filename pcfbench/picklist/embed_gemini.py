"""Re-embed the picklist with Gemini-001 via Vertex AI (the embedding
the existing PSWAgent harness used). This is the embedding backend
used by ``MaterialLibrary``.

Outputs:
    pcfbench/picklist/embeddings_gemini.npy
    pcfbench/picklist/embedding_model_gemini.txt
    pcfbench/picklist/embeddings_gemini_uuids.json   # row-order UUID sidecar
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types as genai_types

from pcfbench.models.vertex_auth import (
    vertex_project_id,
)

GEMINI_MODEL = "gemini-embedding-001"
OUTPUT_DIM = 768
LOCATION = "us-central1"

_PICKLIST_DIR = Path(__file__).resolve().parent
_PICKLIST_JSONL = _PICKLIST_DIR / "ecoinvent_picklist.jsonl"
_EMBEDDINGS_NPY = _PICKLIST_DIR / "embeddings_gemini.npy"
_MODEL_TXT = _PICKLIST_DIR / "embedding_model_gemini.txt"
# Sidecar list of activity_uuid_product_uuid in row order, written
# alongside ``embeddings_gemini.npy``. ``MaterialLibrary`` cross-checks
# this against the current ``ecoinvent_picklist.jsonl`` order to fail
# loudly if the two fall out of sync (e.g. picklist regenerated without
# re-embedding).
_EMBEDDINGS_UUIDS_JSON = _PICKLIST_DIR / "embeddings_gemini_uuids.json"


def main() -> None:
    print(f"Loading picklist from {_PICKLIST_JSONL} ...", flush=True)
    with _PICKLIST_JSONL.open() as f:
        materials = [json.loads(line) for line in f if line.strip()]
    texts: list[str] = []
    uuids: list[str] = []
    for material in materials:
        activity_name = material["activity_name"]
        product_information = material.get("product_information", "") or ""
        texts.append(f"{activity_name} {product_information}".strip())
        uuids.append(material["activity_uuid_product_uuid"])
    print(f"Encoding {len(texts)} picklist items with {GEMINI_MODEL} ...", flush=True)

    client = genai.Client(vertexai=True, project=vertex_project_id(), location=LOCATION)
    config = genai_types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM)
    out: list[np.ndarray[Any, Any] | None] = [None] * len(texts)
    t0 = time.time()
    completed = 0

    def _one(idx: int, text: str) -> tuple[int, np.ndarray[Any, Any] | None]:
        cleaned = text.replace("\n", " ")
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=GEMINI_MODEL, contents=cleaned, config=config
                )
                embeddings = response.embeddings or []
                vec = embeddings[0].values
                arr = np.asarray(vec, dtype=np.float32)
                # L2-normalise so cosine == dot
                n = np.linalg.norm(arr)
                if n > 0:
                    arr = arr / n
                return idx, arr
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        return idx, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, i, t) for i, t in enumerate(texts)]
        for fut in as_completed(futs):
            i, vec = fut.result()
            out[i] = vec
            completed += 1
            if completed % 100 == 0:
                print(
                    f"  {completed}/{len(texts)} ({time.time()-t0:.0f}s)",
                    flush=True,
                )

    # All embeds should be present after the retry loop above; assert
    # narrows the type for np.stack and surfaces missing rows loudly.
    assert all(o is not None for o in out), "missing embedding rows"
    embeddings = np.stack([o for o in out if o is not None]).astype(np.float32)
    np.save(_EMBEDDINGS_NPY, embeddings)
    _MODEL_TXT.write_text(GEMINI_MODEL + "\n")
    _EMBEDDINGS_UUIDS_JSON.write_text(json.dumps(uuids))
    print(
        f"Wrote {embeddings.shape} -> {_EMBEDDINGS_NPY}; "
        f"recorded model at {_MODEL_TXT}; "
        f"row-order UUIDs at {_EMBEDDINGS_UUIDS_JSON}",
        flush=True,
    )


if __name__ == "__main__":
    main()
