"""One-shot export of the ecoinvent picklist for the published
``pcfbench`` package.

The picklist is fully driven by the public ecoinvent v3.11 "Database
Overview" workbook (sheet ``Cut-Off AO``). Each output row carries four
fields, taken straight from that sheet:

  * ``activity_uuid_product_uuid`` — ``Activity UUID & Product UUID``
  * ``activity_name``               — ``Activity Name``
  * ``reference_product_name``      — ``Reference Product Name``
  * ``product_information``         — ``Product Information``

``reference_product_name`` is the form the model picks from in our
prompts and tool outputs (e.g. "corrugated board box"); ``activity_name``
is the production process that yields it (e.g. "market for corrugated
board box") and stays in the JSON for cross-referencing the source.

Filtering follows the same shape as
``get_base_ecoinvent_markets_for_direct_mapping`` but with four of its
filters intentionally dropped, so the new picklist is a strict
super-set of the prior ~1,663-row list:

  * keep ``is_market_activity``           (Special Activity Type)
  * keep ``unit == "kg"``                 (Unit column), plus a small
                                           ``ENERGY_ALLOWLIST_UUIDS`` of
                                           non-kg energy markets admitted
                                           by exception (electricity, kWh;
                                           natural gas, m3; industrial
                                           heat, MJ)
  * keep ``filter_to_single_geo_markets`` (one row per reference product,
                                           preferring GLO > RoW > RER)
  * skip ``is_waste_from_ecoinvent``
  * skip ``is_zero_ef``
  * skip ``is_service``
  * skip ``is_removed_by_activity``

The script reads only the public xlsx, so it lives next to the
picklist artifacts it produces inside the package.

Usage:
    uv run python -m pcfbench.picklist.build_picklist_json
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

ECOINVENT_OVERVIEW_URL = (
    "https://support.ecoinvent.org/hubfs/Knowledge%20Base/Database/"
    "Releases/3.11/Database-Overview-for-ecoinvent-v3.11%20(6).xlsx"
)
SHEET_NAME = "Cut-Off AO"
KEY_COL = "Activity UUID & Product UUID"
MARKET_TYPES = {"market activity", "market group"}
# Geographic fallback order: prefer global, then RoW, then RER.
GEO_FALLBACK_ORDER = ("GLO", "RoW", "RER")

# Non-kg energy-carrier markets admitted by exception so Task 2 triage items
# whose subject is electricity / natural gas / industrial heat have a valid
# target in the picklist.
ENERGY_ALLOWLIST_UUIDS = {
    # electricity, low voltage (GLO, kWh)
    "28e25e38-32b2-5696-85d8-0533f5505073_d69294d7-8d64-4915-a896-9996a014c410",
    # natural gas, high pressure (GLO, m3)
    "5506d618-a91d-5c2a-be9a-bef7cb2d4f2c_a9007f10-7e39-4d50-8f4a-d6d03ce3d673",
    # heat, district or industrial, natural gas (GLO, MJ)
    "c43a83b0-06ff-53ca-bd29-09e91182c66a_1125e767-7b5d-442e-81d6-9b0d3e1919ac",
}

_HERE = Path(__file__).resolve()
# Script lives in pcfbench/picklist/ so the picklist JSONL and
# the xlsx cache sit alongside it.
_OUT_DIR = _HERE.parent
_OUT_PICKLIST_JSONL = _OUT_DIR / "ecoinvent_picklist.jsonl"
# The xlsx is large (~20 MB) and regenerable; cached locally next to
# the build script. Covered by the repo-wide ``*.xlsx`` gitignore rule.
_OVERVIEW_CACHE = _OUT_DIR / "ecoinvent_overview_3.11.xlsx"


def _download_overview_xlsx() -> Path:
    if _OVERVIEW_CACHE.exists():
        return _OVERVIEW_CACHE
    _OVERVIEW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ECOINVENT_OVERVIEW_URL} ...", flush=True)
    with urllib.request.urlopen(ECOINVENT_OVERVIEW_URL) as resp:
        data = resp.read()
    _OVERVIEW_CACHE.write_bytes(data)
    print(f"Cached {len(data) / 1e6:.1f} MB to {_OVERVIEW_CACHE}", flush=True)
    return _OVERVIEW_CACHE


def _load_overview_dataframe() -> pd.DataFrame:
    """Load the ``Cut-Off AO`` sheet as a dataframe.

    ``keep_default_na=False`` is set so that ecoinvent's ``NA`` ISO code
    for Namibia is preserved as the string ``"NA"`` rather than being
    converted to NaN by pandas.
    """
    xlsx = _download_overview_xlsx()
    print(f"Reading sheet '{SHEET_NAME}' from {xlsx.name} ...", flush=True)
    df = pd.read_excel(xlsx, sheet_name=SHEET_NAME, keep_default_na=False)
    required = [
        KEY_COL,
        "Activity Name",
        "Geography",
        "Reference Product Name",
        "Product Information",
        "Unit",
        "Special Activity Type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Cut-Off AO sheet is missing expected columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    return df


def _select_candidate_markets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply: market activity + ``unit == 'kg'`` + single-geography per
    reference product (GLO > RoW > RER). All four post-filter cliq
    rules (waste / zero-EF / service / removed-by) are intentionally
    skipped. A small ``ENERGY_ALLOWLIST_UUIDS`` set bypasses the kg filter
    so a single electricity / natural gas / industrial heat market is
    available as a Task 2 mapping target.
    """
    n_total = len(df)
    markets = df[df["Special Activity Type"].isin(MARKET_TYPES)]
    kg_markets = markets[
        (markets["Unit"] == "kg") | (markets[KEY_COL].isin(ENERGY_ALLOWLIST_UUIDS))
    ]
    energy_admitted = markets[markets[KEY_COL].isin(ENERGY_ALLOWLIST_UUIDS)]
    print(
        f"Activities: {n_total} -> markets: {len(markets)} "
        f"-> kg markets: {len(kg_markets)} "
        f"(incl. {len(energy_admitted)} energy-allowlist exceptions)",
        flush=True,
    )

    chosen_idx: list[Any] = []
    dropped_no_fallback = 0
    for _, group in kg_markets.groupby("Reference Product Name", sort=False):
        # First-row-per-geography mapping (drops duplicates if any).
        geo_to_idx = dict(zip(group["Geography"], group.index))
        for preferred in GEO_FALLBACK_ORDER:
            if preferred in geo_to_idx:
                chosen_idx.append(geo_to_idx[preferred])
                break
        else:
            dropped_no_fallback += 1

    single_geo = kg_markets.loc[chosen_idx]
    print(
        f"-> single-geo markets: {len(single_geo)} "
        f"(dropped {dropped_no_fallback} reference products with no "
        f"GLO/RoW/RER fallback)",
        flush=True,
    )
    return single_geo


def main() -> None:
    overview = _load_overview_dataframe()
    candidates = _select_candidate_markets(overview)
    # Sort by activity_uuid_product_uuid so the output ordering is stable
    # across ecoinvent re-releases (UUIDs are permanent even if the xlsx
    # row order changes). Embedding row order in embeddings.npy then has
    # a permanent reference, and the embed scripts cross-check via the
    # ``embeddings_*uuids.json`` sidecar.
    candidates = candidates.sort_values(KEY_COL, kind="stable")

    materials_out: list[dict[str, Any]] = [
        {
            "activity_uuid_product_uuid": str(row[KEY_COL]),
            "activity_name": str(row["Activity Name"]),
            "reference_product_name": str(row["Reference Product Name"]),
            "product_information": str(row["Product Information"]),
        }
        for _, row in candidates.iterrows()
    ]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    with _OUT_PICKLIST_JSONL.open("w") as f:
        for row in materials_out:
            f.write(json.dumps(row) + "\n")
    print(
        f"Wrote {len(materials_out)} materials to {_OUT_PICKLIST_JSONL}",
        flush=True,
    )


if __name__ == "__main__":
    main()
