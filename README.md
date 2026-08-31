# PCFBench

Process-based Product Carbon Footprint benchmark for evaluating LLMs and agents on
the operational steps of life-cycle assessment (LCA): bill-of-materials decomposition,
mapping triage, ecoinvent process matching, literature extraction of physical input
rates, and total kgCO₂e prediction against expert-grounded EPDs.

Preprint - PCFBench: A Diagnostic Benchmark for Product Carbon Footprint Estimation: https://arxiv.org/abs/2608.27716

## Tasks

| ID | Task | Items | GT claims | Headline metric |
| -- | ---- | ----: | --------: | --------------- |
| 1 | Product decomposition (BOM) | 94 | 94 | Judge-aligned F₁ on compositional match groups |
| 2 | Mapping triage | 200 | 200 | Accuracy / F₁ on `should_map` binary |
| 3 | Background-database mapping | 109 | 109 | Exact-match top-1 against expert reference products |
| 4 | Material input-rate extraction | 22 | 55 | Claim F₁ on greedy (value, unit) match |
| 5 | Energy input-rate extraction | 14 | 34 | Claim F₁ on greedy (value, unit) match |
| 7 | Total kgCO₂e prediction (EPD)       | 175 | 175 | Median \|RE\|, within-2× / within-5× rate |

(Step 6 is deterministic arithmetic and not separately evaluated.)

Tasks 2 and 3 share an **ecoinvent v3.11 picklist** of 2,571 reference products
(strict superset of the prior v3.10 1,663-row list); the picklist is published
alongside the task data.

## Quick start

```bash
# 1. Clone
git clone https://github.com/watershed-climate/pcfbench
cd pcfbench

# 2. Install
pip install -e .   # or: uv sync

# 3. Get the data from Hugging Face into ./pcfbench_data_external/:
#    hf download Watershed-Climate/PCFBench --repo-type dataset \
#      --local-dir pcfbench_data_external

```

To run any of {Opus 4.6, Sonnet 4.6, Haiku 4.5, Gemini 3.1 Pro, Gemini 3 Flash,
DeepSeek v3.2} you need Vertex AI access:

- `gcloud auth application-default login` (or a service-account JSON via
  `GOOGLE_APPLICATION_CREDENTIALS`)
- `VERTEX_PROJECT_ID` / `GOOGLE_CLOUD_PROJECT` / `GCP_PROJECT` set (or rely on
  ADC's default project)

For the OpenAI rows: `OPENAI_API_KEY`. `ANTHROPIC_API_KEY` is **not** supported
— Anthropic models route through Vertex AI.

```bash
# 4. Run an eval (example: decomposition on 10 items with Haiku 4.5)
python -m pcfbench.evals.runner pcfbench_decomposition \
    --model "claude-haiku-4-5@20251001" \
    --data-dir pcfbench_data_external \
    --limit 10
```

Per-item results land in `pcfbench/runs/<eval>__<model>.jsonl`; the summary prints to stdout.

## Available evals

```
pcfbench_decomposition
pcfbench_triage                       pcfbench_triage_agentic
pcfbench_mapping_with_context         pcfbench_mapping_agentic_with_context
pcfbench_extraction                   pcfbench_extraction_query_only_estimate
pcfbench_epd_name_only                pcfbench_epd_with_description
pcfbench_epd_with_composition         pcfbench_epd_with_region
pcfbench_stepwise_with_description
```

The four `pcfbench_epd_*` evals are a context-ablation sweep over the same
175-item EPD set; they differ only in which fields the model sees
(name → +description → +composition → +region). The paper's headline T7
column uses `pcfbench_epd_with_description` to disclosure-match the
compositional pipeline; Figure 3 panel (e) uses all four.

The `_agentic` variants give the model tool access to retrieve from the ecoinvent
picklist on demand instead of putting it all in the prompt.

The `pcfbench_stepwise_with_description` eval drives the bottom-up Task 7
pipeline (decompose → triage → map → rate → sum) end-to-end on the EPD set,
mirroring the disclosure level of the headline T7 single-shot baseline
(`pcfbench_epd_with_description` — name + description only) so direct-vs-
compositional is an apples-to-apples comparison. It grades on **structural
invariants only** — mass balance, ghost components, recursion depth, per-stage
success — not the final kgCO2e number. Material emission factors are licensed
under ecoinvent and are not redistributable, so the shipping default
`ef_resolver` is a no-op (returns `None` for every reference product). The
pipeline still runs to completion: per-component rates, BOM structure, and
energy contributions (using the public ecoinvent v3.10 energy constants baked
into the source) are all populated. License-holders can plug in their own
`ef_resolver` via `StepwiseConfig.ef_resolver` to compute and grade the kgCO2e
column. Aggregate kgCO2e numbers in the paper's direct-vs-compositional figure
are reproducible from the published per-item traces in
`pcfbench/runs/pcfbench_stepwise_with_description/*.jsonl` without an
ecoinvent license.

## Repository layout

```
pcfbench/
├── agents/         # Per-task agents (Pydantic AI), one module per task
├── evals/
│   └── runner.py   # CLI entry point: load JSONL, dispatch, score, write JSONL
├── models/
│   ├── factory.py  # build_agent(model_id) -> typed Agent
│   └── registry.py # Frozen sets of supported model ids per provider
├── picklist/
│   ├── build_picklist_json.py  # Rebuild the picklist from the public ecoinvent xlsx
│   ├── embed_gemini.py         # Gemini embeddings (used by MaterialLibrary)
│   └── embed_gte.py            # GTE-Large embeddings (used by Parakeet)
├── scoring/        # Per-task per-item scorers
├── tools/          # Ecoinvent search/inspect tools for the agentic variants
├── sweep_all.py    # Run every (eval × model) combination and produce a summary table
├── README.md
└── LICENSE
```

## Adding a model

1. Add the model id to the appropriate frozen set in `pcfbench/models/registry.py`
   (`ANTHROPIC_MODELS`, `OPENAI_MODELS`, `GEMINI_MODELS`, `DEEPSEEK_VERTEX_MODELS`).
2. Verify the auth path in `pcfbench/models/factory.py` covers your model's provider.
3. Run a small smoke: `python -m pcfbench.evals.runner pcfbench_decomposition --model <new-id> --limit 2`.

## Reproducing the paper

For headline-table parity, run the full sweep — every model in `MODELS_8`
crossed with every eval in `EVALS_5`:

```bash
python -m pcfbench.sweep_all --data-dir pcfbench_data_external
```

Numbers will differ from the paper if you (a) rebuild the picklist from a newer
ecoinvent release, (b) run with reasoning settings that drift from the pinned
configuration, or (c) due to LLM non-determinism; bootstrap CIs in the paper
resample over benchmark items, not over model runs. The paper's reasoning-on
rows use:
- Claude Opus 4.6: extended thinking, 8192-token budget
- Gemini 3.1 Pro: thinking, 8192-token budget
- GPT-5.5: `reasoning_effort=high`

**Ecoinvent license required for one column / two figure panels.** Tasks 1–5
and the single-shot Task 7 column (Table 2: "Validate") are fully reproducible
from the published artifact with only model-provider credentials. The
**End-to-end / Compositional** column of Table 2 and **panels (a) and (b) of
Figure 3** report the bottom-up pipeline's aggregated kgCO2e and require
emission factors from the licensed ecoinvent v3.11 database to compute. The
shipping default `ef_resolver` is a no-op; license-holders can plug in their
own resolver via `StepwiseConfig.ef_resolver` to populate `kgco2e_predicted`.
Panel (c) of Figure 3 (mass-conservation violation rate) is reproducible
without a license — it scores the structural invariants emitted by the
pipeline regardless of EF availability.

## Datasets

Data is published on Hugging Face at
[Watershed-Climate/PCFBench](https://huggingface.co/datasets/Watershed-Climate/PCFBench)
and consists of six JSONL files (one per task; Tasks 4 and 5 are split by
item-level material/energy tag). Croissant metadata is generated by the Hub and
served at the dataset's `/croissant` endpoint. Schema details and provenance are
in the dataset's `DATASHEET.md`.

Each task JSONL row has the same envelope:

```json
{
  "id": "...",
  "input": { /* task-specific */ },
  "expected_output": { /* task-specific ground truth */ },
  "metadata": { "product_category": "...", /* task-specific extras */ }
}
```

## Citation

```bibtex
@misc{pcfbench2026,
  title  = {{PCFBench}: A Diagnostic Benchmark for Product Carbon Footprint
            Estimation},
  author = {Rao, Krishna and Dumit, Andrew and Ulissi, Shaena and
            Feintzeig, Jacob and Joyce, P. James and Frank, Daniel and
            Watson, Steven and Glidden, Jonathan and Dinc, Gizem Ilayda and
            Kwee, Travis M.},
  year   = {2026},
  eprint = {2608.27716},
  archivePrefix = {arXiv}
}
```

## License

[Apache License 2.0](LICENSE).

The shared ecoinvent v3.11 picklist used by Tasks 2 and 3 ships with this
repository at `pcfbench/picklist/ecoinvent_picklist.jsonl` (2,571 rows;
row order is load-bearing because the precomputed embedding `.npy` files
index into it positionally). It is derived from the publicly available
ecoinvent v3.11 *Database Overview* workbook and can be regenerated with
`python -m pcfbench.picklist.build_picklist_json`.
