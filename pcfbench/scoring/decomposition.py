"""Decomposition (Task 1) scoring via a Gemini-2.5-Flash compositional-
match judge.

The judge emits ``_JudgeResult.matches: list[_JudgeMatchGroup]`` where
each group is a (predicted_indices, expected_indices) pair. From those
groups we derive precision / recall / F1 / Kendall-tau / exact-set per
item, and run-level means.

The judge is invoked synchronously via Pydantic AI's ``GoogleModel``
with structured-tool output (``ToolOutput``). Gemini 2.5 Flash has no
extended thinking, so the ``allow_text_output=False`` /
``tool_choice: any`` mode does not collide with thinking restrictions.
A per-(predicted, expected) ``functools.lru_cache`` keeps the judge
call free for repeated tuple inputs (matching the existing harness).
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Iterable

import pydantic as pyd
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.output import ToolOutput
from pydantic_ai.providers.google import GoogleProvider
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from pcfbench.models.vertex_auth import (
    vertex_project_id,
)

logger = logging.getLogger(__name__)

DECOMPOSITION_JUDGE_MODEL = "gemini-2.5-flash"

DECOMPOSITION_JUDGE_PROMPT = """\
You are an LCA expert grading a model's bill-of-materials decomposition.

You are given:
  - PREDICTED: a model's list of input materials, ordered from highest to \
lowest mass contribution.
  - EXPECTED: the ground-truth list of input materials, ordered from \
highest to lowest mass contribution.

Your job: align the PREDICTED list with the EXPECTED list using semantic \
equivalence at the level of an LCA practitioner.  Models legitimately \
decompose products at different fabrication levels, so both finer- and \
coarser-than-expected breakdowns are valid as long as they are \
COMPOSITIONALLY CONSISTENT.

A "match group" is one-or-more PREDICTED items that together correspond to \
one-or-more EXPECTED items.  Four group types are valid:

  (a) 1-to-1 -- direct match at the same fabrication level.
      "PE" <-> "polyethylene"
      "aluminium scrap (pre-consumer)" <-> "aluminium scrap"
      "PVB" <-> "polyvinyl butyral"
      "flat glass" <-> "PLANICLEAR(R) glass"

  (b) 1-to-N -- one PREDICTED aggregate that decomposes into N EXPECTED \
sub-components (the prediction is COARSER than the ground truth).
      "concrete" <-> {"sand", "water", "cement"}
      "stainless steel" <-> {"iron", "chromium", "nickel"}
      "dough" <-> {"flour", "water", "yeast"}

  (c) N-to-1 -- N PREDICTED sub-components that compose one EXPECTED \
aggregate (the prediction is FINER than the ground truth).
      {"iron", "carbon"} <-> "steel"
      {"sand", "water", "cement"} <-> "concrete"
      {"cotton fibre", "weaving"} <-> "cotton fabric"

  (d) N-to-N -- M PREDICTED items collectively correspond to N EXPECTED \
items where neither side is a single aggregate of the other, but the two \
sides cover the same set of inputs at compatible fabrication levels.
      {"silicon", "copper", "zinc", "magnesium"} <-> {"alloying elements", "alloying elements (scrap)"}
      {"polyethylene", "ethylene vinyl acetate"} <-> {"thermoplastic polymer", "rubber-like polymer"}
      {"flour", "sugar", "yeast"} <-> {"baking dry mix", "leavening agents"}

Recycled-content qualifier: the presence or absence of a "recycled" prefix \
on either side is NOT grounds for refusing a match.  "plastic" matches \
"recycled plastic"; "cotton fibre" matches "recycled cotton fibre"; \
"aluminium" matches "recycled aluminium".  Treat "recycled X" and "X" as \
compositionally equivalent for the purpose of group alignment.

For every group, the predicted side and expected side must be \
COMPOSITIONALLY EQUIVALENT: the aggregated items, taken together, plausibly \
form the other side of the group.  Match by material identity only -- ignore \
percentages, masses, and ordering.

Each PREDICTED index appears in AT MOST ONE group.  Each EXPECTED index \
appears in AT MOST ONE group.  Items that don't fit any defensible group are \
left unmatched.

INVALID matches:
  - "polyethylene" does NOT match "polypropylene" (chemically distinct).
  - "polyethylene" does NOT match "concrete" (no compositional relationship).
  - {"oxygen", "hydrogen"} does NOT match "milk" (true at the molecular \
level but absurd at the LCA fabrication level).
  - "aluminium" does NOT match "iron".
  - "water" does NOT match "fruit concentrates".

Pick the alignment that maximizes coverage of compositionally-defensible \
match groups."""


class _JudgeMatchGroup(pyd.BaseModel):
    """A compositional match group.

    1-to-1, 1-to-N (predicted is coarser), or N-to-1 (predicted is finer).
    Both lists are non-empty; each index appears in at most one group.
    """

    predicted_indices: list[int]
    expected_indices: list[int]


class _JudgeResult(pyd.BaseModel):
    matches: list[_JudgeMatchGroup]


# Lazily build the judge agent once per process. The agent is stateless
# w.r.t. inputs so a single instance is safe across cached calls.
_JUDGE_AGENT: Agent | None = None


def _judge_agent() -> Agent:
    global _JUDGE_AGENT
    if _JUDGE_AGENT is None:
        _JUDGE_AGENT = Agent(
            GoogleModel(
                DECOMPOSITION_JUDGE_MODEL,
                # WHY: pydantic-ai's GoogleProvider overloads don't expose
                # the vertexai+project+location combo, though it's valid at
                # runtime per the docstring.
                provider=GoogleProvider(  # type: ignore[call-overload]
                    vertexai=True,
                    project=vertex_project_id(),
                    location="global",
                ),
                settings=GoogleModelSettings(temperature=0.0),
            ),
            # WHY: pydantic-ai's ToolOutput is typed for str outputs but
            # accepts any pydantic model at runtime; the generic isn't
            # exposed through the public stub.
            output_type=ToolOutput(
                _JudgeResult,  # type: ignore[arg-type]
                name="submit_judgment",
                description="Submit the alignment between PREDICTED and EXPECTED.",
            ),
            system_prompt=DECOMPOSITION_JUDGE_PROMPT,
        )
    return _JUDGE_AGENT


_JUDGE_LOCKS: dict[tuple[tuple[str, ...], tuple[str, ...]], asyncio.Lock] = {}
_JUDGE_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], _JudgeResult | None] = {}


def _is_judge_retryable(exc: BaseException) -> bool:
    """429 (rate limit) and 5xx (transient server) are retryable.
    Schema/auth/4xx-other are not."""
    if isinstance(exc, ModelHTTPError):
        code = exc.status_code
        return code == 429 or 500 <= code < 600
    return False


async def _run_judge_async(
    predicted: tuple[str, ...],
    expected: tuple[str, ...],
) -> _JudgeResult | None:
    """Async, async-cached judge call. Reuses one ``asyncio.Lock`` per
    (predicted, expected) tuple so two concurrent dataset items asking
    the same question collapse to a single Gemini call."""
    key = (predicted, expected)
    if key in _JUDGE_CACHE:
        return _JUDGE_CACHE[key]
    lock = _JUDGE_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if key in _JUDGE_CACHE:
            return _JUDGE_CACHE[key]
        result = await _do_judge_call(predicted, expected)
        _JUDGE_CACHE[key] = result
        return result


@functools.lru_cache(maxsize=4096)
def _run_judge(
    predicted: tuple[str, ...],
    expected: tuple[str, ...],
) -> _JudgeResult | None:
    """Sync judge for ad-hoc use (tests, scripts). Driven through the
    same agent; safe outside an event loop."""
    if not predicted or not expected:
        return _JudgeResult(matches=[])
    return asyncio.run(_do_judge_call(predicted, expected))


async def _do_judge_call(
    predicted: tuple[str, ...],
    expected: tuple[str, ...],
) -> _JudgeResult | None:
    if not predicted or not expected:
        return _JudgeResult(matches=[])

    pred_lines = "\n".join(f"  [{i}] {name}" for i, name in enumerate(predicted))
    exp_lines = "\n".join(f"  [{i}] {name}" for i, name in enumerate(expected))
    prompt = (
        f"PREDICTED ({len(predicted)} items):\n{pred_lines}\n\n"
        f"EXPECTED ({len(expected)} items):\n{exp_lines}\n\n"
        f"Return every valid match group as a (predicted_indices, "
        f"expected_indices) pair.  Use 1-to-1 groups for direct matches, "
        f"1-to-N groups when a predicted item aggregates several expected "
        f"items, and N-to-1 groups when several predicted items together "
        f"compose one expected item."
    )
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(6),
            wait=wait_exponential_jitter(initial=2, max=30),
            retry=retry_if_exception(_is_judge_retryable),
            reraise=True,
        ):
            with attempt:
                result = await _judge_agent().run(prompt)
        out = result.output
        if not isinstance(out, _JudgeResult):
            return None
    except Exception:
        logger.exception("Decomposition judge failed (after retries)")
        return None

    used_pred: set[int] = set()
    used_exp: set[int] = set()
    clean: list[_JudgeMatchGroup] = []
    for m in out.matches:
        pred_idx = [i for i in m.predicted_indices if 0 <= i < len(predicted)]
        exp_idx = [i for i in m.expected_indices if 0 <= i < len(expected)]
        pred_idx = [i for i in pred_idx if i not in used_pred]
        exp_idx = [i for i in exp_idx if i not in used_exp]
        if not pred_idx or not exp_idx:
            continue
        used_pred.update(pred_idx)
        used_exp.update(exp_idx)
        clean.append(
            _JudgeMatchGroup(predicted_indices=pred_idx, expected_indices=exp_idx)
        )
    return _JudgeResult(matches=clean)


# --- Per-item / run-level scorers -----------------------------------------


def _matched_pred_count(judge: _JudgeResult) -> int:
    return sum(len(g.predicted_indices) for g in judge.matches)


def _matched_exp_count(judge: _JudgeResult) -> int:
    return sum(len(g.expected_indices) for g in judge.matches)


def _kendall_tau(
    pred_ranks: list[float],
    exp_ranks: list[float],
) -> float | None:
    n = len(pred_ranks)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for k in range(i + 1, n):
            dp = pred_ranks[i] - pred_ranks[k]
            de = exp_ranks[i] - exp_ranks[k]
            sign = dp * de
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = n * (n - 1) / 2
    if total == 0:
        return None
    return (concordant - discordant) / total


def score_item(
    predicted: list[str] | None,
    expected: list[str] | None,
) -> dict[str, float | bool | None]:
    """One judge call, return precision / recall / F1 / Kendall-tau /
    exact-set-match for the (predicted, expected) pair. ``None`` if
    either side is missing or the judge fails."""
    base: dict[str, float | bool | None] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "kendall_tau": None,
        "exact_set_match": None,
    }
    if predicted is None or expected is None:
        return base
    pred_clean = [c.strip() for c in predicted if c and c.strip()]
    exp_clean = [c.strip() for c in expected if c and c.strip()]
    judge = _run_judge(tuple(pred_clean), tuple(exp_clean))
    if judge is None:
        return base
    matched_pred = _matched_pred_count(judge)
    matched_exp = _matched_exp_count(judge)
    p = matched_pred / len(pred_clean) if pred_clean else 0.0
    r = matched_exp / len(exp_clean) if exp_clean else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    base["precision"] = p
    base["recall"] = r
    base["f1"] = f1
    if len(judge.matches) >= 2:
        pred_ranks = [
            sum(g.predicted_indices) / len(g.predicted_indices) for g in judge.matches
        ]
        exp_ranks = [
            sum(g.expected_indices) / len(g.expected_indices) for g in judge.matches
        ]
        base["kendall_tau"] = _kendall_tau(pred_ranks, exp_ranks)
    base["exact_set_match"] = matched_pred == len(pred_clean) and matched_exp == len(
        exp_clean
    )
    return base


async def score_item_async(
    predicted: list[str] | None,
    expected: list[str] | None,
) -> dict[str, float | bool | None]:
    """Async sibling of ``score_item`` — runs the judge in a thread so
    multiple concurrent dataset items each get their own thread (the LRU
    cache is shared across threads)."""
    base: dict[str, float | bool | None] = {
        "precision": None,
        "recall": None,
        "f1": None,
        "kendall_tau": None,
        "exact_set_match": None,
    }
    if predicted is None or expected is None:
        return base
    pred_clean = [c.strip() for c in predicted if c and c.strip()]
    exp_clean = [c.strip() for c in expected if c and c.strip()]
    judge = await _run_judge_async(tuple(pred_clean), tuple(exp_clean))
    if judge is None:
        return base
    matched_pred = _matched_pred_count(judge)
    matched_exp = _matched_exp_count(judge)
    p = matched_pred / len(pred_clean) if pred_clean else 0.0
    r = matched_exp / len(exp_clean) if exp_clean else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    base["precision"] = p
    base["recall"] = r
    base["f1"] = f1
    if len(judge.matches) >= 2:
        pred_ranks = [
            sum(g.predicted_indices) / len(g.predicted_indices) for g in judge.matches
        ]
        exp_ranks = [
            sum(g.expected_indices) / len(g.expected_indices) for g in judge.matches
        ]
        base["kendall_tau"] = _kendall_tau(pred_ranks, exp_ranks)
    base["exact_set_match"] = matched_pred == len(pred_clean) and matched_exp == len(
        exp_clean
    )
    return base


def run_mean_float(values: Iterable[float | None]) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def run_mean_bool(values: Iterable[bool | None]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(1.0 if v else 0.0 for v in vals) / len(vals)
