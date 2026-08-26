"""Model identity + reasoning-config registries for the PCFBench harness.

Frozen sets of supported model IDs per provider, plus the reasoning-effort
and thinking-budget tables consumed by ``models.factory``.
"""

from __future__ import annotations

ANTHROPIC_MODELS = frozenset(
    {
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5@20251001",
    }
)

OPENAI_MODELS = frozenset({"gpt-5.5", "gpt-5.4-mini"})

GEMINI_MODELS = frozenset(
    {
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
    }
)

DEEPSEEK_VERTEX_MODELS = frozenset({"deepseek-ai/deepseek-v3.2-maas"})

# Locally-served open-weight models. These need no API key and no cloud
# account: Ollama serves them over an OpenAI-compatible endpoint on
# localhost, so every single-shot PCFBench task is reproducible on a
# laptop. Model IDs are Ollama tags.
LOCAL_OLLAMA_MODELS = frozenset({"qwen3.5:4b"})


# Single-shot reasoning configs.
SINGLESHOT_THINKING_BUDGETS: dict[str, int] = {
    "claude-opus-4-6": 8192,
    "gemini-3.1-pro-preview": 8192,
    # Flash retried with explicit thinking-on at 8192 + max_tokens=16000;
    # the canonical sweep had thinking-on but ran into Model token limit
    # because max_tokens defaulted to 8192. Now that max_tokens is bumped
    # in _build_gemini, see whether thinking-on Flash is competitive on
    # latency vs thinking-off.
    "gemini-3-flash-preview": 8192,
}

SINGLESHOT_OPENAI_REASONING_EFFORT: dict[str, str] = {
    "gpt-5.5": "high",
}


# Agentic reasoning configs (everything that supports reasoning gets the
# 8192-token budget; OpenAI gets reasoning_effort=high).
AGENTIC_THINKING_BUDGETS: dict[str, int] = {
    "claude-opus-4-6": 8192,
    "claude-sonnet-4-6": 8192,
    "claude-haiku-4-5@20251001": 8192,
    "gemini-3.1-pro-preview": 8192,
    "gemini-3-flash-preview": 8192,
}

AGENTIC_OPENAI_REASONING_EFFORT: dict[str, str] = {
    "gpt-5.5": "high",
}


def has_reasoning_singleshot(model_id: str) -> bool:
    # Treat ``budget=0`` as reasoning-OFF: an explicit-disable entry
    # belongs in SINGLESHOT_THINKING_BUDGETS so the factory passes
    # ``thinking_budget=0`` to the SDK, but we don't want the
    # temperature=1.0 reasoning-on default.
    return (
        SINGLESHOT_THINKING_BUDGETS.get(model_id, 0) > 0
        or model_id in SINGLESHOT_OPENAI_REASONING_EFFORT
    )


def has_reasoning_agentic(model_id: str) -> bool:
    return (
        AGENTIC_THINKING_BUDGETS.get(model_id, 0) > 0
        or model_id in AGENTIC_OPENAI_REASONING_EFFORT
    )
