"""Task-agent adapters shared by PCFBench singleshot evals.

Two backends sit behind the ``TaskAgent`` Protocol:

  - ``PydanticAISingleshotAgent``: thin wrapper around the existing
    ``run_singleshot`` driver (deps + ``submit_*`` tool that raises to
    terminate).
  - ``AgentSDKSingleshotAgent``: runs the task through ``claude_agent_sdk``
    with the full ``claude_code`` tool preset (Read/Edit/Write/Bash/
    Glob/Grep/Task/...) plus an in-process MCP server that exposes the
    same ``submit_*`` tool the pydantic-ai path uses. Output is captured
    by the submit-tool handler, so the system prompt's "call submit_X"
    instruction is honored on this path. File-system side-effects are
    confined to a per-task temporary working directory via a PreToolUse
    hook plus the SDK's OS-level sandbox; ``WebSearch`` is denied; and
    network is closed off inside the sandbox. Thinking budgets from
    ``models.registry`` and ``ResultMessage.usage`` are mirrored for
    cross-backend parity.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pydantic as pyd
from pydantic_ai import Agent
from pydantic_ai.usage import RequestUsage, Usage

from pcfbench.agents._common import run_singleshot
from pcfbench.models.registry import SINGLESHOT_THINKING_BUDGETS

AGENT_SDK_PREFIX = "agent-sdk:"
_SUBMIT_MCP_SERVER_NAME = "pcfbench"
# Match the agentic pydantic-ai cap (`max_iterations=20`) so SDK and
# pydantic-ai backends get the same exploration budget when comparing.
_SDK_MAX_TURNS = 20


class TaskAgent(Protocol):
    async def run_task(
        self,
        user_prompt: str,
        *,
        usage: Usage | None = None,
        deps: Any = None,
    ) -> Any: ...


class PydanticAISingleshotAgent:
    """Wrap the existing Pydantic AI submit-tool singleshot pattern."""

    def __init__(
        self,
        *,
        agent: Agent,
        make_deps: Callable[[], Any],
        get_output: Callable[[Any], Any],
        output_recovered: Callable[[Any], bool],
    ) -> None:
        self.agent = agent
        self.make_deps = make_deps
        self.get_output = get_output
        self.output_recovered = output_recovered

    async def run_task(
        self,
        user_prompt: str,
        *,
        usage: Usage | None = None,
        deps: Any = None,
    ) -> Any:
        if deps is None:
            deps = self.make_deps()
        await run_singleshot(
            agent=self.agent,
            user_prompt=user_prompt,
            deps=deps,
            output_recovered=self.output_recovered,
            usage=usage,
        )
        return self.get_output(deps)


class AgentSDKSingleshotAgent:
    """Run a PCFBench singleshot task through claude-agent-sdk.

    The submit tool is exposed via an in-process MCP server so the model
    sees a tool with the same bare name as the pydantic-ai path
    (``submit_decomposition`` etc.); the SDK qualifies it with the
    ``mcp__pcfbench__`` prefix when listing it to the model. Alongside
    the submit tool, the model gets the full ``claude_code`` preset so
    it can read/write/grep/bash within the per-task scratch cwd.
    """

    def __init__(
        self,
        *,
        model_id: str,
        system_prompt: str,
        output_type: type[pyd.BaseModel],
        submit_tool_name: str,
        submit_tool_description: str,
    ) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.output_type = output_type
        self.submit_tool_name = submit_tool_name
        self.submit_tool_description = submit_tool_description

    async def run_task(
        self,
        user_prompt: str,
        *,
        usage: Usage | None = None,
        deps: Any = None,
    ) -> Any:
        del deps
        return await _run_submit_tool_query(
            model_id=self.model_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            submit_tool_name=self.submit_tool_name,
            submit_tool_description=self.submit_tool_description,
            output_type=self.output_type,
            usage=usage,
        )


async def _run_submit_tool_query(
    *,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    submit_tool_name: str,
    submit_tool_description: str,
    output_type: type[pyd.BaseModel],
    usage: Usage | None,
) -> Any:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        ThinkingConfigEnabled,
        create_sdk_mcp_server,
        query,
        tool,
    )

    captured: dict[str, Any] = {"output": None}

    @tool(
        submit_tool_name,
        submit_tool_description,
        output_type.model_json_schema(),
    )
    async def _submit(args: dict[str, Any]) -> dict[str, Any]:
        try:
            captured["output"] = output_type.model_validate(args)
        except pyd.ValidationError as exc:
            return {
                "content": [{"type": "text", "text": f"Validation error: {exc}"}],
                "is_error": True,
            }
        return {"content": [{"type": "text", "text": "submitted"}]}

    server = create_sdk_mcp_server(name=_SUBMIT_MCP_SERVER_NAME, tools=[_submit])
    bare_model = _strip_agent_sdk_prefix(model_id)

    with tempfile.TemporaryDirectory(prefix="pcfbench-agent-sdk-") as cwd_str:
        cwd = Path(cwd_str).resolve()
        options_kwargs: dict[str, Any] = {
            "model": bare_model,
            "system_prompt": system_prompt,
            "cwd": cwd_str,
            "tools": {"type": "preset", "preset": "claude_code"},
            "mcp_servers": {_SUBMIT_MCP_SERVER_NAME: server},
            "strict_mcp_config": True,
            "setting_sources": [],
            "skills": [],
            "plugins": [],
            "add_dirs": [],
            "permission_mode": "bypassPermissions",
            # WebSearch would let the model peek at the live web for
            # answers; deny it so benchmark numbers reflect closed-book
            # capability. WebFetch is similarly blocked at the sandbox
            # network layer below.
            "disallowed_tools": ["WebSearch"],
            "sandbox": {
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "network": {"deniedDomains": ["*"]},
            },
            "hooks": {
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[_make_temp_cwd_tool_guard(cwd)],
                    )
                ]
            },
            "max_turns": _SDK_MAX_TURNS,
        }
        budget = SINGLESHOT_THINKING_BUDGETS.get(bare_model, 0)
        if budget > 0:
            options_kwargs["thinking"] = ThinkingConfigEnabled()
            options_kwargs["max_thinking_tokens"] = budget

        options = ClaudeAgentOptions(**options_kwargs)
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, ResultMessage):
                _accumulate_usage(usage, message)
                break
    return captured["output"]


def _make_temp_cwd_tool_guard(cwd: Path) -> Callable[..., Any]:
    """PreToolUse hook that confines file-path tools to ``cwd``.

    Only path-bearing built-in tools are gated here (Read/Write/Edit/
    MultiEdit/NotebookEdit/Glob/Grep). Bash and other built-ins fall
    through to allow; the SDK's OS-level sandbox is the backstop. MCP
    tools (e.g. ``submit_*``) carry no file paths and pass through.
    """

    async def _guard(
        hook_input: Any,
        _tool_use_id: str | None,
        _context: Any,
    ) -> dict:
        if not isinstance(hook_input, dict):
            return _hook_allow()
        tool_name = hook_input.get("tool_name")
        raw_input = hook_input.get("tool_input") or {}
        if not isinstance(raw_input, dict):
            raw_input = {}
        paths = _candidate_tool_paths(tool_name, raw_input)
        if any(not _is_within_cwd(path, cwd) for path in paths):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "Benchmark Agent SDK tools are sandboxed to the "
                        "temporary working directory."
                    ),
                }
            }
        return _hook_allow()

    return _guard


def _hook_allow() -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def _candidate_tool_paths(
    tool_name: str | None, tool_input: dict[str, Any]
) -> list[Path]:
    if tool_name in {"Read", "Write", "Edit", "MultiEdit"}:
        return [_path_from_tool_input(tool_input, "file_path")]
    if tool_name == "NotebookEdit":
        return [_path_from_tool_input(tool_input, "notebook_path")]
    if tool_name in {"Glob", "Grep"}:
        return [_path_from_tool_input(tool_input, "path")]
    return []


def _path_from_tool_input(tool_input: dict[str, Any], key: str) -> Path:
    value = tool_input.get(key)
    return Path(value) if isinstance(value, str) and value else Path(".")


def _is_within_cwd(path: Path, cwd: Path) -> bool:
    resolved = (path if path.is_absolute() else cwd / path).resolve()
    return resolved == cwd or cwd in resolved.parents


def _accumulate_usage(usage: Usage | None, msg: Any) -> None:
    """Roll ``ResultMessage.usage`` into ``usage`` (if provided).

    The SDK exposes the Anthropic-shaped usage dict; map its keys onto
    pydantic-ai's ``RequestUsage`` so downstream telemetry is unified
    regardless of backend.
    """
    if usage is None:
        return
    raw = getattr(msg, "usage", None) or {}
    if not raw:
        return
    request_usage = RequestUsage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens") or 0),
    )
    usage.incr(request_usage)


def _strip_agent_sdk_prefix(model_id: str) -> str:
    return model_id.removeprefix(AGENT_SDK_PREFIX)
