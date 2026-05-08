"""Task-agent adapters shared by PCFBench singleshot evals."""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pydantic as pyd
from pydantic_ai import Agent
from pydantic_ai.usage import Usage

from pcfbench.agents._common import run_singleshot

AGENT_SDK_PREFIX = "agent-sdk:"


class TaskAgent(Protocol):
    async def run_task(
        self, user_prompt: str, usage: Usage | None = None
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
        self, user_prompt: str, usage: Usage | None = None
    ) -> Any:
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
    """Run a PCFBench singleshot task through claude-agent-sdk."""

    def __init__(self, *, model_id: str, system_prompt: str, output_type: type) -> None:
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.output_type = output_type

    async def run_task(
        self, user_prompt: str, usage: Usage | None = None
    ) -> Any:
        del usage
        return await _run_structured_query(
            model_id=self.model_id,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            output_type=self.output_type,
        )


async def _run_structured_query(
    *,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    output_type: type,
) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, ResultMessage, query

    schema = _schema_json(output_type)
    structured_prompt = (
        f"{user_prompt}\n\n"
        "Respond with ONLY valid JSON matching this schema "
        "(no prose, no markdown fences):\n"
        f"{schema}"
    )
    with tempfile.TemporaryDirectory(prefix="pcfbench-agent-sdk-") as cwd_str:
        cwd = Path(cwd_str).resolve()
        options = ClaudeAgentOptions(
            model=_strip_agent_sdk_prefix(model_id),
            system_prompt=system_prompt,
            cwd=cwd_str,
            tools={"type": "preset", "preset": "claude_code"},
            mcp_servers={},
            strict_mcp_config=True,
            setting_sources=[],
            skills=[],
            plugins=[],
            add_dirs=[],
            permission_mode="bypassPermissions",
            disallowed_tools=["WebSearch"],
            sandbox={
                "enabled": True,
                "autoAllowBashIfSandboxed": True,
                "allowUnsandboxedCommands": False,
                "network": {"deniedDomains": ["*"]},
            },
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher=None, hooks=[_make_temp_cwd_tool_guard(cwd)])
                ]
            },
        )
        async for message in query(prompt=structured_prompt, options=options):
            if isinstance(message, ResultMessage):
                if message.structured_output is not None:
                    parsed = _validate_obj(message.structured_output, output_type)
                    if parsed is not None:
                        return parsed
                return _parse_json_result(message.result or "", output_type)
    return None


def _make_temp_cwd_tool_guard(cwd: Path):
    async def _guard(tool_input: Any, _tool_use_id: str | None, _context: Any) -> dict:
        tool_name = tool_input.get("tool_name") if isinstance(tool_input, dict) else None
        raw_input = (
            tool_input.get("tool_input", {}) if isinstance(tool_input, dict) else {}
        )
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
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }

    return _guard


def _candidate_tool_paths(tool_name: str | None, tool_input: dict[str, Any]) -> list[Path]:
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


def _strip_agent_sdk_prefix(model_id: str) -> str:
    return model_id.removeprefix(AGENT_SDK_PREFIX)


def _schema_json(output_type: type) -> str:
    if hasattr(output_type, "model_json_schema"):
        return json.dumps(output_type.model_json_schema(), indent=2)
    return json.dumps({"type": "object"}, indent=2)


def _validate_obj(obj: Any, output_type: type) -> Any:
    if isinstance(output_type, type) and issubclass(output_type, pyd.BaseModel):
        try:
            return output_type.model_validate(obj)
        except pyd.ValidationError:
            return None
    return obj


def _parse_json_result(text: str, output_type: type) -> Any:
    cleaned = text.strip()
    if not cleaned:
        return None

    parsed = _parse_json_text(cleaned, output_type)
    if parsed is not None:
        return parsed

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL):
        parsed = _parse_json_text(match.group(1).strip(), output_type)
        if parsed is not None:
            return parsed

    obj_text = _first_json_object(cleaned)
    if obj_text is not None:
        return _parse_json_text(obj_text, output_type)
    return None


def _parse_json_text(text: str, output_type: type) -> Any:
    if isinstance(output_type, type) and issubclass(output_type, pyd.BaseModel):
        try:
            return output_type.model_validate_json(text)
        except pyd.ValidationError:
            return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None
