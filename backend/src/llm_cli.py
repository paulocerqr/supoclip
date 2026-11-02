"""Structured LLM requests through a locally signed-in coding CLI.

``claude-code:<model>`` runs Claude Code (``claude -p``) and ``codex:<model>``
runs the Codex CLI (``codex exec``). Both bill the subscription the CLI is
signed in with, so no API key is needed. ``<model>`` may be ``default`` to use
the CLI's own default model.

Each agent request is one stateless CLI call carrying the agent's JSON schema.
pydantic-ai validates the reply; its output retries send the validation error
back as part of the next call's conversation, exactly as with an HTTP model.

Environment (all optional):

- ``LLM_CLI_TIMEOUT_SECONDS``: per-call limit, default 600.
- ``LLM_CLI_HOME``: HOME for the CLI process. The Docker setup mounts the
  host's ``~/.claude`` and ``~/.codex`` there.
- ``LLM_CLI_USER``: when the worker runs as root, run the CLI as this user so
  refreshed credentials stay owned by the host user instead of root.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pwd
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

logger = logging.getLogger(__name__)

CLI_PROVIDERS = frozenset({"claude-code", "codex"})
DEFAULT_TIMEOUT_SECONDS = 600
# Keys the CLIs would otherwise prefer over the subscription login.
_API_KEY_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY")
# Only what the CLIs need. The worker environment holds database URLs and
# provider secrets that a subprocess has no reason to see.
_PASSTHROUGH_ENV = (
    "PATH", "LANG", "LC_ALL", "TZ", "TMPDIR",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS",
)
_TRANSIENT_PATTERN = re.compile(
    r"overloaded|rate.?limit|\b(429|500|502|503|504|529)\b|timed? ?out|ECONNRESET|ETIMEDOUT|network",
    re.IGNORECASE,
)


class LLMCliError(RuntimeError):
    """A CLI call that failed. ``transient`` failures are worth retrying."""

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def cli_binary(provider: str) -> str:
    return "claude" if provider == "claude-code" else "codex"


def cli_config_error(provider: str) -> str | None:
    """A setup problem that would make every call fail, or None."""
    binary = cli_binary(provider)
    if shutil.which(binary) is None:
        return (
            f"Selected LLM provider is {provider}, but the '{binary}' CLI is not installed "
            "in this container. Rebuild with docker-compose.llm-cli.yml (see docs/configuration.md)."
        )
    return None


def _render_conversation(messages: list[ModelMessage], instructions: str | None) -> tuple[str, str]:
    """Flatten the agent conversation into (system prompt, user prompt)."""
    system: list[str] = []
    turns: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    system.append(part.content)
                elif isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        turns.append(content)
                    else:
                        turns.extend(item for item in content if isinstance(item, str))
                elif isinstance(part, RetryPromptPart):
                    turns.append(part.model_response())
                elif isinstance(part, ToolReturnPart):
                    turns.append(part.model_response_str())
        elif isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            if text:
                turns.append(f"Your previous answer was:\n{text}")
    if instructions:
        system.append(instructions)
    return "\n\n".join(system), "\n\n".join(turns)


def _process_identity() -> tuple[dict[str, str], str | None]:
    """Environment and (optional) user to run the CLI as."""
    env = {key: os.environ[key] for key in _PASSTHROUGH_ENV if key in os.environ}
    for key in _API_KEY_ENV:
        env.pop(key, None)
    home = os.getenv("LLM_CLI_HOME") or os.path.expanduser("~")
    env["HOME"] = home
    user = os.getenv("LLM_CLI_USER") or None
    if user and os.geteuid() != 0:
        user = None  # Only root can switch users; run as ourselves.
    return env, user


def _make_workdir(user: str | None) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix="supoclip-llm-"))
    if user:
        entry = pwd.getpwnam(user)
        os.chown(workdir, entry.pw_uid, entry.pw_gid)
    return workdir


async def _run(command: list[str], stdin: str, cwd: Path, env: dict[str, str], user: str | None) -> tuple[int, str, str]:
    timeout = float(os.getenv("LLM_CLI_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
        user=user,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin.encode()), timeout)
    except BaseException:
        # Timeout or cancellation: never leave a CLI running in the background.
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise
    return process.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _failure(provider: str, detail: str) -> LLMCliError:
    # Keep only the tail: the CLI's own error, not an echo of the transcript.
    detail = detail.strip()[-500:] or "no output"
    return LLMCliError(
        f"{provider} CLI call failed: {detail}",
        transient=bool(_TRANSIENT_PATTERN.search(detail)),
    )


async def _call_claude_code(model: str, system: str, prompt: str, schema: dict[str, Any] | None) -> str:
    env, user = _process_identity()
    workdir = _make_workdir(user)
    command = [
        "claude", "-p", "--output-format", "json",
        "--tools", "",                       # Plain completion: no file or shell access.
        "--setting-sources", "project",      # Skip the user's hooks, plugins and settings.
        "--strict-mcp-config",               # And their MCP servers.
        "--no-session-persistence",
    ]
    if system:
        command += ["--system-prompt", system]
    if schema is not None:
        command += ["--json-schema", json.dumps(schema)]
    if model != "default":
        command += ["--model", model]
    try:
        code, stdout, stderr = await _run(command, prompt, workdir, env, user)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        raise _failure("claude-code", stderr or stdout) from None
    if code != 0 or result.get("is_error"):
        raise _failure("claude-code", str(result.get("result") or stderr))
    if schema is not None and result.get("structured_output") is not None:
        return json.dumps(result["structured_output"], ensure_ascii=False)
    return str(result.get("result") or "")


async def _call_codex(model: str, system: str, prompt: str, schema: dict[str, Any] | None) -> str:
    env, user = _process_identity()
    workdir = _make_workdir(user)
    output_path = workdir / "answer.txt"
    command = [
        "codex", "exec",
        "--skip-git-repo-check", "--ephemeral",
        "--ignore-user-config", "--ignore-rules",  # Skip the user's hooks, MCP and rules.
        "--sandbox", "read-only",
        "--color", "never",
        "--output-last-message", str(output_path),
    ]
    if schema is not None:
        schema_path = workdir / "schema.json"
        schema_path.write_text(json.dumps(schema))
        command += ["--output-schema", str(schema_path)]
    if model != "default":
        command += ["--model", model]
    command.append("-")  # Prompt on stdin.
    # codex exec has no system prompt flag; lead with the instructions.
    full_prompt = f"{system}\n\n---\n\n{prompt}" if system else prompt
    try:
        if user:
            entry = pwd.getpwnam(user)
            for path in workdir.iterdir():
                os.chown(path, entry.pw_uid, entry.pw_gid)
        code, stdout, stderr = await _run(command, full_prompt, workdir, env, user)
        answer = output_path.read_text() if output_path.is_file() else ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if code != 0 or not answer.strip():
        raise _failure("codex", stderr or stdout)
    return answer


def build_cli_model(provider: str, model: str) -> FunctionModel:
    """A pydantic-ai model that answers through the given CLI."""
    call = _call_claude_code if provider == "claude-code" else _call_codex

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        system, prompt = _render_conversation(messages, info.instructions)
        output_object = info.model_request_parameters.output_object
        schema = output_object.json_schema if output_object else None
        try:
            text = await call(model, system, prompt, schema)
        except (TimeoutError, asyncio.TimeoutError):
            raise LLMCliError(f"{provider} CLI call timed out", transient=True) from None
        return ModelResponse(parts=[TextPart(text)], model_name=f"{provider}:{model}")

    profile = ModelProfile(
        supports_json_schema_output=True,
        default_structured_output_mode="native",
        # Codex sends the schema to OpenAI's strict structured outputs, which
        # require every property and no extra keys. Claude Code accepts either.
        json_schema_transformer=OpenAIJsonSchemaTransformer if provider == "codex" else None,
    )
    return FunctionModel(respond, model_name=f"{provider}:{model}", profile=profile)
