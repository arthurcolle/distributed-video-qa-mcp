"""Render the tool contract (docs/tool-contract.md) from the live server definition.

The document is generated, never hand-edited: a test regenerates it and fails
if the committed copy differs, so the docs cannot drift from the code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from mcp.client import Client

from . import __version__
from .config import Settings
from .errors import CODES

GROUPS: list[tuple[str, str, list[str]]] = [
    ("Acquire", "Turn a link into a file.", ["video_probe", "video_download"]),
    ("Index", "Make videos searchable. Long work runs as jobs.", ["video_ingest", "video_ingest_batch"]),
    ("Jobs", "Follow and control background work.", ["video_job", "video_jobs", "video_cancel"]),
    (
        "Retrieve",
        "Find evidence across every indexed video.",
        ["video_ask", "video_search", "video_search_by_image", "video_find_phrase"],
    ),
    (
        "Inspect",
        "Read, look at and cut one video.",
        ["video_transcript", "video_frames", "video_chapters", "video_clip"],
    ),
    ("Library", "Manage what is indexed.", ["video_list", "video_info", "video_delete", "video_status"]),
]


def _type(schema: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in schema:
        parts = [_type(s, defs) for s in schema["anyOf"]]
        nullable = "null" in parts
        parts = [p for p in parts if p != "null"]
        return " \\| ".join(parts) + (" \\| null" if nullable else "")
    if "enum" in schema:
        return " \\| ".join(f"`{v}`" for v in schema["enum"])
    if "const" in schema:
        return f"`{schema['const']}`"
    kind = schema.get("type", "any")
    if kind == "array":
        inner = _type(schema.get("items", {}), defs)
        return f"({inner})[]" if "|" in inner else f"{inner}[]"
    if kind == "object" and "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        return f"map<string, {_type(schema['additionalProperties'], defs)}>"
    return {"integer": "int", "number": "number", "string": "string", "boolean": "bool", "object": "object"}.get(
        kind, str(kind)
    )


def _limits(schema: dict[str, Any]) -> str:
    keys = {
        "minimum": "≥", "exclusiveMinimum": ">", "maximum": "≤", "exclusiveMaximum": "<",
        "minLength": "min len ", "maxLength": "max len ", "minItems": "min ", "maxItems": "max ",
    }  # fmt: skip
    found = []
    for source in [schema, *schema.get("anyOf", [])]:
        for key, label in keys.items():
            if key in source:
                found.append(f"{label}{source[key]}")
    return ", ".join(dict.fromkeys(found))


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _fields(schema: dict[str, Any], defs: dict[str, Any], *, inputs: bool) -> list[str]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return ["_No arguments._"] if inputs else ["_Empty._"]
    if inputs:
        rows = ["| Argument | Type | Default | Limits | Description |", "| --- | --- | --- | --- | --- |"]
    else:
        rows = ["| Field | Type | Description |", "| --- | --- | --- |"]
    for name, prop in props.items():
        kind = _type(prop, defs)
        description = _cell(prop.get("description", ""))
        if inputs:
            default = "**required**" if name in required else f"`{prop.get('default')!r}`".replace("None", "null")
            rows.append(f"| `{name}` | {kind} | {default} | {_limits(prop)} | {description} |")
        else:
            rows.append(f"| `{name}` | {kind} | {description} |")
    return rows


def _hints(tool: Any) -> str:
    a = tool.annotations
    flags = ["read-only" if a.read_only_hint else "writes"]
    if a.destructive_hint:
        flags.append("destructive")
    if a.idempotent_hint:
        flags.append("idempotent")
    flags.append("network" if a.open_world_hint else "local")
    return " · ".join(flags)


def _referenced(schema: dict[str, Any], defs: dict[str, Any], seen: list[str]) -> None:
    """Collect $defs names reachable from a schema, depth-first, in first-use order."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if ref:
            name = ref.rsplit("/", 1)[-1]
            if name not in seen:
                seen.append(name)
                _referenced(defs.get(name, {}), defs, seen)
        for value in schema.values():
            _referenced(value, defs, seen)
    elif isinstance(schema, list):
        for value in schema:
            _referenced(value, defs, seen)


async def render() -> str:
    from .server import build_server

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(state_dir=Path(tmp), embed_backend="hash")
        mcp, app = build_server(settings)
        try:
            async with Client(mcp) as client:
                tools = {tool.name: tool for tool in (await client.list_tools()).tools}
                resources = (await client.list_resources()).resources
                templates = (await client.list_resource_templates()).resource_templates
                prompts = (await client.list_prompts()).prompts
        finally:
            await app.shutdown()

    out: list[str] = [
        "# Tool contract",
        "",
        f"_Generated from the server definition (v{__version__}) by `distributed-video-qa-mcp contract`. "
        "Do not edit by hand; `tests/test_contract_doc.py` fails when this file is stale._",
        "",
        "## Conventions",
        "",
        "- **Typed results.** Every tool publishes an `outputSchema` and returns matching `structuredContent`, "
        "also serialized as the first text block. Results are validated against the schema before they are sent.",
        "- **Images.** `video_ask`, `video_frames` and `video_chapters` follow the JSON with caption + JPEG pairs, "
        "in the order the structured result lists them.",
        "- **Jobs.** Ingest and download return a `Job` at once. `wait_seconds` (≤ 300) blocks until it finishes "
        "or the time is up, streaming MCP progress notifications meanwhile; then poll `video_job`.",
        "- **Errors.** A failed call returns `isError: true` with the text "
        "`Error executing tool <tool>: <code>: <message> — <hint>`, plus ` Retryable.` when retrying can succeed. "
        "Codes are listed [below](#error-codes).",
        "- **Times** are seconds as numbers (`start`, `t`) with an `at`/`span` rendering as `m:ss` or `h:mm:ss`.",
        "",
        "## Tools",
        "",
        "| Group | Tool | Purpose | Hints |",
        "| --- | --- | --- | --- |",
    ]
    for group, _, names in GROUPS:
        for name in names:
            tool = tools[name]
            out.append(f"| {group} | [`{name}`](#{name}) | {_cell(tool.title)} | {_hints(tool)} |")
    missing = set(tools) - {n for _, _, names in GROUPS for n in names}
    if missing:
        raise RuntimeError(f"tools missing from contract groups: {sorted(missing)}")

    all_defs: dict[str, Any] = {}
    for group, blurb, names in GROUPS:
        out += ["", f"## {group}", "", blurb]
        for name in names:
            tool = tools[name]
            output = tool.output_schema or {}
            defs = output.get("$defs", {})
            all_defs.update(defs)
            model = output.get("title", "result")
            out += [
                "",
                f"### `{name}`",
                "",
                f"**{_cell(tool.title)}** · {_hints(tool)} · returns **`{model}`**",
                "",
                " ".join((tool.description or "").split()),
                "",
                *_fields(tool.input_schema, tool.input_schema.get("$defs", {}), inputs=True),
                "",
                f"Returns `{model}`" + (f": {output['description']}" if output.get("description") else ":"),
                "",
                *_fields(output, defs, inputs=False),
            ]

    names: list[str] = []
    for tool in tools.values():
        _referenced(tool.output_schema or {}, all_defs, names)
    if names:
        out += ["", "## Nested types"]
        for name in names:
            about = all_defs[name].get("description")
            out += [
                "",
                f"### `{name}`",
                "",
                *([about, ""] if about else []),
                *_fields(all_defs[name], all_defs, inputs=False),
            ]

    out += ["", "## Resources", "", "| URI | Type | Description |", "| --- | --- | --- |"]
    for item in [*resources, *templates]:
        uri = getattr(item, "uri", None) or item.uri_template
        out.append(f"| `{uri}` | `{item.mime_type}` | {_cell(item.description or item.title)} |")

    out += ["", "## Prompts", "", "| Prompt | Arguments | Description |", "| --- | --- | --- |"]
    for prompt in prompts:
        args = ", ".join(f"`{a.name}`" + ("" if a.required else "?") for a in prompt.arguments or [])
        out.append(f"| `{prompt.name}` | {args or '—'} | {_cell(prompt.description)} |")

    out += ["", "## Error codes", "", "| Code | Category | Retryable | What to do |", "| --- | --- | --- | --- |"]
    for code, spec in CODES.items():
        out.append(f"| `{code}` | {spec.category} | {'yes' if spec.retryable else 'no'} | {_cell(spec.hint)} |")
    return "\n".join(out) + "\n"
