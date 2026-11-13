from __future__ import annotations

from pathlib import Path

from distributed_video_qa_mcp.contract import render

DOC = Path(__file__).resolve().parents[1] / "docs" / "tool-contract.md"


async def test_tool_contract_is_current():
    expected = await render()
    assert DOC.read_text() == expected, (
        "docs/tool-contract.md is stale: run `distributed-video-qa-mcp contract -o docs/tool-contract.md`"
    )


async def test_every_argument_and_field_is_described(settings):
    from mcp.client import Client

    from distributed_video_qa_mcp.server import build_server

    mcp, app = build_server(settings)
    missing = []
    try:
        async with Client(mcp) as client:
            for tool in (await client.list_tools()).tools:
                for name, prop in tool.input_schema.get("properties", {}).items():
                    if not prop.get("description"):
                        missing.append(f"{tool.name}({name})")
                schemas = [("", tool.output_schema or {})]
                schemas += list((tool.output_schema or {}).get("$defs", {}).items())
                for owner, schema in schemas:
                    for name, prop in schema.get("properties", {}).items():
                        if not prop.get("description"):
                            missing.append(f"{tool.name} -> {owner or 'result'}.{name}")
    finally:
        await app.shutdown()
    assert not missing, sorted(set(missing))
