import anyio
from mcp import Client
from mcp.types import TextContent

from app.workflow_engine.mcp_server import mcp


def test_mcp_server_exposes_the_same_read_only_tools():
    async def inspect_server():
        async with Client(mcp) as client:
            listed = await client.list_tools()
            tool_names = {tool.name for tool in listed.tools}
            result = await client.call_tool(
                "search_pricing_documents",
                {"query": "Northstar Labs pricing"},
            )
            text_content = [
                block.text
                for block in result.content
                if isinstance(block, TextContent)
            ]
            return tool_names, result.is_error, text_content

    tool_names, is_error, text_content = anyio.run(inspect_server)

    assert tool_names == {
        "search_pricing_documents",
        "check_calendar_availability",
    }
    assert is_error is False
    assert any("mock-pricing-001" in text for text in text_content)
