from mcp.server import MCPServer

from app.workflow_engine.mock_tools import (
    check_calendar_availability as check_local_calendar,
    search_pricing_documents as search_local_pricing_documents,
)


mcp = MCPServer(
    "workflow-ai-read-only-tools",
    instructions=(
        "These tools read a small fictional local dataset. "
        "They do not access or change live business systems."
    ),
)


@mcp.tool(title="Search fictional pricing documents")
def search_pricing_documents(query: str, limit: int = 3) -> str:
    """Search the local fictional pricing examples by terms."""
    return search_local_pricing_documents(query, limit)


@mcp.tool(title="Check fictional calendar availability")
def check_calendar_availability(query: str) -> str:
    """Find sample open slots matching a date or time of day."""
    return check_local_calendar(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
