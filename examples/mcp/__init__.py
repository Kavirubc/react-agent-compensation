"""MCP Family Coordination Example.

Demonstrates MCP integration with automatic compensation/rollback.

Components:
- server.py: FastMCP server with 19 tools and compensation annotations
- client.py: LangChain agent using MCP tools with compensation
- database.py: MongoDB connection

To run:
    1. Start MongoDB: mongod
    2. Start MCP server: python -m examples.mcp.server
    3. Run client: python -m examples.mcp.client
"""
