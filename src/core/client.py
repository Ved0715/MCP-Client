"""MCP HTTP Client implementation"""

import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator
from contextlib import asynccontextmanager

try:
    from fastmcp.client.transports import StreamableHttpTransport
    from fastmcp import Client
except ImportError:
    raise ImportError("fastmcp is required. Install with: pip install fastmcp")

from ..config.settings import Settings

class McpHttpClient:
    """HTTP-based MCP client with connection management"""
    
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.transport = StreamableHttpTransport(
            url=settings.mcp_server_url,
        )
        self.client = Client(self.transport)
        self._connected = False

    async def __aenter__(self) -> "McpHttpClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Establish connection to MCP server"""
        if self._connected:
            return
        await self.client.__aenter__()
        await self.client.ping()
        self._connected = True

    async def close(self) -> None:
        """Close connection to MCP server"""
        if not self._connected:
            return
        self._connected = False
        await self.client.__aexit__(None, None, None)

    def _require_connection(self) -> None:
        """Ensure client is connected"""
        if not self._connected:
            raise RuntimeError("MCP client not connected. Use 'async with McpHttpClient(...)'")

    # Discovery methods
    async def list_tools(self) -> List[Any]:
        """List available tools"""
        self._require_connection()
        return await self.client.list_tools()

    async def list_resources(self) -> List[Any]:
        """List available resources"""
        self._require_connection()
        return await self.client.list_resources()

    async def list_prompts(self) -> List[Any]:
        """List available prompts"""
        self._require_connection()
        return await self.client.list_prompts()

    # Operation methods
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool"""
        self._require_connection()
        return await self.client.call_tool(name, arguments)

    async def read_resource(self, uri: str) -> Any:
        """Read a resource"""
        self._require_connection()
        return await self.client.read_resource(uri)

    async def get_prompt(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Get a prompt"""
        self._require_connection()
        return await self.client.get_prompt(name, params or {})

    async def stream_tool(self, name: str, arguments: Dict[str, Any]) -> AsyncIterator[Any]:
        """Stream tool results"""
        self._require_connection()
        async for msg in self.client.stream_tool(name, arguments):
            yield msg