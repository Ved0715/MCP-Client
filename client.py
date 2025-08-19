import asyncio
from typing import Any, Dict, List, Optional, Sequence, AsyncIterator, Union, Tuple, Callable
from contextlib import AsyncExitStack
from fastmcp import Client
from mcp.client.stdio import stdio_client
import os 
from dotenv import load_dotenv
import httpx
from dataclasses import dataclass


try:
    from fastmcp.client.transports import StreamableHttpTransport
    print("StreamableHttpTransport found")
except ImportError:
    StreamableHttpTransport = None
    print("StreamableHttpTransport not found")






from typing import Optional

@dataclass
class Settings:
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
    bearer_token: Optional[str] = os.getenv("MCP_AUTH_BEARER")  # optional if behind gateway
    http_connect_timeout_s: float = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "10"))
    http_request_timeout_s: float = float(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "60"))

    # LLM (choose your provider; here we use OpenAI via langchain-openai)
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # OpenAI key for ChatOpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")




# Add these to your existing code

import json
from pydantic import BaseModel, create_model
from typing import Type
import logging

def unwrap_tool_result(mcp_response: Any) -> Any:
    """Extract the actual result from MCP response"""
    if hasattr(mcp_response, 'content'):
        # If it's a structured response with content
        content = mcp_response.content
        if isinstance(content, list) and len(content) > 0:
            # Take first content item if it's a list
            first_item = content[0]
            if hasattr(first_item, 'text'):
                return first_item.text
            elif isinstance(first_item, dict) and 'text' in first_item:
                return first_item['text']
            return first_item
        return content
    elif hasattr(mcp_response, 'result'):
        return mcp_response.result
    elif isinstance(mcp_response, dict):
        # Try common result keys
        for key in ['result', 'content', 'data', 'output']:
            if key in mcp_response:
                return mcp_response[key]
        return mcp_response
    else:
        return mcp_response

def json_schema_to_pydantic_model(model_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    """Convert JSON Schema to Pydantic model"""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    
    field_definitions = {}
    
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "string")
        default_value = field_schema.get("default", ...)
        
        # Map JSON Schema types to Python types
        if field_type == "string":
            python_type = str
        elif field_type == "integer":
            python_type = int
        elif field_type == "number":
            python_type = float
        elif field_type == "boolean":
            python_type = bool
        elif field_type == "array":
            python_type = list
        elif field_type == "object":
            python_type = dict
        else:
            python_type = str
        
        # Handle optional vs required fields
        if field_name not in required:
            if default_value is not ...:
                field_definitions[field_name] = (python_type, default_value)
            else:
                field_definitions[field_name] = (Optional[python_type], None)
        else:
            field_definitions[field_name] = (python_type, ...)
    
    return create_model(model_name, **field_definitions)




class McpHttpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.transport = StreamableHttpTransport(
            url=settings.mcp_server_url,
            # headers={"Authorization": f"Bearer {settings.bearer_token}"} if settings.bearer_token else None,
            # connect_timeout=settings.http_connect_timeout_s,
            # request_timeout=settings.http_request_timeout_s,
        )
        self.client = Client(self.transport)
        self._connected = False

    async def __aenter__(self) -> "McpHttpClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._connected:
            return
        await self.client.__aenter__()   # FastMCP handles initialize internally
        await self.client.ping()         # verify connectivity
        self._connected = True

    async def close(self) -> None:
        if not self._connected:
            return
        self._connected = False
        await self.client.__aexit__(None, None, None)

    def _require(self) -> None:
        if not self._connected:
            raise RuntimeError("MCP client not connected; use 'async with McpHttpClient(...)'")

    # Discovery
    async def list_tools(self):
        self._require()
        return await self.client.list_tools()

    async def list_resources(self):
        self._require()
        return await self.client.list_resources()

    async def list_prompts(self):
        self._require()
        return await self.client.list_prompts()

    # Operations
    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        self._require()
        return await self.client.call_tool(name, arguments)

    async def read_resource(self, uri: str):
        self._require()
        return await self.client.read_resource(uri)

    async def get_prompt(self, name: str, params: Optional[Dict[str, Any]] = None):
        self._require()
        return await self.client.get_prompt(name, params or {})

    async def stream_tool(self, name: str, arguments: Dict[str, Any]) -> AsyncIterator[Any]:
        self._require()
        async for msg in self.client.stream_tool(name, arguments):
            yield msg









# class McpClient:
#     def __init__(
#         self,
#         server_url: str,
#         # auth_header: Optional[str] = None,
#         # connect_timeout: float = 10.0,
#         # request_timeout: float = 60.0,
#         # extra_headers: Optional[Dict[str, str]] = None,
#         # transport_kwargs: Optional[dict] = None,
#         ) -> None:

#         self.session: Optional[Client] = None
#         self.exit_stack = AsyncExitStack()
#         self.server_url = server_url

#     async def connect_to_server(self):
#         if self.session is not None:
#             return self.session

#         load_dotenv(override=False)

#         transporter = StreamableHttpTransport(self.server_url)
#         session = Client(transporter)
#         # Open the client context via our exit stack (handles init/cleanup)
#         await self.exit_stack.enter_async_context(session)

#         # Optional connectivity check
#         await session.ping()

#         self.session = session
#         return session
    
#     async def disconnect_from_server(self) -> None:
#         if self.session is None:
#             return
#         self.session = None
#         await self.exit_stack.aclose()
#         self.exit_stack = AsyncExitStack()

#     async def __aenter__(self):
#         return await self.connect_to_server()
    
#     async def __aexit__(self, exc_type, exc_value, traceback):
#         await self.disconnect_from_server()



           

# async def main():
#     async with McpClient("http://localhost:8000/mcp") as client:
#         print("Connected to server")
#         tools = await client.list_tools()
#         print([t.name for t in tools])

# if __name__ == "__main__":
#     asyncio.run(main())
   