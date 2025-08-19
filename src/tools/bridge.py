"""LangChain tool bridge for MCP tools"""

import asyncio
import json
from typing import Any, Dict, List, Type, Optional

from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, create_model
from pydantic import create_model as create_pydantic_model

from ..core.client import McpHttpClient
from .utils import unwrap_tool_result

class McpDelegatingTool(BaseTool):
    """LangChain tool that delegates to MCP server"""
    
    name: str
    description: str
    mcp: McpHttpClient
    _mcp_input_schema: Dict[str, Any]
    args_schema: Optional[type[BaseModel]] = None

    def _run(self, **kwargs) -> str:
        """Sync wrapper for async execution"""
        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Execute tool via MCP server"""
        try:
            resp = await self.mcp.call_tool(self.name, kwargs)
            unwrapped = unwrap_tool_result(resp)
            
            if isinstance(unwrapped, (dict, list)):
                return json.dumps(unwrapped, ensure_ascii=False, indent=2)
            else:
                return str(unwrapped)
                
        except Exception as e:
            return f"Error calling MCP tool '{self.name}': {str(e)}"

async def build_langchain_tools_from_mcp(mcp: McpHttpClient) -> List[BaseTool]:
    """Build LangChain tools from MCP server tools"""
    
    tools = await mcp.list_tools()
    lc_tools: List[BaseTool] = []

    for tool in tools:
        name = getattr(tool, "name", None) 
        description = getattr(tool, "description", "") 
        input_schema = getattr(tool, "inputSchema", None)
        
        # Create Pydantic model from schema using LangChain's pydantic v1
        if input_schema and isinstance(input_schema, dict):
            try:
                # Use LangChain's Pydantic v1 create_model
                properties = input_schema.get("properties", {})
                required = set(input_schema.get("required", []))
                
                field_definitions = {}
                for field_name, field_schema in properties.items():
                    field_type = field_schema.get("type", "string")
                    default_value = field_schema.get("default", ...)
                    
                    # Map JSON Schema types to Python types
                    type_mapping = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                        "array": list,
                        "object": dict
                    }
                    
                    python_type = type_mapping.get(field_type, str)
                    
                    # Handle optional vs required fields
                    if field_name not in required:
                        field_definitions[field_name] = (python_type, default_value if default_value is not ... else None)
                    else:
                        field_definitions[field_name] = (python_type, ...)
                
                args_model = create_model(f"{name}_Args", __base__=BaseModel, **field_definitions)
            except Exception as e:
                args_model = create_model(f"{name}_Args", __base__=BaseModel)
        else:
            args_model = create_model(f"{name}_Args", __base__=BaseModel)
        
        # Create tool instance
        tool_instance = McpDelegatingTool(
            name=name,
            description=description or f"MCP tool '{name}'",
            mcp=mcp,
            _mcp_input_schema=input_schema or {},
            args_schema=args_model
        )
        
        
        lc_tools.append(tool_instance)

    return lc_tools