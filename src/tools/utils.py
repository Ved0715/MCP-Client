"""Utility functions for tool processing"""

from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, create_model

def unwrap_tool_result(mcp_response: Any) -> Any:
    """Extract actual result from MCP response"""
    if hasattr(mcp_response, 'content'):
        content = mcp_response.content
        if isinstance(content, list) and len(content) > 0:
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
            if default_value is not ...:
                field_definitions[field_name] = (python_type, default_value)
            else:
                field_definitions[field_name] = (Optional[python_type], None)
        else:
            field_definitions[field_name] = (python_type, ...)
    
    return create_model(model_name, **field_definitions)