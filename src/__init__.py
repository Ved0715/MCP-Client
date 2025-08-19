"""
MCP Client Package

A production-ready Model Context Protocol client with LangChain integration
and PostgreSQL memory persistence.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core exports
from .core.client import McpHttpClient
from .core.memory import PostgreSQLMemoryManager
from .core.orchestrator import MCPLLMOrchestrator
from .config.settings import Settings
from .tools.bridge import McpDelegatingTool, build_langchain_tools_from_mcp

# API exports
from .api.models import (
    ChatRequest,
    ChatResponse,
    SessionResponse,
    HistoryResponse,
    ErrorResponse
)

__all__ = [
    # Core classes
    "McpHttpClient",
    "PostgreSQLMemoryManager", 
    "MCPLLMOrchestrator",
    "Settings",
    
    # Tools
    "McpDelegatingTool",
    "build_langchain_tools_from_mcp",
    
    # API models
    "ChatRequest",
    "ChatResponse", 
    "SessionResponse",
    "HistoryResponse",
    "ErrorResponse",
]

# Package metadata
PACKAGE_INFO = {
    "name": "mcp-client",
    "version": __version__,
    "description": "MCP client with LangChain integration and PostgreSQL persistence",
    "author": __author__,
    "email": __email__,
    "requires_python": ">=3.8",
}