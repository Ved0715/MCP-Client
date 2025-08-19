"""
API Layer

FastAPI routes and models for HTTP API access to MCP client functionality.
Provides REST endpoints for chat, session management, and tool access.
"""

from .models import (
    ChatRequest,
    ChatResponse,
    SessionResponse,
    HistoryResponse,
    ErrorResponse,
    ToolInfo,
    SessionInfo
)
from .routes import router

__all__ = [
    "ChatRequest",
    "ChatResponse", 
    "SessionResponse",
    "HistoryResponse",
    "ErrorResponse",
    "ToolInfo",
    "SessionInfo",
    "router",
]