"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    @validator('message')
    def message_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in this response")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")
    
class ToolInfo(BaseModel):
    """Information about an available tool"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    schema: Optional[Dict[str, Any]] = Field(None, description="Tool input schema")

class SessionInfo(BaseModel):
    """Information about a chat session"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    title: str = Field(..., description="Session title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    message_count: int = Field(..., description="Number of messages in session")

class SessionResponse(BaseModel):
    """Response model for session operations"""
    sessions: List[SessionInfo] = Field(..., description="List of user sessions")
    user_id: str = Field(..., description="User ID")
    total_sessions: int = Field(..., description="Total number of sessions")

class MessageInfo(BaseModel):
    """Information about a chat message"""
    role: str = Field(..., description="Message role (human/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")

class HistoryResponse(BaseModel):
    """Response model for chat history"""
    history: List[MessageInfo] = Field(..., description="Chat history messages")
    session_id: str = Field(..., description="Session ID")
    total_messages: int = Field(..., description="Total number of messages")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for tracking")

class SystemStatusResponse(BaseModel):
    """System status response"""
    status: str = Field(..., description="System status")
    mcp_server_connected: bool = Field(..., description="MCP server connection status")
    database_connected: bool = Field(..., description="Database connection status")
    available_tools: List[ToolInfo] = Field(..., description="Available MCP tools")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Application uptime in seconds")

class CreateSessionRequest(BaseModel):
    """Request to create a new session"""
    user_id: str = Field(..., description="User identifier")
    title: Optional[str] = Field(None, description="Optional session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional session metadata")

class UpdateSessionRequest(BaseModel):
    """Request to update session information"""
    title: Optional[str] = Field(None, description="New session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")

class DeleteSessionRequest(BaseModel):
    """Request to delete a session"""
    session_id: str = Field(..., description="Session ID to delete")
    user_id: str = Field(..., description="User ID for authorization")

class ToolCallRequest(BaseModel):
    """Request to call a specific tool directly"""
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    session_id: Optional[str] = Field(None, description="Optional session context")

class ToolCallResponse(BaseModel):
    """Response from direct tool call"""
    result: Any = Field(..., description="Tool execution result")
    tool_name: str = Field(..., description="Tool that was called")
    execution_time_seconds: float = Field(..., description="Execution time")
    success: bool = Field(..., description="Whether the call succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")