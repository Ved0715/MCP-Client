"""
FastAPI routes for MCP client API

Provides REST endpoints for:
- Chat with MCP + LLM integration
- Session management with PostgreSQL persistence
- Tool discovery and direct calling
- System health monitoring
- Administrative functions
"""

import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
import asyncio
import json

from ..config.settings import Settings
from ..core.orchestrator import MCPLLMOrchestrator
from ..core.memory import PostgreSQLMemoryManager
from ..core.client import McpHttpClient
from .models import (
    ChatRequest, ChatResponse, SessionResponse, HistoryResponse,
    ErrorResponse, SystemStatusResponse, CreateSessionRequest,
    UpdateSessionRequest, DeleteSessionRequest, ToolCallRequest,
    ToolCallResponse, ToolInfo, SessionInfo, MessageInfo
)

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/api/v1", tags=["mcp-client"])

# Global state for orchestrators (in production, use Redis or similar)
active_orchestrators: Dict[str, MCPLLMOrchestrator] = {}
app_start_time = time.time()

# Security (optional)
security = HTTPBearer(auto_error=False)

# Dependency to get settings
def get_settings() -> Settings:
    """Get application settings"""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        raise HTTPException(
            status_code=500,
            detail="Configuration error. Please check environment variables."
        )

# Dependency for request logging
async def log_request(request: Request):
    """Log incoming requests for monitoring"""
    start_time = time.time()
    logger.info(f"📥 {request.method} {request.url.path} from {request.client.host}")
    
    yield
    
    processing_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} completed in {processing_time:.3f}s")

# Dependency to get or create orchestrator
async def get_orchestrator(
    session_id: str,
    user_id: str = "default",
    settings: Settings = Depends(get_settings)
) -> MCPLLMOrchestrator:
    """
    Get or create orchestrator for session with connection pooling
    """
    
    orchestrator_key = f"{user_id}:{session_id}"
    
    # Check if orchestrator already exists and is healthy
    if orchestrator_key in active_orchestrators:
        orchestrator = active_orchestrators[orchestrator_key]
        
        # Verify orchestrator is still healthy
        try:
            session_info = orchestrator.get_session_info()
            if session_info.get("initialized", False):
                logger.debug(f"Reusing existing orchestrator for {orchestrator_key}")
                return orchestrator
            else:
                logger.warning(f"Orchestrator {orchestrator_key} not initialized, recreating")
                # Remove unhealthy orchestrator
                try:
                    await orchestrator.close()
                except Exception:
                    pass
                del active_orchestrators[orchestrator_key]
        except Exception as e:
            logger.error(f"Health check failed for orchestrator {orchestrator_key}: {e}")
            # Remove unhealthy orchestrator
            try:
                await orchestrator.close()
            except Exception:
                pass
            del active_orchestrators[orchestrator_key]
    
    # Create new orchestrator
    logger.info(f"Creating new orchestrator for session {session_id}, user {user_id}")
    
    orchestrator = MCPLLMOrchestrator(
        settings=settings,
        session_id=session_id,
        user_id=user_id
    )
    
    try:
        await orchestrator.initialize()
        active_orchestrators[orchestrator_key] = orchestrator
        logger.info(f"✅ Orchestrator created and initialized for {orchestrator_key}")
        
        # Cleanup old orchestrators if we have too many
        await _cleanup_old_orchestrators()
        
        return orchestrator
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator for {orchestrator_key}: {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        
        # Cleanup failed orchestrator
        try:
            await orchestrator.close()
        except Exception:
            pass
            
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize chat session: {str(e)}"
        )

async def _cleanup_old_orchestrators(max_orchestrators: int = 10):
    """Clean up old orchestrators to prevent memory leaks"""
    
    if len(active_orchestrators) <= max_orchestrators:
        return
    
    logger.info(f"Cleaning up old orchestrators. Current count: {len(active_orchestrators)}")
    
    # Sort by last usage (would need to track this properly in production)
    # For now, just remove some random ones
    orchestrator_keys = list(active_orchestrators.keys())
    to_remove = orchestrator_keys[:-max_orchestrators]  # Keep the last N
    
    for key in to_remove:
        try:
            orchestrator = active_orchestrators[key]
            await orchestrator.close()
            del active_orchestrators[key]
            logger.debug(f"Cleaned up orchestrator: {key}")
        except Exception as e:
            logger.error(f"Error cleaning up orchestrator {key}: {e}")

# CHAT ENDPOINTS

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    _: None = Depends(log_request)
):
    """
    Main chat endpoint - processes user messages through MCP + LLM flow
    
    This endpoint:
    1. Creates or reuses an orchestrator for the session
    2. Processes the message through LLM + MCP tools
    3. Returns the response with metadata
    4. Automatically saves to PostgreSQL
    """
    
    start_time = time.time()
    
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid4())
    user_id = request.user_id or "anonymous"
    
    logger.info(f"💬 Chat request from user {user_id}, session {session_id}")
    logger.debug(f"Message preview: {request.message[:100]}...")
    
    try:
        # Get orchestrator for this session
        orchestrator = await get_orchestrator(session_id, user_id, settings)
        
        # Process the message through MCP + LLM
        logger.info("🔄 Processing message through MCP + LLM...")
        response_text = await orchestrator.process_user_query(request.message)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get tools that were used (simplified tracking)
        available_tools = orchestrator.get_available_tools()
        tools_used = [tool["name"] for tool in available_tools]  # In production, track actual usage
        
        # Create response
        response = ChatResponse(
            response=response_text,
            session_id=session_id,
            user_id=user_id,
            tools_used=tools_used,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"✅ Chat response generated in {processing_time:.2f}s")
        
        # Background task for analytics/cleanup
        background_tasks.add_task(_log_chat_analytics, user_id, session_id, processing_time)
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Chat endpoint error after {processing_time:.2f}s: {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat message: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    settings: Settings = Depends(get_settings)
):
    """
    Streaming chat endpoint for real-time responses
    """
    
    session_id = request.session_id or str(uuid4())
    user_id = request.user_id or "anonymous"
    
    logger.info(f"🌊 Streaming chat request from user {user_id}, session {session_id}")
    
    async def generate_response():
        try:
            # Get orchestrator
            orchestrator = await get_orchestrator(session_id, user_id, settings)
            
            # Start processing
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            
            # Process message (in production, implement actual streaming)
            response_text = await orchestrator.process_user_query(request.message)
            
            # Stream the response in chunks
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.1)  # Simulate streaming delay
            
            # End streaming
            yield f"data: {json.dumps({'type': 'end', 'session_id': session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# SESSION MANAGEMENT ENDPOINTS

@router.get("/sessions/{user_id}", response_model=SessionResponse)
async def get_user_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    settings: Settings = Depends(get_settings)
):
    """Get all sessions for a user with pagination"""
    
    logger.info(f"📋 Getting sessions for user {user_id} (limit={limit}, offset={offset})")
    
    try:
        memory_manager = PostgreSQLMemoryManager(settings)
        sessions_data = memory_manager.get_user_sessions(user_id)
        
        # Apply pagination
        paginated_sessions = sessions_data[offset:offset + limit]
        
        sessions = []
        for session in paginated_sessions:
            try:
                sessions.append(SessionInfo(
                    session_id=session["session_id"],
                    user_id=user_id,
                    title=session["title"],
                    created_at=datetime.fromisoformat(session["created_at"]) if session["created_at"] else datetime.now(),
                    updated_at=datetime.fromisoformat(session["updated_at"]) if session["updated_at"] else datetime.now(),
                    message_count=session["message_count"]
                ))
            except Exception as e:
                logger.warning(f"Failed to parse session data: {e}")
                continue
        
        logger.info(f"✅ Found {len(sessions)} sessions for user {user_id}")
        
        return SessionResponse(
            sessions=sessions,
            user_id=user_id,
            total_sessions=len(sessions_data)
        )
        
    except Exception as e:
        logger.error(f"❌ Get sessions error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user sessions: {str(e)}"
        )

@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_chat_history(
    session_id: str,
    limit: int = 100,
    settings: Settings = Depends(get_settings)
):
    """Get chat history for a specific session"""
    
    logger.info(f"📜 Getting history for session {session_id} (limit={limit})")
    
    try:
        memory_manager = PostgreSQLMemoryManager(settings)
        history_data = memory_manager.get_chat_history_for_frontend(session_id)
        
        # Apply limit
        limited_history = history_data[-limit:] if len(history_data) > limit else history_data
        
        history = []
        for msg in limited_history:
            try:
                history.append(MessageInfo(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"]
                ))
            except Exception as e:
                logger.warning(f"Failed to parse message data: {e}")
                continue
        
        logger.info(f"✅ Retrieved {len(history)} messages for session {session_id}")
        
        return HistoryResponse(
            history=history,
            session_id=session_id,
            total_messages=len(history_data)
        )
        
    except Exception as e:
        logger.error(f"❌ Get history error for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get chat history: {str(e)}"
        )

@router.post("/sessions", response_model=SessionInfo)
async def create_session(
    request: CreateSessionRequest,
    settings: Settings = Depends(get_settings)
):
    """Create a new chat session"""
    
    session_id = str(uuid4())
    
    logger.info(f"🆕 Creating new session {session_id} for user {request.user_id}")
    
    try:
        memory_manager = PostgreSQLMemoryManager(settings)
        
        # Create session metadata
        memory_manager._update_session_metadata(session_id, request.user_id)
        
        # Set title if provided
        title = request.title or "New Chat"
        if request.title:
            memory_manager.update_session_title(session_id, request.title)
        
        logger.info(f"✅ Created session {session_id} with title: {title}")
        
        return SessionInfo(
            session_id=session_id,
            user_id=request.user_id,
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            message_count=0
        )
        
    except Exception as e:
        logger.error(f"❌ Create session error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )

@router.put("/sessions/{session_id}")
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    settings: Settings = Depends(get_settings)
):
    """Update session information"""
    
    logger.info(f"📝 Updating session {session_id}")
    
    try:
        memory_manager = PostgreSQLMemoryManager(settings)
        
        updates_made = []
        
        if request.title:
            memory_manager.update_session_title(session_id, request.title)
            updates_made.append(f"title='{request.title}'")
        
        # In production, also handle metadata updates
        if request.metadata:
            logger.info(f"Metadata update requested but not implemented: {request.metadata}")
        
        logger.info(f"✅ Session {session_id} updated: {', '.join(updates_made)}")
        
        return {
            "message": "Session updated successfully",
            "session_id": session_id,
            "updates": updates_made
        }
        
    except Exception as e:
        logger.error(f"❌ Update session error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update session: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: DeleteSessionRequest,
    settings: Settings = Depends(get_settings)
):
    """Delete a chat session"""
    
    logger.info(f"🗑️ Deleting session {session_id} for user {request.user_id}")
    
    try:
        memory_manager = PostgreSQLMemoryManager(settings)
        
        # Delete from database
        memory_manager.delete_session(session_id)
        
        # Remove from active orchestrators
        orchestrator_key = f"{request.user_id}:{session_id}"
        if orchestrator_key in active_orchestrators:
            orchestrator = active_orchestrators[orchestrator_key]
            try:
                await orchestrator.close()
                logger.info(f"Closed active orchestrator for {orchestrator_key}")
            except Exception as e:
                logger.warning(f"Error closing orchestrator {orchestrator_key}: {e}")
            finally:
                del active_orchestrators[orchestrator_key]
        
        logger.info(f"✅ Session {session_id} deleted successfully")
        
        return {
            "message": "Session deleted successfully",
            "session_id": session_id,
            "deleted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Delete session error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

# TOOL MANAGEMENT ENDPOINTS

@router.get("/tools", response_model=List[ToolInfo])
async def get_available_tools(
    settings: Settings = Depends(get_settings)
):
    """Get list of available MCP tools"""
    
    logger.info("🔧 Getting available MCP tools")
    
    try:
        # Create temporary MCP client to discover tools
        async with McpHttpClient(settings) as mcp_client:
            tools = await mcp_client.list_tools()
            
            tool_infos = []
            for tool in tools:
                try:
                    name = getattr(tool, "name", None) 
                    description = getattr(tool, "description", "") 
                    schema = getattr(tool, "input_schema", None)
                    
                    tool_infos.append(ToolInfo(
                        name=name,
                        description=description,
                        schema=schema
                    ))
                except Exception as e:
                    logger.warning(f"Failed to parse tool data: {e}")
                    continue
            
            logger.info(f"✅ Found {len(tool_infos)} MCP tools")
            return tool_infos
        
    except Exception as e:
        logger.error(f"❌ Get tools error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available tools: {str(e)}"
        )

@router.post("/tools/call", response_model=ToolCallResponse)
async def call_tool_directly(
    request: ToolCallRequest,
    settings: Settings = Depends(get_settings)
):
    """Call a specific MCP tool directly without LLM involvement"""
    
    start_time = time.time()
    
    logger.info(f"🔧 Direct tool call: {request.tool_name} with args: {request.arguments}")
    
    try:
        # Create temporary MCP client
        async with McpHttpClient(settings) as mcp_client:
            result = await mcp_client.call_tool(request.tool_name, request.arguments)
            
            execution_time = time.time() - start_time
            
            logger.info(f"✅ Tool {request.tool_name} executed in {execution_time:.3f}s")
            
            return ToolCallResponse(
                result=result,
                tool_name=request.tool_name,
                execution_time_seconds=execution_time,
                success=True
            )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Direct tool call error for {request.tool_name}: {e}")
        
        return ToolCallResponse(
            result=None,
            tool_name=request.tool_name,
            execution_time_seconds=execution_time,
            success=False,
            error=str(e)
        )

# SYSTEM MONITORING ENDPOINTS

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    settings: Settings = Depends(get_settings)
):
    """Get system status and health information"""
    
    logger.info("🔍 Checking system status")
    
    try:
        # Test MCP server connection
        mcp_connected = False
        available_tools = []
        
        try:
            async with McpHttpClient(settings) as mcp_client:
                tools = await mcp_client.list_tools()
                mcp_connected = True
                
                for tool in tools:
                    try:
                        name = getattr(tool, "name", None) 
                        description = getattr(tool, "description", "") 
                        schema = getattr(tool, "input_schema", None)
                        
                        available_tools.append(ToolInfo(
                            name=name,
                            description=description,
                            schema=schema
                        ))
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.warning(f"MCP server connection failed: {e}")
            mcp_connected = False
        
        # Test database connection
        db_connected = False
        try:
            memory_manager = PostgreSQLMemoryManager(settings)
            # Try a simple query
            test_sessions = memory_manager.get_user_sessions("health_check_user")
            db_connected = True
            logger.debug(f"Database health check: found {len(test_sessions)} test sessions")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            db_connected = False
        
        # Calculate uptime
        uptime = time.time() - app_start_time
        
        # Determine overall status
        if mcp_connected and db_connected:
            status = "healthy"
        elif mcp_connected or db_connected:
            status = "degraded"
        else:
            status = "unhealthy"
        
        logger.info(f"System status: {status} (MCP: {mcp_connected}, DB: {db_connected})")
        
        return SystemStatusResponse(
            status=status,
            mcp_server_connected=mcp_connected,
            database_connected=db_connected,
            available_tools=available_tools,
            version="1.0.0",  # Should come from package info
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )

@router.get("/metrics")
async def get_system_metrics():
    """Get detailed system metrics for monitoring"""
    
    try:
        uptime = time.time() - app_start_time
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "active_orchestrators": len(active_orchestrators),
            "orchestrator_sessions": list(active_orchestrators.keys()),
            "memory_usage": {
                # In production, add actual memory usage metrics
                "orchestrators_count": len(active_orchestrators)
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

# ADMINISTRATIVE ENDPOINTS

@router.post("/admin/cleanup")
async def cleanup_orchestrators(
    force: bool = False,
    settings: Settings = Depends(get_settings)
):
    """Clean up all active orchestrators (admin only)"""
    
    logger.info(f"🧹 Starting orchestrator cleanup (force={force})")
    
    try:
        cleanup_count = 0
        errors = []
        
        orchestrator_items = list(active_orchestrators.items())
        
        for orchestrator_key, orchestrator in orchestrator_items:
            try:
                logger.debug(f"Closing orchestrator: {orchestrator_key}")
                await orchestrator.close()
                del active_orchestrators[orchestrator_key]
                cleanup_count += 1
            except Exception as e:
                error_msg = f"Error cleaning up orchestrator {orchestrator_key}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                if force:
                    # Force remove even if cleanup failed
                    del active_orchestrators[orchestrator_key]
                    cleanup_count += 1
        
        result = {
            "message": f"Cleaned up {cleanup_count} orchestrators",
            "remaining_active": len(active_orchestrators),
            "cleanup_errors": errors,
            "force_cleanup": force
        }
        
        logger.info(f"✅ Cleanup completed: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup orchestrators: {str(e)}"
        )

@router.post("/admin/reset")
async def reset_system(
    confirm: bool = False,
    settings: Settings = Depends(get_settings)
):
    """Reset system state (admin only - use with caution)"""
    
    if not confirm:
        return {
            "message": "Reset not performed. Set confirm=true to proceed.",
            "warning": "This will close all active sessions and clear system state."
        }
    
    logger.warning("🔄 SYSTEM RESET INITIATED")
    
    try:
        # Close all orchestrators
        cleanup_result = await cleanup_orchestrators(force=True, settings=settings)
        
        # Reset global state
        global app_start_time
        app_start_time = time.time()
        
        logger.warning("⚠️ System reset completed")
        
        return {
            "message": "System reset completed",
            "reset_timestamp": datetime.now().isoformat(),
            "cleanup_result": cleanup_result
        }
        
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset system: {str(e)}"
        )

# BACKGROUND TASKS

async def _log_chat_analytics(user_id: str, session_id: str, processing_time: float):
    """Background task for logging analytics"""
    try:
        # In production, send to analytics service
        logger.info(f"📊 Analytics: user={user_id}, session={session_id}, time={processing_time:.3f}s")
    except Exception as e:
        logger.error(f"Analytics logging error: {e}")

# ERROR HANDLERS

# @router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper logging"""
    
    request_id = str(uuid4())
    
    logger.error(f"🚨 Unhandled exception [{request_id}]: {exc}")
    logger.error(f"Request: {request.method} {request.url}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=request_id
        ).dict()
    )

# @router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    
    request_id = str(uuid4())
    
    logger.warning(f"⚠️ HTTP Exception [{request_id}]: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException",
            request_id=request_id
        ).dict()
    )

# HEALTH CHECK (simple endpoint for load balancers)
@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Root endpoint
@router.get("/")
async def root():
    """API root endpoint with information"""
    return {
        "name": "MCP Client API",
        "version": "1.0.0",
        "description": "REST API for MCP client with LangChain integration",
        "endpoints": {
            "chat": "/api/v1/chat",
            "sessions": "/api/v1/sessions/{user_id}",
            "tools": "/api/v1/tools",
            "status": "/api/v1/status",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }