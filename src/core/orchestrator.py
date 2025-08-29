"""
MCP + LLM Orchestrator

Coordinates the complete flow between user queries, LLM reasoning,
MCP tool execution, and response generation with persistent memory.
"""

import uuid
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain.agents import initialize_agent, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.llms.base import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

from ..config.settings import Settings
from ..core.client import McpHttpClient
from ..core.memory import PostgreSQLMemoryManager
from ..tools.bridge import build_langchain_tools_from_mcp

logger = logging.getLogger(__name__)

class MCPLLMOrchestrator:
    """
    Orchestrates the complete MCP + LLM flow with PostgreSQL memory
    
    This class implements the 7-step flow:
    1. User query received
    2. LLM analyzes query with available tools
    3. LLM decides which tools to use
    4. MCP tools are called
    5. Tool results are processed
    6. LLM synthesizes final response
    7. Response returned to user (with persistence)
    """
    
    def __init__(
        self, 
        settings: Settings, 
        session_id: Optional[str] = None, 
        user_id: Optional[str] = None
    ):
        self.settings = settings
        self.session_id = session_id or str(uuid.uuid4())
        self.user_id = user_id or "default_user"
        
        # Core components
        self.memory_manager = PostgreSQLMemoryManager(settings)
        self.mcp_client: Optional[McpHttpClient] = None
        self.llm: Optional[BaseLanguageModel] = None
        self.agent: Optional[AgentExecutor] = None
        self.tools: List[BaseTool] = []
        self.memory: Optional[ConversationBufferMemory] = None
        
        # State tracking
        self._initialized = False
        self._tool_call_count = 0
        
        logger.info(f"Created orchestrator for session {self.session_id}, user {self.user_id}")
        
    async def initialize(self) -> None:
        """
        Initialize all components in the correct order
        
        Raises:
            RuntimeError: If initialization fails
            ValueError: If required settings are missing
        """
        
        if self._initialized:
            logger.warning("Orchestrator already initialized")
            return
            
        logger.info(f"Initializing MCP + LLM system for session: {self.session_id}")
        
        try:
            # Step 1: Validate settings
            self.settings.validate()
            logger.info("✅ Settings validated")
            
            # Step 2: Initialize MCP client
            await self._initialize_mcp_client()
            
            # Step 3: Discover and create tools
            await self._initialize_tools()
            
            # Step 4: Initialize LLM
            await self._initialize_llm()
            
            # Step 5: Setup PostgreSQL memory
            await self._initialize_memory()
            
            # Step 6: Create agent
            await self._initialize_agent()
            
            # Step 7: Load conversation context
            await self._load_conversation_context()
            
            self._initialized = True
            logger.info("🎉 MCP + LLM system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.close()  # Cleanup on failure
            raise RuntimeError(f"Initialization failed: {e}") from e
    
    async def _initialize_mcp_client(self) -> None:
        """Initialize MCP client connection"""
        logger.info("🔌 Connecting to MCP server...")
        
        self.mcp_client = McpHttpClient(self.settings)
        await self.mcp_client.connect()
        
        logger.info(f"✅ Connected to MCP server: {self.settings.mcp_server_url}")
    
    async def _initialize_tools(self) -> None:
        """Discover and create LangChain tools from MCP server"""
        logger.info("🔍 Building LangChain tools from MCP...")
        
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized")
            
        self.tools = await build_langchain_tools_from_mcp(self.mcp_client)
        
        if not self.tools:
            logger.warning("⚠️ No tools found! Make sure MCP server has tools available.")
        else:
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"✅ Created {len(self.tools)} tools: {tool_names}")
    
    async def _initialize_llm(self) -> None:
        """Initialize the language model"""
        logger.info("🧠 Initializing LLM...")
        
        try:
            self.llm = ChatOpenAI(
                model=self.settings.llm_model_name,
                temperature=self.settings.llm_temperature,
                api_key=self.settings.openai_api_key,
                request_timeout=60,  # Reasonable timeout
                max_retries=2,  # Retry failed requests
            )
            
            # Test LLM connection with proper async handling
            try:
                from langchain.schema import HumanMessage
                test_messages = [HumanMessage(content="Hello")]
                test_response = await self.llm.agenerate([test_messages])
                
                # Check if response is valid
                if test_response and test_response.generations:
                    generation = test_response.generations[0][0]
                    if hasattr(generation, 'text'):
                        response_text = generation.text
                    elif hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                        response_text = generation.message.content
                    else:
                        response_text = str(generation)
                    
                    logger.info(f"✅ LLM initialized and tested: {self.settings.llm_model_name}")
                    logger.debug(f"Test response: {response_text[:100]}...")
                else:
                    logger.warning("LLM test returned empty response, but connection seems OK")
                    
            except Exception as test_error:
                logger.warning(f"LLM test failed but proceeding: {test_error}")
                # Don't fail initialization just because test failed
                
            logger.info(f"✅ LLM initialized: {self.settings.llm_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e
    
    async def _initialize_memory(self) -> None:
        """Setup PostgreSQL-backed memory"""
        logger.info("🗄️ Setting up PostgreSQL memory...")
        
        try:
            self.memory = self.memory_manager.create_memory_for_session(
                session_id=self.session_id,
                user_id=self.user_id
            )
            logger.info("✅ PostgreSQL memory initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory: {e}")
            raise RuntimeError(f"Memory initialization failed: {e}") from e
    
    async def _initialize_agent(self) -> None:
        """Create LangChain agent with tools and memory"""
        logger.info("🤖 Creating LangChain agent...")
        
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        if not self.memory:
            raise RuntimeError("Memory not initialized")
            
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,  # Best for tool use
                verbose=True,  # Shows reasoning steps
                memory=self.memory,  # PostgreSQL-backed memory
                handle_parsing_errors=True,  # Graceful error handling
                max_iterations=10,  # Allow multiple tool calls
                early_stopping_method="force",  # Allow multiple iterations before stopping
                return_intermediate_steps=True,  # For debugging
            )
            
            logger.info("✅ Agent created with PostgreSQL memory and tools")
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e
    
    async def _load_conversation_context(self) -> None:
        """Load and display existing conversation context"""
        try:
            history_messages = self.memory_manager.get_chat_history_for_frontend(self.session_id)
            
            if history_messages:
                logger.info(f"💭 Loaded {len(history_messages)} previous messages")
                
                # Log recent context for debugging
                recent_messages = history_messages[-3:] if len(history_messages) > 3 else history_messages
                for msg in recent_messages:
                    role = "👤" if msg["role"] == "human" else "🤖"
                    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    logger.debug(f"   Context: {role} {content}")
            else:
                logger.info("💭 Starting new conversation (no previous history)")
                
        except Exception as e:
            logger.warning(f"Failed to load conversation context: {e}")
    
    async def process_user_query(self, user_query: str) -> str:
        """
        Process user query through the complete MCP + LLM flow
        
        Args:
            user_query: The user's input message
            
        Returns:
            The LLM's response after potential tool usage
            
        Raises:
            RuntimeError: If system not initialized or processing fails
        """
        
        if not self._initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
            
        if not user_query.strip():
            return "Please provide a valid query."
            
        logger.info(f"👤 Processing user query: {user_query}")
        start_time = datetime.now()
        
        try:
            # Reset tool call counter for this query
            self._tool_call_count = 0
            
            # Get chat history from memory to provide context
            chat_history = ""
            if self.memory:
                chat_history = self.memory.buffer
                
            # Process through agent with context (this handles the 7-step flow automatically)
            if chat_history:
                contextualized_query = f"Previous conversation:\n{chat_history}\n\nCurrent question: {user_query}"
            else:
                contextualized_query = user_query
                
            result = await self.agent.ainvoke({"input": contextualized_query})
            response = result["output"]
            
            # Manually save to memory if agent didn't do it automatically
            if self.memory:
                self.memory.save_context({"input": user_query}, {"output": response})
            
            # Log processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"🤖 Query processed in {processing_time:.2f}s with {self._tool_call_count} tool calls")
            
            # Auto-generate session title if this is the first meaningful exchange
            await self._maybe_generate_session_title(user_query)
            
            # Log final response
            logger.info(f"🤖 Final response: {response[:200]}...")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            
            # Return user-friendly error message
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _maybe_generate_session_title(self, first_query: str) -> None:
        """Generate a session title based on the first query"""
        try:
            sessions = self.memory_manager.get_user_sessions(self.user_id)
            current_session = next(
                (s for s in sessions if s["session_id"] == self.session_id), 
                None
            )
            
            # Generate title if session is new or untitled
            if (current_session and 
                current_session.get("message_count", 0) <= 2 and  # First exchange
                (current_session.get("title") == "Untitled Chat" or not current_session.get("title"))):
                
                # Create a meaningful title from the first query
                title = first_query.strip()
                if len(title) > 50:
                    title = title[:47] + "..."
                
                self.memory_manager.update_session_title(self.session_id, title)
                logger.info(f"📝 Generated session title: {title}")
                
        except Exception as e:
            logger.warning(f"Failed to generate session title: {e}")
    
    def increment_tool_call_count(self) -> None:
        """Track tool usage for analytics"""
        self._tool_call_count += 1
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get chat history for frontend display"""
        try:
            return self.memory_manager.get_chat_history_for_frontend(self.session_id)
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            return []
    
    def get_user_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions for current user"""
        try:
            return self.memory_manager.get_user_sessions(self.user_id)
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            return []
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools for display"""
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools
        ]
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "initialized": self._initialized,
            "tools_count": len(self.tools),
            "llm_model": self.settings.llm_model_name,
            "mcp_server_url": self.settings.mcp_server_url,
        }
    
    async def close(self) -> None:
        """Clean shutdown of all components"""
        logger.info("🔄 Shutting down orchestrator...")
        
        try:
            if self.mcp_client:
                await self.mcp_client.close()
                logger.info("🔌 Disconnected from MCP server")
            
            # Clear references
            self.agent = None
            self.llm = None
            self.memory = None
            self.tools = []
            
            logger.info(f"💾 Session {self.session_id} data saved to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        finally:
            self._initialized = False
            logger.info("✅ Orchestrator shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()