"""PostgreSQL memory management with better import handling"""

import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from sqlalchemy import create_engine, text

# Try different PostgreSQL chat history implementations
try:
    from langchain_postgres import PostgresChatMessageHistory
    POSTGRES_IMPORT_TYPE = "langchain_postgres"
except ImportError:
    try:
        from langchain_community.chat_message_histories import PostgresChatMessageHistory
        POSTGRES_IMPORT_TYPE = "langchain_community"
    except ImportError:
        PostgresChatMessageHistory = None
        POSTGRES_IMPORT_TYPE = "none"

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class PostgreSQLMemoryManager:
    """Manages PostgreSQL-backed conversation memory with Neon support"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection_string = settings.postgres_connection_string
        
        # Ensure SSL mode for Neon
        if "sslmode=" not in self.connection_string.lower():
            separator = "&" if "?" in self.connection_string else "?"
            self.connection_string += f"{separator}sslmode=require"
        
        logger.info(f"🔗 Using PostgreSQL import: {POSTGRES_IMPORT_TYPE}")
        logger.info(f"🔗 Connecting to Neon PostgreSQL")
        self._ensure_tables_exist()
    
    def _ensure_tables_exist(self) -> None:
        """Create necessary database tables with Neon-specific optimizations"""
        try:
            # Neon-optimized engine configuration
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=5,
                max_overflow=10,
                connect_args={
                    "sslmode": "require",
                    "connect_timeout": 30,
                    "application_name": "mcp_client"
                }
            )
            
            with engine.connect() as conn:
                # Create message_store table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS message_store (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        message JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))
                
                # Create index for performance
                conn.execute(text("""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes 
                            WHERE indexname = 'idx_message_store_session_id'
                        ) THEN
                            CREATE INDEX idx_message_store_session_id 
                            ON message_store (session_id);
                        END IF;
                    END $$;
                """))
                
                # Create sessions table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255),
                        title VARCHAR(500),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    );
                """))
                
                conn.commit()
                logger.info("✅ Neon PostgreSQL tables verified/created successfully")
                
        except Exception as e:
            logger.error(f"❌ Neon PostgreSQL setup failed: {e}")
            raise Exception(f"Failed to connect to Neon PostgreSQL: {e}")
    
    def create_memory_for_session(
        self, 
        session_id: str, 
        user_id: Optional[str] = None
    ) -> ConversationBufferMemory:
        """Create memory with Neon PostgreSQL backend"""
        
        try:
            # Update session metadata first
            self._update_session_metadata(session_id, user_id)
            
            # Create memory based on available implementation
            if PostgresChatMessageHistory is not None:
                # Use PostgreSQL chat history
                history = PostgresChatMessageHistory(
                    connection_string=self.connection_string,
                    session_id=session_id,
                    table_name="message_store"
                )
                
                memory = ConversationBufferWindowMemory(
                    k=10,  # Keep only last 10 message exchanges
                    chat_memory=history,
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="input",
                    output_key="output"
                )
                
                logger.info(f"🧠 Created Neon PostgreSQL memory for session: {session_id}")
            else:
                # Fallback to simple memory if PostgreSQL import fails
                logger.warning("⚠️ PostgreSQL chat history not available, using simple memory")
                memory = ConversationBufferWindowMemory(
                    k=10,  # Keep only last 10 message exchanges
                    memory_key="chat_history",
                    return_messages=True,
                    input_key="input",
                    output_key="output"
                )
                logger.info(f"🧠 Created fallback memory for session: {session_id}")
            
            return memory
            
        except Exception as e:
            logger.error(f"❌ Failed to create memory for session {session_id}: {e}")
            # Create fallback memory
            logger.warning("⚠️ Creating fallback memory due to PostgreSQL error")
            memory = ConversationBufferWindowMemory(
                k=10,  # Keep only last 10 message exchanges
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
            return memory
    
    def _update_session_metadata(self, session_id: str, user_id: Optional[str] = None) -> None:
        """Update session metadata in Neon database"""
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                connect_args={
                    "sslmode": "require",
                    "connect_timeout": 30
                }
            )
            
            with engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO chat_sessions (session_id, user_id, updated_at)
                    VALUES (:session_id, :user_id, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) 
                    DO UPDATE SET 
                        updated_at = CURRENT_TIMESTAMP,
                        user_id = COALESCE(:user_id, chat_sessions.user_id)
                """), {
                    "session_id": session_id,
                    "user_id": user_id
                })
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to update session metadata: {e}")
    
    def get_chat_history_for_frontend(self, session_id: str) -> List[Dict[str, Any]]:
        """Get formatted chat history from Neon"""
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                connect_args={"sslmode": "require", "connect_timeout": 30}
            )
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT message, created_at 
                    FROM message_store 
                    WHERE session_id = :session_id 
                    ORDER BY created_at ASC
                    LIMIT 1000
                """), {"session_id": session_id})
                
                messages = []
                
                for row in result:
                    try:
                        message_data = row[0]
                        created_at = row[1]
                        
                        if isinstance(message_data, dict):
                            role = message_data.get("type", "unknown")
                            content = ""
                            
                            if "data" in message_data:
                                content = message_data["data"].get("content", "")
                            elif "content" in message_data:
                                content = message_data["content"]
                            else:
                                content = str(message_data)
                        else:
                            role = "unknown"
                            content = str(message_data)
                        
                        formatted_message = {
                            "role": role,
                            "content": content,
                            "timestamp": created_at.isoformat() if created_at else None
                        }
                        messages.append(formatted_message)
                        
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse message: {parse_error}")
                        continue
                
                return messages
                
        except Exception as e:
            logger.error(f"Failed to get chat history from Neon: {e}")
            return []
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user from Neon"""
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                connect_args={"sslmode": "require", "connect_timeout": 30}
            )
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        cs.session_id,
                        cs.title,
                        cs.created_at,
                        cs.updated_at,
                        COUNT(ms.id) as message_count
                    FROM chat_sessions cs
                    LEFT JOIN message_store ms ON cs.session_id = ms.session_id
                    WHERE cs.user_id = :user_id
                    GROUP BY cs.session_id, cs.title, cs.created_at, cs.updated_at
                    ORDER BY cs.updated_at DESC
                    LIMIT 100
                """), {"user_id": user_id})
                
                sessions = []
                for row in result:
                    try:
                        session = {
                            "session_id": row[0],
                            "title": row[1] or "Untitled Chat",
                            "created_at": row[2].isoformat() if row[2] else None,
                            "updated_at": row[3].isoformat() if row[3] else None,
                            "message_count": row[4] or 0
                        }
                        sessions.append(session)
                    except Exception:
                        continue
                
                return sessions
                
        except Exception as e:
            logger.error(f"Failed to get user sessions from Neon: {e}")
            return []
    
    def update_session_title(self, session_id: str, title: str):
        """Update session title in Neon"""
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                connect_args={"sslmode": "require", "connect_timeout": 30}
            )
            
            with engine.connect() as conn:
                conn.execute(text("""
                    UPDATE chat_sessions 
                    SET title = :title, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = :session_id
                """), {
                    "session_id": session_id,
                    "title": title
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update session title in Neon: {e}")
    
    def delete_session(self, session_id: str):
        """Delete session from Neon"""
        try:
            engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                connect_args={"sslmode": "require", "connect_timeout": 30}
            )
            
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM message_store WHERE session_id = :session_id"), 
                           {"session_id": session_id})
                conn.execute(text("DELETE FROM chat_sessions WHERE session_id = :session_id"), 
                           {"session_id": session_id})
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to delete session from Neon: {e}")