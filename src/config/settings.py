import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    """Application settings from environment variables"""
    
    # MCP Server
    mcp_server_url: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
    bearer_token: Optional[str] = os.getenv("MCP_AUTH_BEARER")
    http_connect_timeout_s: float = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "10"))
    http_request_timeout_s: float = float(os.getenv("MCP_REQUEST_TIMEOUT_SECONDS", "60"))
    
    # LLM
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Database
    postgres_connection_string: str = os.getenv(
        "POSTGRES_CONNECTION_STRING"
    )
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    def validate(self) -> None:
        """Validate required settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        if not self.postgres_connection_string:
            raise ValueError("POSTGRES_CONNECTION_STRING is required")