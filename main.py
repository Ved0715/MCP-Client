"""Main entry point for MCP client application"""

import asyncio
import sys
import logging
from src.config.settings import Settings
from src.cli.interactive import interactive_chat, run_example_queries
from src.core.orchestrator import MCPLLMOrchestrator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    """Main application entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "chat":
            await interactive_chat()
        elif command == "test":
            await run_example_queries()
        elif command == "api":
            await run_api_server()
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  chat  - Start interactive chat")
            print("  test  - Run example queries")
            print("  api   - Start FastAPI server")
    else:
        await interactive_chat()

async def run_api_server():
    """Run FastAPI server"""
    import uvicorn
    from fastapi import FastAPI
    from src.api.routes import router
    
    app = FastAPI(
        title="MCP Client API",
        description="REST API for MCP client with LangChain integration",
        version="1.0.0"
    )
    
    app.include_router(router)
    
    # Add CORS middleware
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    print("🚀 Starting MCP Client API server...")
    print("📖 API documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    asyncio.run(main())