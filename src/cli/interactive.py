"""
Interactive CLI for MCP client
"""

import asyncio
import logging
import sys
from typing import Optional
from datetime import datetime

from ..config.settings import Settings
from ..core.orchestrator import MCPLLMOrchestrator
from ..core.memory import PostgreSQLMemoryManager

logger = logging.getLogger(__name__)

class InteractiveCLI:
    """Interactive command-line interface for MCP client"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.orchestrator: Optional[MCPLLMOrchestrator] = None
        self.memory_manager = PostgreSQLMemoryManager(settings)
        self.current_user_id = "cli_user"
        
    async def start(self):
        """Start the interactive CLI"""
        
        print("🌟 Welcome to MCP + LLM Interactive Chat!")
        print("💾 All conversations are automatically saved to PostgreSQL")
        print(f"🔗 MCP Server: {self.settings.mcp_server_url}")
        print("=" * 70)
        
        try:
            # Get user ID
            self.current_user_id = self._get_user_id()
            
            # Handle session selection
            session_id = await self._handle_session_selection()
            
            # Initialize orchestrator
            self.orchestrator = MCPLLMOrchestrator(
                settings=self.settings,
                session_id=session_id,
                user_id=self.current_user_id
            )
            
            await self.orchestrator.initialize()
            
            # Show available tools
            self._show_available_tools()
            
            # Start chat loop
            await self._chat_loop()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            await self._cleanup()
    
    def _get_user_id(self) -> str:
        """Get user ID from input"""
        user_id = input("👤 Enter your user ID (or press Enter for 'cli_user'): ").strip()
        return user_id if user_id else "cli_user"
    
    async def _handle_session_selection(self) -> Optional[str]:
        """Handle session selection or creation"""
        
        print(f"\n📋 Session Management for user: {self.current_user_id}")
        print("1. Start new conversation")
        print("2. Continue existing conversation")
        print("3. List all sessions")
        
        while True:
            choice = input("Choose option (1, 2, or 3): ").strip()
            
            if choice == "1":
                return None  # New session
            elif choice == "2":
                return await self._select_existing_session()
            elif choice == "3":
                await self._list_all_sessions()
                continue
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    async def _select_existing_session(self) -> Optional[str]:
        """Select from existing sessions"""
        
        sessions = self.memory_manager.get_user_sessions(self.current_user_id)
        
        if not sessions:
            print("📭 No previous conversations found. Starting new session.")
            return None
        
        print(f"\n📜 Your previous conversations ({len(sessions)} total):")
        for i, session in enumerate(sessions[:20]):  # Show last 20
            updated = session['updated_at'][:19] if session['updated_at'] else "Unknown"
            print(f"{i+1:2d}. {session['title'][:50]:<50} "
                  f"({session['message_count']:2d} msgs) - {updated}")
        
        if len(sessions) > 20:
            print(f"    ... and {len(sessions) - 20} more sessions")
        
        try:
            choice = input("\nEnter session number (or 0 for new session): ").strip()
            
            if choice == "0":
                return None
            
            session_index = int(choice) - 1
            if 0 <= session_index < min(len(sessions), 20):
                selected_session = sessions[session_index]
                print(f"📂 Continuing session: {selected_session['title']}")
                return selected_session["session_id"]
            else:
                print("❌ Invalid session number. Starting new session.")
                return None
                
        except ValueError:
            print("❌ Invalid input. Starting new session.")
            return None
    
    async def _list_all_sessions(self):
        """List all user sessions with details"""
        
        sessions = self.memory_manager.get_user_sessions(self.current_user_id)
        
        if not sessions:
            print("📭 No sessions found.")
            return
        
        print(f"\n📊 All Sessions for {self.current_user_id} ({len(sessions)} total):")
        print("-" * 90)
        print(f"{'#':<3} {'Title':<40} {'Messages':<8} {'Created':<20} {'Updated':<20}")
        print("-" * 90)
        
        for i, session in enumerate(sessions, 1):
            created = session['created_at'][:19] if session['created_at'] else "Unknown"
            updated = session['updated_at'][:19] if session['updated_at'] else "Unknown"
            title = session['title'][:37] + "..." if len(session['title']) > 40 else session['title']
            
            print(f"{i:<3} {title:<40} {session['message_count']:<8} {created:<20} {updated:<20}")
        
        print("-" * 90)
        input("Press Enter to continue...")
    
    def _show_available_tools(self):
        """Show available tools and system info"""
        
        if not self.orchestrator:
            return
            
        session_info = self.orchestrator.get_session_info()
        
        print(f"\n🆔 Session ID: {session_info['session_id']}")
        print(f"🤖 LLM Model: {session_info['llm_model']}")
        print(f"🔧 Available Tools ({session_info['tools_count']}):")
        
        tools = self.orchestrator.get_available_tools()
        for tool in tools:
            print(f"   • {tool['name']}: {tool['description']}")
        
        print("\n💬 Start chatting! Commands available:")
        print("   - 'quit' or 'exit': Exit the chat")
        print("   - 'help': Show this help message")
        print("   - 'history': Show chat history")
        print("   - 'sessions': Show all sessions")
        print("   - 'tools': Show available tools")
        print("   - 'status': Show session status")
        print("=" * 70)
    
    async def _chat_loop(self):
        """Main chat interaction loop"""
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Your conversation has been saved.")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'history':
                    await self._show_history()
                    continue
                
                elif user_input.lower() == 'sessions':
                    await self._list_all_sessions()
                    continue
                
                elif user_input.lower() == 'tools':
                    self._show_available_tools()
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                elif user_input.lower() == 'clear':
                    print("\n" * 50)  # Clear screen
                    continue
                
                # Process user query
                print("🤔 Processing...")
                start_time = datetime.now()
                
                response = await self.orchestrator.process_user_query(user_input)
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                print(f"\n🤖 Assistant: {response}")
                print(f"⏱️ Response time: {processing_time:.2f}s")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Your conversation has been saved!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat loop error: {e}")
    
    def _show_help(self):
        """Show help message"""
        print("\n🆘 Help - Available Commands:")
        print("   quit, exit, bye  - Exit the chat")
        print("   help            - Show this help message")
        print("   history         - Show chat history from database")
        print("   sessions        - Show all your chat sessions") 
        print("   tools           - Show available MCP tools")
        print("   status          - Show current session status")
        print("   clear           - Clear the screen")
        print("\nJust type your message to chat with the assistant!")
    
    async def _show_history(self):
        """Show chat history from PostgreSQL"""
        
        if not self.orchestrator:
            return
            
        try:
            history = self.orchestrator.get_chat_history()
            
            if not history:
                print("📭 No chat history found for this session.")
                return
            
            print(f"\n📜 Chat History ({len(history)} messages):")
            print("-" * 60)
            
            for msg in history:
                role_icon = "👤" if msg["role"] == "human" else "🤖"
                timestamp = msg.get("timestamp", "Unknown")[:19] if msg.get("timestamp") else "Unknown"
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                
                print(f"{role_icon} [{timestamp}] {content}")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Error retrieving history: {e}")
    
    def _show_status(self):
        """Show current session status"""
        
        if not self.orchestrator:
            return
            
        try:
            session_info = self.orchestrator.get_session_info()
            history = self.orchestrator.get_chat_history()
            
            print(f"\n📊 Session Status:")
            print(f"   Session ID: {session_info['session_id']}")
            print(f"   User ID: {session_info['user_id']}")
            print(f"   Initialized: {'✅' if session_info['initialized'] else '❌'}")
            print(f"   Available Tools: {session_info['tools_count']}")
            print(f"   LLM Model: {session_info['llm_model']}")
            print(f"   Messages in Session: {len(history)}")
            print(f"   MCP Server: {session_info['mcp_server_url']}")
            
        except Exception as e:
            print(f"❌ Error getting status: {e}")
    
    async def _cleanup(self):
        """Clean up resources"""
        
        try:
            if self.orchestrator:
                await self.orchestrator.close()
                print("💾 Session saved to PostgreSQL")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

async def interactive_chat():
    """Main entry point for interactive chat"""
    
    try:
        # Load and validate settings
        settings = Settings()
        settings.validate()
        
        # Create and start CLI
        cli = InteractiveCLI(settings)
        await cli.start()
        
    except Exception as e:
        print(f"❌ Failed to start interactive chat: {e}")
        logger.error(f"Interactive chat startup error: {e}")

async def run_example_queries():
    """Run example queries for testing"""
    
    try:
        settings = Settings()
        settings.validate()
        
        print("🧪 Running Example Queries...")
        print("=" * 50)
        
        async with MCPLLMOrchestrator(settings, user_id="test_user") as orchestrator:
            
            example_queries = [
                "What tools do you have available?",
                "List files in the current directory",
                "Calculate 25 * 4 + 10",
                "Search for documents about machine learning",
                "What's the weather like today?",
                "Help me write a Python function to sort a list"
            ]
            
            for i, query in enumerate(example_queries, 1):
                print(f"\n🔄 Example {i}: {query}")
                print("-" * 30)
                
                try:
                    response = await orchestrator.process_user_query(query)
                    print(f"✅ Response: {response[:200]}...")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                # Small delay between queries
                await asyncio.sleep(1)
        
        print("\n🎉 Example queries completed!")
        
    except Exception as e:
        print(f"❌ Failed to run example queries: {e}")
        logger.error(f"Example queries error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        asyncio.run(run_example_queries())
    else:
        asyncio.run(interactive_chat())