# test_neon_detailed.py
import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_neon_detailed():
    load_dotenv()
    
    connection_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not connection_string:
        print("❌ POSTGRES_CONNECTION_STRING not found in .env")
        return
    
    # Ensure SSL mode for Neon
    if "sslmode=" not in connection_string.lower():
        separator = "&" if "?" in connection_string else "?"
        connection_string += f"{separator}sslmode=require"
    
    print(f"🔗 Testing Neon PostgreSQL connection...")
    print(f"📍 Endpoint: {connection_string.split('@')[1].split('/')[0] if '@' in connection_string else 'Unknown'}")
    
    try:
        # Create engine with Neon-optimized settings
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=5,
            max_overflow=10,
            connect_args={
                "sslmode": "require",
                "connect_timeout": 30,
                "application_name": "mcp_client_test"
            }
        )
        
        print("🔧 Engine created, testing connection...")
        
        with engine.connect() as conn:
            # Test 1: Basic connection
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connected to Neon PostgreSQL")
            print(f"📊 Database version: {version[:100]}...")
            
            # Test 2: Check current database
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.fetchone()[0]
            print(f"🗄️ Current database: {db_name}")
            
            # Test 3: Test JSONB support (required for LangChain)
            print("🧪 Testing JSONB support...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_jsonb (
                    id SERIAL PRIMARY KEY,
                    data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Test 4: Insert JSONB data
            conn.execute(text("""
                INSERT INTO test_jsonb (data) 
                VALUES (:json_data)
            """), {"json_data": '{"test": "message", "type": "human"}'})
            
            # Test 5: Query JSONB data
            result = conn.execute(text("""
                SELECT data FROM test_jsonb WHERE data->>'test' = 'message'
            """))
            json_result = result.fetchone()
            if json_result:
                print(f"✅ JSONB test successful: {json_result[0]}")
            
            # Test 6: Create LangChain message_store table
            print("🧪 Testing LangChain table structure...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS test_message_store (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    message JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Test 7: Insert LangChain-style message
            conn.execute(text("""
                INSERT INTO test_message_store (session_id, message)
                VALUES (:session_id, :message)
            """), {
                "session_id": "test_session",
                "message": '{"type": "human", "data": {"content": "Hello Neon!"}}'
            })
            
            # Test 8: Query LangChain-style message
            result = conn.execute(text("""
                SELECT session_id, message FROM test_message_store 
                WHERE session_id = 'test_session'
            """))
            message_result = result.fetchone()
            if message_result:
                print(f"✅ LangChain table test successful")
                print(f"   Session: {message_result[0]}")
                print(f"   Message: {message_result[1]}")
            
            # Cleanup test tables
            conn.execute(text("DROP TABLE IF EXISTS test_jsonb"))
            conn.execute(text("DROP TABLE IF EXISTS test_message_store"))
            conn.commit()
            
            print("🎉 All Neon PostgreSQL tests passed!")
            print("🔗 Your Neon database is ready for MCP client!")
        
    except Exception as e:
        print(f"❌ Neon connection error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Verify your Neon connection string in .env")
        print("2. Check that your Neon database is not in sleep mode")
        print("3. Ensure your IP is allowed (Neon allows all by default)")
        print("4. Try connecting via Neon's web console first")
        
        # Try to provide more specific error info
        if "SSL" in str(e):
            print("5. SSL issue - ensure sslmode=require in connection string")
        elif "timeout" in str(e):
            print("5. Timeout issue - Neon database might be sleeping")
        elif "authentication" in str(e):
            print("5. Auth issue - check username/password in connection string")

if __name__ == "__main__":
    asyncio.run(test_neon_detailed())