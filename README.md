  The Big Picture: How MCP + LLM Works

  Here's the complete flow you need to implement:

  User: "What files are in my home directory?"
       ↓
  1. Send to LLM with available tools
       ↓
  2. LLM responds: "I need to use the 'list_files' tool"
       ↓
  3. Your client calls MCP server: call_tool("list_files", {"path": "/home/user"})
       ↓
  4. MCP server executes tool and returns: ["file1.txt", "file2.py", "folder1/"]
       ↓
  5. Send results back to LLM
       ↓
  6. LLM responds: "Here are the files in your home directory: file1.txt, file2.py, folder1/"
       ↓
  7. Show final answer to user