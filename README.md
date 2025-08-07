# simple-agent-mcp
An LLM-based agent that uses MCP (Model Context Protocol) to connect with external tools, including SQL query execution. Implements a simple Thought → Action → Observation cycle using LangChain and Ollama.

This repository contains a working demo of an LLM-powered agent that interacts with external tools via Model Context Protocol (MCP). The system includes:

- A Python-based MCP server exposing tools
- An async Python agent that selects tools via LLM reasoning
- Tool execution + result refinement using a `refine` prompt

  ## Features

- MCP-based tool communication (JSON-RPC 2.0)
- Tool selection with LLM (qwen3:4b via Ollama)
- Thought → Action → Observation logic
- Refine step to turn raw tool outputs into natural language

  ### 📊 Veri Seti

Bu projede kullanılan örnek çalışan veritabanı, Bytebase ekibi tarafından sağlanan açık kaynaklı bir dataset'ten alınmıştır:  
🔗 https://github.com/bytebase/employee-sample-database  
Kullanılan dosya: [`employee.db`](https://github.com/bytebase/employee-sample-database/blob/main/sqlite/dataset_small/employee.db)

Telif hakkı ve lisans detayları için ilgili repoya göz atabilirsiniz.


## Setup

```bash
git clone https://github.com/zeynepsudenaz/simple-agent-mcp.git
cd simple-agent-mcp
pip install -r requirements.txt
python mcp_server.py

### 📂 Dosya Yapısı

```markdown
## Project Structure

- `mcp_server.py`: MCP-compatible tool server
- `agent.py`: Agent script using LLM + tool call logic

