# 🤖 FastAPI AI Worker - Advanced OSINT Agent

> **The core intelligence engine of the OSINT platform - a sophisticated AI agent system built with LangGraph, featuring vector memory, MCP tools, and comprehensive tracking.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange.svg)](https://www.trychroma.com/)

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [🏗️ Architecture](#️-architecture)
- [🔧 Usage Examples](#-usage-examples)
- [🧪 Testing](#-testing)
- [📊 Monitoring](#-monitoring)
- [🔧 Development](#-development)

## 🎯 Overview

The FastAPI AI Worker is the intelligent core of the OSINT platform, providing:

- **Multi-Agent Intelligence**: LangGraph-powered agent workflows with specialized nodes
- **Vector Memory System**: Semantic memory with ChromaDB and Sentence Transformers
- **MCP Tool Integration**: 13+ specialized OSINT tools for comprehensive investigations
- **Professional Reporting**: Executive-level intelligence assessments with confidence scoring
- **Real-time Tracking**: Langfuse integration for observability and debugging

## ✨ Key Features

### 🧠 **Intelligent Agent System**
- **LangGraph Workflow**: 6-node pipeline for systematic investigations
- **Adaptive Planning**: Dynamic investigation strategies based on findings
- **Multi-Model Support**: OpenAI GPT-4, Anthropic Claude, and custom model selection
- **Confidence Assessment**: Multi-factor scoring with uncertainty quantification

### 🔍 **Comprehensive OSINT Tools**
- **Domain Intelligence**: WHOIS, DNS analysis, SSL certificate inspection
- **Network Analysis**: IP reputation, port scanning, subdomain enumeration
- **Social Media**: Profile analysis and content verification
- **Security Tools**: File hash analysis, breach detection, paste site monitoring
- **Image Intelligence**: Metadata extraction and reverse image search

### 🧠 **Advanced Memory System**
- **Vector Memory**: Semantic similarity search with ChromaDB
- **Conversation Memory**: Context-aware dialogue management
- **Investigation History**: Complete audit trail and case management
- **Memory Statistics**: Usage analytics and performance metrics

### 📊 **Professional Reporting**
- **Executive Summaries**: High-level intelligence assessments
- **Risk Indicators**: Threat level classification and impact analysis
- **Confidence Scoring**: Statistical confidence with uncertainty bounds
- **Actionable Recommendations**: Specific next steps and mitigation strategies

## 🚀 Quick Start

### Prerequisites
- **Python 3.12** (exact version required)
- **uv** package manager (recommended) or pip
- **API Keys**: OpenAI, Anthropic, Tavily (see configuration)

### Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/your-org/osint-app.git
cd osint-app/fastapi-ai-worker

# Install with uv (recommended)
uv sync

# OR install with pip
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your API keys (see Configuration section)
```

### Quick Launch

```bash
# Start the AI Worker
uv run uvicorn app.main:app --reload --port 8001

# OR with Python directly
python -m uvicorn app.main:app --reload --port 8001
```

### Access Points

- **API Documentation**: http://localhost:8001/docs
- **ReDoc Documentation**: http://localhost:8001/redoc
- **Health Check**: http://localhost:8001/health
- **Vector Memory Stats**: http://localhost:8001/memory/stats

## Environment Variables

```bash
# Copy the example environment file
cp .env.example .env
```

Create a `.env` file with the following configuration:

```bash
# Core Configuration
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Keys (Required)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
TAVILY_API_KEY=tvly-your-tavily-key-here

# Model Configuration
DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.1
MAX_TOKENS=4000

# Vector Memory Configuration
VECTOR_DB_PATH=./data/vector_db
ENABLE_MEMORY_PERSISTENCE=true
MEMORY_SIMILARITY_THRESHOLD=0.7
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search Configuration
MAX_SEARCH_RESULTS=20
SEARCH_TIMEOUT=30
ENABLED_SEARCH_SOURCES=tavily,whois,dns

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=900

# Langfuse Tracking (Optional)
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
ENABLE_LANGFUSE_TRACKING=true

# Security Settings
ENABLE_PII_DETECTION=true
MAX_REQUEST_SIZE=1048576
```

### API Key Setup Guide

#### 1. OpenAI API Key
```bash
# Visit https://platform.openai.com/api-keys
# Create a new API key
export OPENAI_API_KEY=sk-your-key-here
```

#### 2. Anthropic API Key
```bash
# Visit https://console.anthropic.com/
# Generate an API key
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

#### 3. Tavily Search API
```bash
# Visit https://tavily.com/
# Sign up and get your API key
export TAVILY_API_KEY=tvly-your-key-here
```

## 🏗️ Architecture

### Agent Workflow Pipeline

```mermaid
graph TD
    A[Start Investigation] --> B[Planner Node]
    B --> C[Search Intel Node]
    C --> D[Synthesis Node]
    D --> E[MCP Identifier Node]
    E --> F[MCP Executor Node]
    F --> G[Final Updater Node]
    G --> H[Complete]
    
    subgraph "Memory System"
        I[Vector Memory]
        J[Conversation Memory]
        K[ChromaDB Storage]
    end
    
    D -.-> I
    F -.-> I
    G -.-> I
```

### Workflow Stages

| Stage | Purpose | Output |
|-------|---------|--------|
| **Planner** | Analyze topic and create search strategy | Search queries and investigation plan |
| **Search Intel** | Execute multi-source searches | Raw search results and web intelligence |
| **Synthesis** | Analyze and synthesize search data | Structured findings and identified gaps |
| **MCP Identifier** | Determine verification tools needed | List of relevant MCP tools |
| **MCP Executor** | Execute specialized OSINT tools | Verification results and additional data |
| **Final Updater** | Generate professional intelligence report | Complete assessment with confidence scores |

### Available MCP Tools (13 total)

1. **domain_whois** - WHOIS domain information lookup
2. **ip_reputation** - IP address reputation and geolocation analysis
3. **email_breach_check** - Email address breach detection
4. **url_safety_check** - URL safety and reputation verification
5. **ssl_certificate_check** - SSL certificate analysis and validation
6. **social_media_lookup** - Social media profile investigation
7. **file_hash_analysis** - File hash reputation and malware detection
8. **phone_number_validation** - Phone number validation and carrier lookup
9. **cryptocurrency_investigation** - Cryptocurrency address analysis
10. **dns_analysis** - Comprehensive DNS record analysis
11. **image_metadata_extraction** - Image EXIF and metadata analysis
12. **network_port_scan** - Network port scanning and service detection
13. **paste_site_exposure** - Paste site and data leak detection

## 🔧 Usage Examples

### 1. Basic Investigation via API

```python
import asyncio
import httpx

async def run_investigation():
    async with httpx.AsyncClient() as client:
        # Start investigation
        response = await client.post(
            "http://localhost:8001/agent/start",
            json={
                "topic": "Cybersecurity assessment of example.com",
                "model_id": "gpt-4o",
                "temperature": 0.3
            }
        )
        
        case_data = response.json()
        case_id = case_data["case_id"]
        
        # Monitor progress
        status_response = await client.get(f"/agent/status/{case_id}")
        print(status_response.json())

asyncio.run(run_investigation())
```

### 2. Direct Agent Usage

```python
from app.agent import agent_executor
from app.enhanced_agent_state import AgentState

async def direct_investigation():
    state = AgentState(
        topic="Domain security analysis for suspicious-domain.com",
        case_id="direct-001",
        model_id="gpt-4o",
        temperature=0.1,
        long_term_memory=[],
        search_results=[],
        mcps_executed=[],
        agent_iterations=[]
    )
    
    result = await agent_executor.ainvoke(state)
    
    print("Investigation Results:")
    print(f"Assessment: {result.get('final_assessment', 'No assessment')}")
    print(f"MCPs Executed: {len(result.get('mcps_executed', []))}")

asyncio.run(direct_investigation())
```

### 3. Memory System Usage

```python
from app.vector_memory import VectorMemoryManager

async def memory_operations():
    memory = VectorMemoryManager()
    
    # Store investigation results
    memory_data = {
        "investigation": "Domain security assessment",
        "target": "example.com",
        "findings": {"ssl_grade": "A+", "security_score": 95},
        "confidence": 0.92
    }
    
    memory_id = await memory.store_investigation_memory(memory_data)
    print(f"Stored memory: {memory_id}")
    
    # Search similar investigations
    similar_cases = await memory.find_similar_memories(
        "SSL certificate analysis", k=3, min_similarity=0.7
    )
    
    print(f"Found {len(similar_cases)} similar cases")

asyncio.run(memory_operations())
```

## 🧪 Testing

### Running Tests

```bash
# Install test dependencies
uv sync --group test

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/test_vector_memory.py -v
pytest tests/integration/ -v
```

### Key Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Memory and speed benchmarks
- **Edge Case Tests**: Error handling and recovery

## 📊 Monitoring with Langfuse

### Quick Langfuse Setup

```bash
# 1. Get API keys from https://cloud.langfuse.com
# 2. Add to .env file:
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=https://cloud.langfuse.com
ENABLE_LANGFUSE_TRACKING=true

# 3. Verify setup
python -c "from app.langfuse_tracker import is_langfuse_enabled; print(is_langfuse_enabled())"
```

### What Gets Tracked

- **Agent Sessions**: Complete investigation workflows
- **LLM Calls**: All model interactions with costs
- **Tool Executions**: MCP tool performance
- **Memory Operations**: Vector store activities
- **Error Tracking**: Failures and exceptions

## 🔧 Development

### Development Setup

```bash
# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black app/
isort app/
flake8 app/
mypy app/
```

### Adding New Features

#### New MCP Tools

```python
# app/mcps.py
async def new_osint_tool(data: dict) -> dict:
    """New OSINT analysis tool"""
    target = data.get("target")
    
    # Your analysis logic here
    results = perform_analysis(target)
    
    return {
        "tool_name": "new_osint_tool",
        "findings": results,
        "confidence": calculate_confidence(results),
        "execution_time": time.time()
    }

# Register the tool
AVAILABLE_MCPS["new_osint_tool"] = new_osint_tool
```

#### New Agent Nodes

```python
# app/agent.py
def new_analysis_node(state: AgentState) -> AgentState:
    """Custom analysis node"""
    # Analysis logic
    analysis_results = perform_custom_analysis(state)
    
    state["custom_analysis"] = analysis_results
    return state

# Add to workflow
workflow.add_node("new_analysis", new_analysis_node)
workflow.add_edge("existing_node", "new_analysis")
```

## 📚 Related Documentation

- **[Main Project README](../README.md)**: Complete OSINT platform overview
- **[Express Backend](../express-backend/README.md)**: Backend API documentation
- **[Vector Memory Bug Fixes](VECTOR_MEMORY_BUG_FIXES.md)**: Memory system improvements
- **[Testing Summary](TESTING_SUMMARY.md)**: Comprehensive testing information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
