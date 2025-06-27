# FastAPI AI Worker - Advanced OSINT Agent

A sophisticated Open Source Intelligence (OSINT) agent built with LangGraph, featuring comprehensive MCP (Mission Critical Protocol) tools, Langfuse tracking, and professional intelligence reporting capabilities.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Langfuse Setup Instructions](#langfuse-setup-instructions)
- [Development](#development)
- [Architecture](#architecture)
- [License](#license)

## Features

- **Advanced LangGraph Workflow**: 6-node intelligent agent pipeline
- **13 MCP Tools**: Domain WHOIS, IP reputation, email breach detection, URL safety, SSL certificates, social media analysis, file hash analysis, phone number validation, cryptocurrency investigation, DNS analysis, image metadata extraction, network port scanning, and paste site exposure checking
- **Professional Intelligence Reporting**: Executive-level reports with confidence assessments, risk indicators, and actionable recommendations
- **Enhanced Langfuse Integration**: Comprehensive tracking with error handling, debug mode, and usage monitoring
- **FastAPI Backend**: High-performance async web service with streaming responses
- **Modern Frontend**: Clean, responsive interface for OSINT investigations

## Workflow

```
planner → search → synthesis → mcp_identifier → mcp_executor → final_updater → END
```

## Quick Start

1. **Install dependencies with uv:**
   ```bash
   uv sync
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the application:**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

4. **Access the interface:**
   - Web UI: http://localhost:8000
   - API docs: http://localhost:8000/docs

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models
- `TAVILY_API_KEY`: Tavily API key for web search
- `LANGFUSE_SECRET_KEY`: Langfuse secret key for tracking
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key
- `LANGFUSE_HOST`: Langfuse host URL
- `LANGFUSE_ENABLED`: Enable/disable Langfuse tracking (default: true)
- `LANGFUSE_DEBUG`: Enable debug mode for Langfuse (default: false)

## Langfuse Setup Instructions

Langfuse provides comprehensive tracking and observability for your OSINT agent. Follow these steps to set it up:

### 1. Create a Langfuse Account

**Option A: Cloud (Recommended for most users)**
1. Go to [https://cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a free account (50,000 traces/month)
3. Create a new project for your OSINT agent

**Option B: Self-hosted (For unlimited usage or privacy)**
```bash
# Using Docker Compose
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up -d
```

### 2. Get Your API Keys

1. In your Langfuse dashboard, go to **Settings** → **API Keys**
2. Copy your:
   - **Public Key** (starts with `pk-lf-...`)
   - **Secret Key** (starts with `sk-lf-...`)

### 3. Configure Environment Variables

Add the following to your `.env` file:

```bash
# Langfuse Configuration
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Control Options
LANGFUSE_ENABLED=true     # Enable/disable tracking
LANGFUSE_DEBUG=false      # Debug mode for development
```

### 4. Install Langfuse Package

The Langfuse package is already included in the project dependencies. If installing manually:

```bash
uv add langfuse
# or
pip install langfuse
```

### 5. Verify Setup

Test your Langfuse configuration:

```bash
uv run python -c "
from app.langfuse_tracker import agent_tracker, is_langfuse_enabled
print('Langfuse enabled:', is_langfuse_enabled())
print('Usage stats:', agent_tracker.get_usage_stats())
"
```

### 6. What Gets Tracked

Your OSINT agent automatically tracks:

**Agent Execution**
- Complete investigation workflows
- Input parameters (topic, model, temperature)
- Execution times and success rates
- Long-term memory usage

**Individual Components**
- **Planner**: Search query generation and strategy
- **Search**: Web search performance and results quality
- **Synthesis**: Report generation and analysis
- **MCP Tools**: All 13 verification tools with metrics
- **Final Reports**: Intelligence report quality and completeness

**LLM Operations**
- Model calls with parameters and costs
- Prompt engineering effectiveness
- Token usage and optimization opportunities

**User Interactions**
- User feedback and ratings
- Investigation request patterns
- System performance metrics

### 7. Monitoring Your Usage

**Free Tier Limits:**
- 50,000 traces/month (generous for most OSINT work)
- Unlimited users and projects
- 30-day data retention

**Usage Estimation:**
- Typical OSINT investigation: 1 main trace + 6-10 spans
- Monthly capacity: ~5,000-8,000 investigations
- Daily capacity: ~160-260 cases

**Monitor Usage:**
1. Check your [Langfuse Dashboard](https://cloud.langfuse.com)
2. View analytics and performance metrics
3. Set up alerts for usage thresholds

### 8. Advanced Configuration

**Disable Tracking (for testing):**
```bash
LANGFUSE_ENABLED=false
```

**Enable Debug Mode:**
```bash
LANGFUSE_DEBUG=true
```

**Self-hosted Instance:**
```bash
LANGFUSE_HOST=https://your-langfuse-instance.com
```

### 9. Dashboard Features

Once configured, your Langfuse dashboard provides:

- **Traces View**: Complete investigation workflows
- **Analytics**: Performance trends and success rates
- **Costs**: Token usage and LLM costs tracking
- **Feedback**: User satisfaction and quality metrics
- **Debugging**: Detailed error tracking and resolution

### 10. Troubleshooting

**Common Issues:**

1. **Authentication Error**: Check your API keys in `.env`
2. **Network Issues**: Verify `LANGFUSE_HOST` is correct
3. **Missing Traces**: Ensure environment variables are loaded
4. **Performance Impact**: Use `LANGFUSE_ENABLED=false` to test

**Debug Steps:**
```bash
# Test connection
uv run python -c "from app.langfuse_tracker import langfuse; print('Client:', langfuse)"

# Check environment
uv run python -c "import os; print('Keys loaded:', bool(os.getenv('LANGFUSE_SECRET_KEY')))"

# Enable debug logging
export LANGFUSE_DEBUG=true
```

### 11. Free Tier Tips

**Optimize Usage:**
- Each investigation creates 1 main trace
- Most users stay well within the 50,000/month limit
- Monitor usage monthly in the dashboard

**If You Hit Limits:**
1. **Self-host** for unlimited usage
2. **Optimize tracking** by reducing trace frequency
3. **Temporarily disable** with `LANGFUSE_ENABLED=false`
4. **Upgrade** to paid plan for higher limits

**Privacy for Sensitive Investigations:**
- Use self-hosted instance for classified work
- Disable tracking for highly sensitive cases
- Consider on-premises deployment for maximum security

---

## Quick Reference

### Essential Commands
```bash
# Setup
uv sync
cp .env.example .env
# Edit .env with your API keys

# Run
uv run uvicorn app.main:app --reload

# Test Langfuse
uv run python -c "from app.langfuse_tracker import is_langfuse_enabled; print('Langfuse:', is_langfuse_enabled())"

# Disable tracking (for testing)
export LANGFUSE_ENABLED=false
```

### Key URLs
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Langfuse Cloud**: https://cloud.langfuse.com
- **Self-hosted Setup**: https://langfuse.com/docs/deployment/self-host

### Support & Resources
- **Documentation**: Complete setup guides in this README
- **Langfuse Docs**: https://langfuse.com/docs
- **Community**: https://discord.gg/7NXusRtqYU
- **Issues**: GitHub Issues for this repository

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .
```

## Architecture

- **Agent Pipeline**: Strategic planner → Web search → Synthesis → MCP identification → MCP execution → Final reporting
- **MCP Tools**: 13 specialized OSINT verification tools
- **Tracking**: Comprehensive Langfuse integration for all operations
- **Quality Assurance**: Automated validation, confidence scoring, and report quality metrics

## License

MIT License - see LICENSE file for details.
