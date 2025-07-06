# The FastAPI application for the AI worker service.

import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from .agent import agent_executor
from .llm_selector import AVAILABLE_MODELS_CONFIG
from .langfuse_tracker import track_agent_session_start, track_agent_session_end, agent_tracker
from .agent_memory import create_agent_memory_manager
from .enhanced_agent_wrapper import (
    run_enhanced_agent_with_memory,
    get_conversation_history,
    continue_conversation,
    start_new_conversation
)
from . import mcps

app = FastAPI(
    title="OSINT FastAPI AI Worker",
    description="Advanced OSINT Agent with LangGraph, MCP tools, and conversation memory",
    version="1.0.0"
)

# Initialize global memory manager for API endpoints
memory_manager = create_agent_memory_manager()

# Simple MCP function wrapper
def list_available_mcps():
    """List available MCP functions"""
    return [
        "get_domain_whois",
        "analyze_dns_records",
        "get_ssl_certificate", 
        "analyze_phone_number",
        "extract_image_metadata",
        "check_ip_reputation",
        "verify_email_breach",
        "analyze_url_safety"
    ]

async def execute_mcp_function(mcp_name: str, query: str, **parameters):
    """Execute an MCP function"""
    try:
        if mcp_name == "get_domain_whois":
            return mcps.get_domain_whois(query)
        elif mcp_name == "analyze_dns_records":
            return mcps.analyze_dns_records(query)
        elif mcp_name == "get_ssl_certificate":
            return mcps.get_ssl_certificate(query)
        elif mcp_name == "analyze_phone_number":
            return mcps.analyze_phone_number(query)
        elif mcp_name == "extract_image_metadata":
            return mcps.extract_image_metadata(query)
        elif mcp_name == "check_ip_reputation":
            return mcps.check_ip_reputation(query)
        elif mcp_name == "verify_email_breach":
            return mcps.verify_email_breach(query)
        elif mcp_name == "analyze_url_safety":
            return mcps.analyze_url_safety(query)
        else:
            return {"error": f"Unknown MCP function: {mcp_name}"}
    except Exception as e:
        return {"error": str(e)}

class AgentRequest(BaseModel):
    topic: str
    case_id: str
    model_id: str
    temperature: float
    long_term_memory: List[Dict] = Field(default_factory=list)

async def run_agent_and_stream(request: AgentRequest):
    initial_state = request.dict()
    
    # Start tracking the agent session
    trace_id = track_agent_session_start(
        case_id=request.case_id,
        topic=request.topic,
        model_id=request.model_id,
        temperature=request.temperature,
        long_term_memory=request.long_term_memory
    )
    
    init_event = {"event": "task_created", "data": {"case_id": request.case_id, "trace_id": trace_id}}
    yield f"event: {init_event['event']}\ndata: {json.dumps(init_event['data'])}\n\n"

    success = True
    final_state: Dict[str, Any] = {}

    try:
        async for event in agent_executor.astream_events(initial_state, version="v2"):
            event_type = event["event"]
            if event_type.endswith("on_chain_end"):
                node_name = event["name"]
                if node_name == "planner":
                    message = "Planning research..."
                elif node_name == "search":
                    message = "Executing web searches... 🔎"
                elif node_name == "synthesis":
                    message = "Synthesizing findings..."
                elif node_name == "mcp_identifier":
                    message = "Identifying facts for verification..."
                elif node_name == "mcp_executor":
                    message = "Running verification tools (MCPs)... 🛡️"
                elif node_name == "final_updater":
                    message = "Compiling final verified report..."
                else:
                    continue
                stream_event = {"event": "status_update", "data": {"message": message}}
                yield f"event: {stream_event['event']}\ndata: {json.dumps(stream_event['data'])}\n\n"
        
        final_state = await agent_executor.ainvoke(initial_state)
        
    except Exception as e:
        success = False
        final_state = {"error": str(e), "synthesized_findings": f"Error occurred: {str(e)}"}
        
    # End tracking the agent session
    track_agent_session_end(final_state, success)
    
    review_event = {
        "event": "review_required",
        "data": {
            "synthesized_findings": final_state.get("synthesized_findings", "Error: No findings."),
            "trace_id": trace_id,
            "success": success
        }
    }
    yield f"event: {review_event['event']}\ndata: {json.dumps(review_event['data'])}\n\n"

@app.get("/models")
async def get_models():
    return AVAILABLE_MODELS_CONFIG

@app.post("/run_agent_stream")
async def run_agent(request: AgentRequest):
    return StreamingResponse(
        run_agent_and_stream(request),
        media_type="text/event-stream"
    )

class FeedbackRequest(BaseModel):
    case_id: str
    feedback_type: str  # "approve", "reject", "modify"
    feedback_data: Dict
    trace_id: Optional[str] = None

@app.post("/submit_feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback on agent outputs for tracking and improvement."""
    agent_tracker.track_user_feedback(
        case_id=request.case_id,
        feedback_type=request.feedback_type,
        feedback_data=request.feedback_data
    )
    return {"status": "feedback_recorded", "case_id": request.case_id}

class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 10

@app.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Search long-term memory for similar investigations."""
    results = await memory_manager.search_memories(request.query, limit=request.limit)
    return {"results": results}

@app.get("/memory/stats")
async def get_memory_stats():
    """Get statistics about the memory system."""
    stats = await memory_manager.get_memory_statistics()
    return stats

@app.get("/memory/patterns")
async def get_memory_patterns(topic: Optional[str] = None):
    """Get investigation pattern insights from memory."""
    patterns = await memory_manager.get_investigation_patterns(topic_filter=topic)
    return patterns

# Enhanced agent request models
class EnhancedAgentRequest(BaseModel):
    topic: str
    model_id: str = "gpt-4o"
    temperature: float = 0.1
    conversation_id: Optional[str] = None
    case_id: Optional[str] = None
    user_id: Optional[str] = None

class ConversationContinueRequest(BaseModel):
    conversation_id: str
    new_query: str
    model_id: str = "gpt-4o"
    temperature: float = 0.1

@app.post("/enhanced_agent/start")
async def start_enhanced_investigation(request: EnhancedAgentRequest):
    """Start a new OSINT investigation with conversation memory tracking."""
    try:
        result = await start_new_conversation(
            topic=request.topic,
            model_id=request.model_id,
            temperature=request.temperature,
            user_id=request.user_id
        )
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/enhanced_agent/continue")
async def continue_enhanced_investigation(request: ConversationContinueRequest):
    """Continue an existing conversation with a new query."""
    try:
        result = await continue_conversation(
            conversation_id=request.conversation_id,
            new_query=request.new_query,
            model_id=request.model_id,
            temperature=request.temperature
        )
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/conversation/{conversation_id}/history")
async def get_conversation_history_endpoint(conversation_id: str):
    """Get conversation history for a specific conversation."""
    try:
        history = await get_conversation_history(conversation_id)
        return {"success": True, "data": history}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/enhanced_agent/run")
async def run_enhanced_agent_endpoint(request: EnhancedAgentRequest):
    """Run enhanced agent with conversation memory (one-shot or continue existing)."""
    try:
        result = await run_enhanced_agent_with_memory(
            topic=request.topic,
            model_id=request.model_id,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            case_id=request.case_id,
            user_id=request.user_id
        )
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Core OSINT Investigation endpoints
class InvestigationRequest(BaseModel):
    topic: str
    search_queries: List[str] = Field(default_factory=list)
    priority: str = "medium"  # low, medium, high
    case_id: Optional[str] = None
    model_id: str = "gpt-4o-mini"
    temperature: float = 0.1

class EnhancedInvestigationRequest(BaseModel):
    topic: str
    search_queries: List[str] = Field(default_factory=list)
    priority: str = "medium"
    case_id: Optional[str] = None
    model_id: str = "gpt-4o-mini"
    temperature: float = 0.1
    use_memory: bool = True
    max_iterations: int = 3
    conversation_id: Optional[str] = None

class ConversationCreateRequest(BaseModel):
    topic: str
    initial_context: Optional[str] = None

class ConversationMessageRequest(BaseModel):
    message: str
    search_sources: List[str] = Field(default_factory=lambda: ["tavily"])
    use_memory: bool = True

class MCPExecuteRequest(BaseModel):
    mcp_name: str
    query: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

@app.post("/investigate")
async def investigate_topic(request: InvestigationRequest):
    """Run a basic OSINT investigation"""
    try:
        # Use the agent executor to run investigation
        initial_state = {
            "topic": request.topic,
            "case_id": request.case_id or f"case_{int(time.time())}",
            "model_id": request.model_id,
            "temperature": request.temperature,
            "long_term_memory": []
        }
        
        final_state = await agent_executor.ainvoke(initial_state)
        
        return {
            "case_id": final_state.get("case_id"),
            "findings": final_state.get("synthesized_findings", "No findings available"),
            "confidence_score": final_state.get("confidence_score", 0.5),
            "search_results": final_state.get("search_results", []),
            "mcp_results": final_state.get("mcp_results", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/investigate_enhanced")
async def investigate_enhanced(request: EnhancedInvestigationRequest):
    """Run enhanced OSINT investigation with memory and context reuse"""
    try:
        result = await run_enhanced_agent_with_memory(
            topic=request.topic,
            model_id=request.model_id,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            case_id=request.case_id
        )
        
        return {
            "case_id": result.get("case_id"),
            "findings": result.get("synthesized_findings", "No findings available"),
            "confidence_score": result.get("confidence_score", 0.5),
            "memory_summary": result.get("memory_context", []),
            "context_reused": result.get("context_reused", False),
            "tokens_saved": result.get("tokens_saved", 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Conversation management endpoints
@app.post("/conversations")
async def create_conversation(request: ConversationCreateRequest):
    """Create a new conversation for investigation tracking"""
    try:
        result = await start_new_conversation(
            topic=request.topic,
            model_id="gpt-4o-mini"
        )
        
        return {
            "conversation_id": result.get("conversation_id"),
            "status": "created",
            "topic": request.topic
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/{conversation_id}/message")
async def send_conversation_message(conversation_id: str, request: ConversationMessageRequest):
    """Send a message to an existing conversation"""
    try:
        result = await continue_conversation(
            conversation_id=conversation_id,
            new_query=request.message,
            model_id="gpt-4o-mini"
        )
        
        return {
            "response": result.get("synthesized_findings", "No response available"),
            "context_reused": result.get("context_reused", False),
            "tokens_saved": result.get("tokens_saved", 0),
            "confidence_score": result.get("confidence_score", 0.5)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# MCP tools endpoints
@app.get("/mcps")
async def list_mcps():
    """List available MCP tools"""
    try:
        mcps = list_available_mcps()
        return {"mcps": mcps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute_mcp")
async def execute_mcp(request: MCPExecuteRequest):
    """Execute a specific MCP tool"""
    try:
        result = await execute_mcp_function(
            request.mcp_name,
            request.query,
            **request.parameters
        )
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Memory endpoints
@app.get("/memory/status")
async def get_memory_status():
    """Get memory system status"""
    try:
        stats = await memory_manager.get_memory_statistics()
        return {
            "vector_store_active": stats.get("vector_store_available", False),
            "total_entries": stats.get("total_entries", 0),
            "memory_type": stats.get("memory_type", "basic")
        }
    except Exception as e:
        return {
            "vector_store_active": False,
            "total_entries": 0,
            "memory_type": "basic",
            "error": str(e)
        }

@app.get("/memory/vector/stats")
async def get_vector_memory_stats():
    """Get vector memory statistics"""
    try:
        # This endpoint might not work if vector memory is not available
        stats = await memory_manager.get_memory_statistics()
        return {
            "collections": stats.get("collections", 0),
            "total_documents": stats.get("total_documents", 0),
            "available": stats.get("vector_store_available", False)
        }
    except Exception as e:
        return {
            "collections": 0,
            "total_documents": 0,
            "available": False,
            "error": str(e)
        }

# Health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "OSINT FastAPI AI Worker"
    }

def main():
    """Main entry point for the OSINT agent application."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()