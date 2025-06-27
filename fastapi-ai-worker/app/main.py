# The FastAPI application for the AI worker service.

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from .agent import agent_executor
from .llm_selector import AVAILABLE_MODELS_CONFIG
from .langfuse_tracker import track_agent_session_start, track_agent_session_end, agent_tracker

app = FastAPI()

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

def main():
    """Main entry point for the OSINT agent application."""
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()