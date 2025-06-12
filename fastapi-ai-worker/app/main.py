# The FastAPI application for the AI worker service.

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict

from .agent import agent_executor
from .llm_selector import AVAILABLE_MODELS_CONFIG

app = FastAPI()

class AgentRequest(BaseModel):
    topic: str
    case_id: str
    model_id: str
    temperature: float
    long_term_memory: List[Dict] = Field(default_factory=list)

async def run_agent_and_stream(request: AgentRequest):
    initial_state = request.dict()
    
    init_event = {"event": "task_created", "data": {"case_id": request.case_id}}
    yield f"event: {init_event['event']}\ndata: {json.dumps(init_event['data'])}\n\n"

    async for event in agent_executor.astream_events(initial_state, version="v2"):
        event_type = event["event"]
        if event_type.endswith("on_chain_end"):
            node_name = event["name"]
            if node_name == "planner": message = "Planning research..."
            elif node_name == "search": message = "Executing web searches... 🔎"
            elif node_name == "synthesis": message = "Synthesizing findings..."
            elif node_name == "mcp_identifier": message = "Identifying facts for verification..."
            elif node_name == "mcp_executor": message = "Running verification tools (MCPs)... 🛡️"
            elif node_name == "final_updater": message = "Compiling final verified report..."
            else: continue
            stream_event = {"event": "status_update", "data": {"message": message}}
            yield f"event: {stream_event['event']}\ndata: {json.dumps(stream_event['data'])}\n\n"
    
    final_state = await agent_executor.ainvoke(initial_state)
    
    review_event = {
        "event": "review_required",
        "data": {"synthesized_findings": final_state.get("synthesized_findings", "Error: No findings.")}
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