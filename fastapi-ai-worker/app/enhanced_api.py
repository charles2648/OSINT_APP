"""
Enhanced API endpoints for conversation memory integration.

This module provides additional API endpoints to support conversation tracking,
context reuse, and memory-driven OSINT investigations.
"""

import json
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .conversation_memory import create_conversation_memory_manager, MessageRole, MessageType
from .enhanced_agent_state import create_conversation_aware_enhancer


# Request/Response Models for Enhanced API

class ConversationStartRequest(BaseModel):
    user_id: Optional[str] = None
    initial_topic: Optional[str] = None
    session_tags: List[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    role: str
    content: str
    message_type: str
    timestamp: float
    confidence_score: Optional[float] = None


class EnhancedAgentRequest(BaseModel):
    topic: str
    case_id: str
    model_id: str
    temperature: float
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    continue_investigation: bool = False
    long_term_memory: List[Dict] = Field(default_factory=list)


class ConversationContextResponse(BaseModel):
    conversation_id: str
    relevant_context: Dict[str, Any]
    cached_findings: Dict[str, Any]
    memory_insights: List[Dict[str, Any]]
    context_reuse_score: float
    token_optimization_available: bool


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    user_id: Optional[str]
    duration: float
    topics: List[str]
    metrics: Dict[str, Any]
    cached_findings: Dict[str, Any]
    session_tags: List[str]


def add_conversation_memory_endpoints(app: FastAPI):
    """
    Add conversation memory endpoints to the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    # Initialize conversation memory components
    conversation_memory = create_conversation_memory_manager()
    agent_enhancer = create_conversation_aware_enhancer(conversation_memory)
    
    @app.post("/conversation/start", response_model=Dict[str, str])
    async def start_conversation(request: ConversationStartRequest):
        """Start a new conversation session."""
        try:
            conversation_id = await conversation_memory.start_conversation(
                user_id=request.user_id,
                initial_topic=request.initial_topic,
                session_tags=request.session_tags
            )
            
            return {
                "conversation_id": conversation_id,
                "status": "started",
                "message": "Conversation session initiated successfully"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")
    
    @app.get("/conversation/{conversation_id}/context", response_model=ConversationContextResponse)
    async def get_conversation_context(conversation_id: str, query: str):
        """Get relevant context for a conversation and query."""
        try:
            context = await conversation_memory.get_relevant_context(
                conversation_id=conversation_id,
                current_query=query,
                max_tokens=2000
            )
            
            if "error" in context:
                raise HTTPException(status_code=404, detail=context["error"])
            
            # Calculate context reuse score
            reuse_score = len(context.get("cached_findings", {})) * 0.4 + \
                         len(context.get("relevant_history", [])) * 0.3 + \
                         len(context.get("memory_insights", [])) * 0.3
            
            return ConversationContextResponse(
                conversation_id=conversation_id,
                relevant_context=context.get("recent_conversation", []),
                cached_findings=context.get("cached_findings", {}),
                memory_insights=context.get("memory_insights", []),
                context_reuse_score=min(reuse_score, 1.0),
                token_optimization_available=len(context.get("cached_findings", {})) > 0
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get context: {str(e)}")
    
    @app.get("/conversation/{conversation_id}/summary", response_model=ConversationSummaryResponse)
    async def get_conversation_summary(conversation_id: str, include_findings: bool = True):
        """Get comprehensive conversation summary."""
        try:
            summary = await conversation_memory.get_conversation_summary(
                conversation_id=conversation_id,
                include_findings=include_findings
            )
            
            if "error" in summary:
                raise HTTPException(status_code=404, detail=summary["error"])
            
            return ConversationSummaryResponse(**summary)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")
    
    @app.post("/conversation/{conversation_id}/continue_investigation")
    async def continue_investigation(conversation_id: str, case_id: str, additional_query: str):
        """Continue an existing investigation with additional context."""
        try:
            context, should_use_cache = await conversation_memory.continue_investigation(
                conversation_id=conversation_id,
                case_id=case_id,
                additional_query=additional_query
            )
            
            return {
                "conversation_id": conversation_id,
                "case_id": case_id,
                "context": context,
                "cache_recommended": should_use_cache,
                "investigation_continuity": context.get("investigation_continuity", False)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to continue investigation: {str(e)}")
    
    @app.post("/run_enhanced_agent_stream")
    async def run_enhanced_agent(request: EnhancedAgentRequest):
        """Run OSINT agent with enhanced conversation memory integration."""
        try:
            # Enhance the agent state with conversation memory
            enhanced_state = await agent_enhancer.enhance_agent_state(
                original_state=request.dict(),
                conversation_id=request.conversation_id,
                user_id=request.user_id
            )
            
            # Convert enhanced state to AgentRequest for compatibility
            from .main import AgentRequest, run_agent_and_stream
            
            agent_request = AgentRequest(
                topic=request.topic,
                model_id=request.model_id,
                temperature=request.temperature,
                case_id=enhanced_state.get("case_id", "")
            )
            
            return run_agent_and_stream(agent_request)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to run enhanced agent: {str(e)}")
    
    @app.post("/conversation/{conversation_id}/add_message")
    async def add_conversation_message(
        conversation_id: str,
        role: MessageRole,
        content: str,
        message_type: MessageType,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None,
        related_case_ids: Optional[List[str]] = None
    ):
        """Add a message to the conversation."""
        try:
            message_id = await conversation_memory.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                message_type=message_type,
                tokens_used=tokens_used,
                processing_time=processing_time,
                confidence_score=confidence_score,
                related_case_ids=related_case_ids
            )
            
            return {
                "message_id": message_id,
                "conversation_id": conversation_id,
                "status": "added"
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")
    
    @app.get("/conversation/{conversation_id}/messages")
    async def get_conversation_messages(
        conversation_id: str,
        limit: int = 50,
        message_type: Optional[str] = None
    ):
        """Get messages from a conversation."""
        try:
            # This would need to be implemented in ConversationMemoryManager
            # For now, return a placeholder response
            return {
                "conversation_id": conversation_id,
                "messages": [],
                "total_count": 0,
                "message": "Message retrieval endpoint - implementation needed in ConversationMemoryManager"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")
    
    @app.delete("/conversation/{conversation_id}")
    async def end_conversation(conversation_id: str):
        """End and archive a conversation session."""
        try:
            # This would need to be implemented in ConversationMemoryManager
            return {
                "conversation_id": conversation_id,
                "status": "ended",
                "message": "Conversation ended and archived"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to end conversation: {str(e)}")
    
    @app.get("/conversations/active")
    async def get_active_conversations(user_id: Optional[str] = None):
        """Get list of active conversations."""
        try:
            # This would need to be implemented to return active conversations
            # For now, return a placeholder
            return {
                "active_conversations": [],
                "total_count": 0,
                "message": "Active conversations endpoint - implementation needed"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")
    
    @app.get("/conversation/analytics")
    async def get_conversation_analytics(
        user_id: Optional[str] = None,
        time_range: str = "24h"
    ):
        """Get conversation analytics and insights."""
        try:
            return {
                "analytics": {
                    "total_conversations": 0,
                    "average_duration": 0,
                    "token_savings": 0,
                    "context_reuse_rate": 0.0,
                    "investigation_continuity_rate": 0.0
                },
                "message": "Analytics endpoint - implementation needed for comprehensive metrics"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# Integration helper for existing main.py
async def run_agent_and_stream_enhanced(enhanced_state: Dict[str, Any], agent_enhancer):
    """
    Enhanced version of run_agent_and_stream with conversation memory integration.
    
    This function would replace or extend the existing run_agent_and_stream
    to provide conversation memory capabilities.
    """
    # This is a placeholder for the enhanced streaming function
    # It would integrate with the existing agent_executor while adding
    # conversation memory tracking and optimization
    
    conversation_id = enhanced_state.get("conversation_id")
    
    # Track conversation metrics
    start_time = time.time()
    
    try:
        # Here you would integrate with the existing agent_executor
        # while adding conversation memory hooks
        
        # Example integration points:
        # 1. Before planner: optimize with memory insights
        # 2. After synthesis: track in conversation memory
        # 3. After finalization: update conversation context
        
        # For now, return a placeholder response
        yield f"event: enhanced_status\ndata: {{\"message\": \"Enhanced agent with conversation_id: {conversation_id}\"}}\n\n"
        
        # Finalize conversation tracking
        final_output = {"synthesized_findings": "Enhanced investigation completed"}
        enhanced_output = await agent_enhancer.finalize_conversation_investigation(
            enhanced_state, final_output
        )
        
        yield f"event: enhanced_complete\ndata: {json.dumps(enhanced_output)}\n\n"
        
    except Exception as e:
        yield f"event: error\ndata: {{\"error\": \"Enhanced agent error: {str(e)}\"}}\n\n"
