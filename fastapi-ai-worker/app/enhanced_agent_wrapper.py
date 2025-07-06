# Simple enhanced agent wrapper that adds conversation memory to existing agent.

import time
import uuid
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv

from .agent import agent_executor
from .conversation_memory import ConversationMemoryManager, MessageRole, MessageType
from .agent_memory import create_agent_memory_manager

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize memory managers
memory_manager = create_agent_memory_manager()
conversation_memory = ConversationMemoryManager(memory_manager)

async def run_enhanced_agent_with_memory(
    topic: str,
    model_id: str = "gpt-4o",
    temperature: float = 0.1,
    conversation_id: Optional[str] = None,
    case_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced agent that integrates conversation memory with the existing agent.
    
    This function wraps the existing agent with conversation memory capabilities,
    enabling context reuse and chat history tracking without major refactoring.
    """
    
    # Generate IDs if not provided
    if not case_id:
        case_id = f"case_{int(time.time())}"
    
    try:
        # Check if we have existing conversation context
        cached_findings = {}
        context_summary = ""
        
        # If no conversation_id provided, start a new conversation
        if not conversation_id:
            logger.info(f"Creating new conversation for user {user_id} on topic: {topic}")
            try:
                conversation_id = await conversation_memory.start_conversation(
                    user_id=user_id or "anonymous",
                    initial_topic=topic,
                    session_tags=["osint", "investigation"]
                )
                logger.info(f"✅ Started new conversation: {conversation_id}")
                logger.info(f"Active conversations: {list(conversation_memory.active_conversations.keys())}")
            except Exception as e:
                logger.error(f"❌ Failed to start conversation: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        # Check if we have existing conversation context
        if conversation_id and conversation_id in conversation_memory.active_conversations:
            try:
                # Get relevant context from conversation
                context_data = await conversation_memory.get_relevant_context(
                    conversation_id, topic, max_tokens=1000
                )
                
                if context_data and not context_data.get("error"):
                    cached_findings = context_data.get("cached_findings", {})
                    
                    # Create context summary from conversation
                    if "recent_messages" in context_data:
                        context_summary = "Recent conversation context:\\n"
                        for msg in context_data["recent_messages"]:
                            role_label = "User" if msg.role == MessageRole.USER else "Assistant"
                            context_summary += f"{role_label}: {msg.content[:100]}...\\n"
                    
                    logger.info(f"✅ Retrieved conversation context: {len(cached_findings)} cached findings")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve conversation context: {e}")
                # Continue without context - not a critical error
        
        # Log the user's investigation request
        logger.info(f"🔍 About to add message to conversation: {conversation_id}")
        logger.info(f"🔍 Active conversations before add_message: {list(conversation_memory.active_conversations.keys())}")
        
        try:
            await conversation_memory.add_message(
                conversation_id,
                MessageRole.USER,
                f"Starting OSINT investigation: {topic}",
                MessageType.INVESTIGATION_REQUEST,
                metadata={"case_id": case_id, "model_id": model_id}
            )
            logger.info(f"✅ Successfully added message to conversation")
        except Exception as e:
            logger.error(f"❌ Failed to add message: {e}")
            raise
        
        # Check for context reuse opportunities
        context_reuse_applied = False
        if cached_findings and len(cached_findings) >= 3:
            context_reuse_applied = True
            logger.info(f"✅ Context reuse applied: {len(cached_findings)} cached findings available")
            
            # Enhanced topic with context
            enhanced_topic = f"{topic}\\n\\nContext from previous investigation:\\n"
            for key, value in list(cached_findings.items())[:3]:  # Use top 3 cached findings
                if isinstance(value, dict) and 'content' in value:
                    enhanced_topic += f"- {key}: {str(value['content'])[:100]}...\\n"
                else:
                    enhanced_topic += f"- {key}: {str(value)[:100]}...\\n"
            
            enhanced_topic += "\\nPlease build upon this existing information rather than duplicating it."
            topic = enhanced_topic
        
        # Run the existing agent with enhanced context
        initial_state = {
            "topic": topic,
            "case_id": case_id,
            "model_id": model_id,
            "temperature": temperature,
            "long_term_memory": [],
            "search_queries": [],
            "search_results": [],
            "synthesized_findings": "",
            "num_steps": 0,
            "mcp_verification_list": [],
            "verified_data": {},
            "planner_reasoning": "",
            "synthesis_confidence": "",
            "information_gaps": [],
            "search_quality_metrics": {},
            "query_performance": [],
            "mcp_execution_results": [],
            "verification_strategy": "",
            "final_confidence_assessment": "",
            "final_risk_indicators": [],
            "final_verification_summary": "",
            "final_actionable_recommendations": [],
            "final_information_reliability": "",
            "report_quality_metrics": {}
        }
        
        agent_result = await agent_executor.ainvoke(initial_state)
        
        # Extract and cache new findings for future reuse
        new_cached_findings = {}
        if agent_result.get('search_results'):
            for i, result in enumerate(agent_result['search_results'][:5]):  # Cache top 5 results
                if isinstance(result, dict) and result.get('score', 0) > 0.7:
                    cache_key = f"search_finding_{i}_{result.get('title', 'unknown')[:30]}"
                    new_cached_findings[cache_key] = {
                        'content': result.get('content', ''),
                        'source': result.get('url', ''),
                        'timestamp': time.time(),
                        'relevance_score': result.get('score', 0)
                    }
        
        # Cache significant findings from synthesis
        if agent_result.get('synthesized_findings'):
            synthesis_text = agent_result['synthesized_findings']
            if len(synthesis_text) > 200:  # Only cache substantial synthesis
                cache_key = f"synthesis_{case_id}_{int(time.time())}"
                new_cached_findings[cache_key] = {
                    'content': synthesis_text,
                    'source': 'agent_synthesis',
                    'timestamp': time.time(),
                    'type': 'synthesis'
                }
        
        # Store new findings in conversation memory
        if new_cached_findings:
            await conversation_memory.cache_investigation_findings(
                conversation_id,
                case_id,
                new_cached_findings,
                confidence_score=0.8  # Default confidence
            )
            print(f"✅ Cached {len(new_cached_findings)} new findings for future reuse")
        
        # Log the agent's response
        response_summary = f"Investigation completed with {len(agent_result.get('search_results', []))} search results"
        if new_cached_findings:
            response_summary += f" and {len(new_cached_findings)} findings cached for future reuse"
        
        await conversation_memory.add_message(
            conversation_id,
            MessageRole.ASSISTANT,
            response_summary,
            MessageType.ANALYSIS_RESPONSE,
            metadata={
                "case_id": case_id,
                "search_results_count": len(agent_result.get('search_results', [])),
                "new_cached_findings": len(new_cached_findings),
                "context_reuse_applied": context_reuse_applied,
                "synthesis_length": len(agent_result.get('synthesized_findings', ''))
            }
        )
        
        # Enhance the response with conversation memory metadata
        enhanced_response = {
            **agent_result,
            "conversation_id": conversation_id,
            "context_reuse_applied": context_reuse_applied,
            "cached_findings_count": len(cached_findings),
            "new_cached_findings_count": len(new_cached_findings),
            "conversation_memory_enabled": True,
            "memory_optimization": {
                "cached_findings_utilized": len(cached_findings),
                "new_findings_cached": len(new_cached_findings),
                "context_reuse_score": 0.8 if context_reuse_applied else 0.0
            }
        }
        
        return enhanced_response
        
    except Exception as e:
        # Log error
        if conversation_id:
            try:
                await conversation_memory.add_message(
                    conversation_id,
                    MessageRole.SYSTEM,
                    f"Enhanced agent execution failed: {str(e)}",
                    MessageType.ANALYSIS_RESPONSE,
                    metadata={"error": True, "case_id": case_id}
                )
            except Exception:
                pass  # Don't fail if logging fails
        
        # Return error response
        return {
            "case_id": case_id,
            "conversation_id": conversation_id,
            "topic": topic,
            "error": f"Enhanced agent execution failed: {str(e)}",
            "synthesized_findings": f"Investigation could not be completed due to error: {str(e)}",
            "conversation_memory_enabled": True,
            "context_reuse_applied": False
        }

async def get_conversation_history(conversation_id: str) -> Dict[str, Any]:
    """Get conversation history for display in the frontend."""
    try:
        if conversation_id not in conversation_memory.active_conversations:
            return {"messages": [], "cached_findings": {}}
        
        # Get conversation summary which includes metrics and context
        summary = await conversation_memory.get_conversation_summary(
            conversation_id, include_findings=True
        )
        
        if summary.get("error"):
            return {"messages": [], "cached_findings": {}, "error": summary["error"]}
        
        # Get the actual conversation context
        context = conversation_memory.active_conversations[conversation_id]
        
        # Format messages for frontend display
        formatted_messages = []
        for msg in context.messages:
            formatted_messages.append({
                "id": msg.message_id,
                "role": msg.role.value,
                "content": msg.content,
                "type": msg.message_type.value,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata or {}
            })
        
        return {
            "messages": formatted_messages,
            "cached_findings": context.cached_findings,
            "conversation_metadata": summary.get("metrics", {}),
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        return {
            "error": f"Failed to retrieve conversation history: {str(e)}",
            "messages": [],
            "cached_findings": {}
        }

async def continue_conversation(
    conversation_id: str,
    new_query: str,
    model_id: str = "gpt-4o",
    temperature: float = 0.1
) -> Dict[str, Any]:
    """Continue an existing conversation with a new query."""
    
    try:
        # Check if conversation exists
        if conversation_id not in conversation_memory.active_conversations:
            return {
                "error": "Conversation not found",
                "conversation_id": conversation_id
            }
        
        # Get conversation context
        context = conversation_memory.active_conversations[conversation_id]
        
        # Extract case_id from conversation metadata
        case_id = None
        for msg in context.messages:
            if msg.metadata and "case_id" in msg.metadata:
                case_id = msg.metadata["case_id"]
                break
        
        if not case_id:
            case_id = f"case_{int(time.time())}"
        
        # Run enhanced agent with conversation context
        result = await run_enhanced_agent_with_memory(
            topic=new_query,
            model_id=model_id,
            temperature=temperature,
            conversation_id=conversation_id,
            case_id=case_id
        )
        
        return result
        
    except Exception as e:
        return {
            "error": f"Failed to continue conversation: {str(e)}",
            "conversation_id": conversation_id,
            "topic": new_query
        }

async def start_new_conversation(
    topic: str,
    model_id: str = "gpt-4o",
    temperature: float = 0.1,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Start a new conversation with conversation memory tracking."""
    
    # Don't pass conversation_id - let run_enhanced_agent_with_memory create it
    result = await run_enhanced_agent_with_memory(
        topic=topic,
        model_id=model_id,
        temperature=temperature,
        conversation_id=None,  # Let the function create a new conversation
        user_id=user_id
    )
    
    return result
