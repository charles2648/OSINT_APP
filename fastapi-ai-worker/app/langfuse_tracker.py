# Enhanced Langfuse tracking integration for the OSINT agent.

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Enhanced control options
LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
LANGFUSE_DEBUG = os.getenv("LANGFUSE_DEBUG", "false").lower() == "true"

# Initialize Langfuse with error handling
langfuse = None
try:
    if LANGFUSE_ENABLED:
        from langfuse import Langfuse  # type: ignore
        
        langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            debug=LANGFUSE_DEBUG
        )
        
        if LANGFUSE_DEBUG:
            logger.info("✅ Langfuse tracking enabled with debug mode")
        else:
            logger.info("✅ Langfuse tracking enabled")
    else:
        logger.info("ℹ️ Langfuse tracking disabled via LANGFUSE_ENABLED=false")
        
except ImportError:
    logger.warning("⚠️ Langfuse not installed - tracking disabled")
    LANGFUSE_ENABLED = False
except Exception as e:
    logger.error(f"❌ Failed to initialize Langfuse: {e}")
    LANGFUSE_ENABLED = False

class OSINTAgentTracker:
    """Enhanced OSINT agent tracker with better error handling."""
    
    def __init__(self):
        self.current_trace = None
        self.current_spans = {}
        self.agent_metadata = {}
    
    def start_agent_trace(self, case_id: str, topic: str, model_id: str, temperature: float, 
                         long_term_memory: Optional[List[Dict]] = None) -> str:
        """Start a new trace for an agent execution."""
        trace_id = f"osint_agent_{case_id}_{int(time.time())}"
        
        self.agent_metadata = {
            "case_id": case_id,
            "topic": topic,
            "model_id": model_id,
            "temperature": temperature,
            "memory_size": len(long_term_memory) if long_term_memory else 0,
            "start_time": datetime.now().isoformat()
        }
        
        if not LANGFUSE_ENABLED or not langfuse:
            logger.debug("Langfuse disabled - skipping trace creation")
            return trace_id
        
        try:
            self.current_trace = langfuse.trace(
                id=trace_id,
                name="OSINT Agent Research",
                input={
                    "topic": topic,
                    "case_id": case_id,
                    "model_id": model_id,
                    "temperature": temperature,
                    "long_term_memory": long_term_memory or []
                },
                metadata=self.agent_metadata,
                tags=["osint", "research", "agent", model_id.split("/")[-1]]
            )
            logger.debug(f"Created Langfuse trace: {trace_id}")
        except Exception as e:
            logger.error(f"Failed to create Langfuse trace: {e}")
            self.current_trace = None
        
        return trace_id
    
    def end_agent_trace(self, final_state: Dict[str, Any], success: bool = True):
        """End the current agent trace with results."""
        if not self.current_trace:
            return
        
        try:
            output_data = {
                "synthesized_findings": final_state.get("synthesized_findings", ""),
                "search_queries": final_state.get("search_queries", []),
                "num_steps": final_state.get("num_steps", 0),
                "verified_data": final_state.get("verified_data", {}),
                "mcp_verification_count": len(final_state.get("mcp_verification_list", [])),
                "success": success,
                "end_time": datetime.now().isoformat()
            }
            
            execution_time = 0.0
            if self.agent_metadata.get("start_time"):
                start_time = datetime.fromisoformat(self.agent_metadata["start_time"])
                execution_time = (datetime.now() - start_time).total_seconds()
            
            self.current_trace.update(
                output=output_data,
                metadata={
                    **self.agent_metadata,
                    "execution_time_seconds": execution_time,
                    "total_steps": final_state.get("num_steps", 0),
                    "verification_tasks": len(final_state.get("mcp_verification_list", [])),
                    "domains_verified": len(final_state.get("verified_data", {}))
                }
            )
            logger.debug("Updated Langfuse trace with final results")
        except Exception as e:
            logger.error(f"Failed to update Langfuse trace: {e}")
        
        # Reset for next execution
        self.current_trace = None
        self.current_spans = {}
        self.agent_metadata = {}
    
    @contextmanager
    def track_node_execution(self, node_name: str, input_data: Dict[str, Any]):
        """Context manager to track individual node execution."""
        if not self.current_trace or not LANGFUSE_ENABLED:
            yield
            return
        
        start_time = time.time()
        span = None
        
        try:
            span = self.current_trace.span(
                name=f"{node_name}_node",
                input=input_data,
                metadata={
                    "node_type": node_name,
                    "start_time": datetime.now().isoformat()
                }
            )
            self.current_spans[node_name] = span
            
            yield span
            
        except Exception as e:
            logger.error(f"Failed to create span for {node_name}: {e}")
            yield
            
        finally:
            if span:
                try:
                    execution_time = time.time() - start_time
                    span.update(
                        metadata={
                            "node_type": node_name,
                            "execution_time_seconds": execution_time,
                            "end_time": datetime.now().isoformat()
                        }
                    )
                    span.end()
                    logger.debug(f"Completed span for {node_name}")
                except Exception as e:
                    logger.error(f"Failed to end span for {node_name}: {e}")
    
    def track_llm_call(self, node_name: str, model_id: str, prompt: str, response: Any, temperature: float):
        """Track LLM API calls."""
        if not LANGFUSE_ENABLED or not langfuse:
            return
        
        try:
            _ = langfuse.generation(
                name=f"{node_name}_llm_call",
                model=model_id,
                input=prompt,
                output=str(response),
                metadata={
                    "temperature": temperature,
                    "node": node_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.debug(f"Tracked LLM call for {node_name}")
        except Exception as e:
            logger.error(f"Failed to track LLM call: {e}")
    
    def track_search_operation(self, query: str, results: List[Dict], execution_time: float, success: bool):
        """Track search operations."""
        if not LANGFUSE_ENABLED or not langfuse:
            return
        
        try:
            _ = langfuse.span(
                name="web_search",
                input={"query": query},
                output={"results_count": len(results), "success": success},
                metadata={
                    "execution_time_seconds": execution_time,
                    "results_count": len(results),
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.debug(f"Tracked search operation: {query[:50]}...")
        except Exception as e:
            logger.error(f"Failed to track search operation: {e}")
    
    def track_mcp_operation(self, mcp_name: str, input_data: str, result: Dict, execution_time: float):
        """Track MCP verification operations."""
        if not LANGFUSE_ENABLED or not langfuse:
            return
        
        try:
            success = "error" not in result
            _ = langfuse.span(
                name=f"mcp_{mcp_name}",
                input={"mcp_tool": mcp_name, "input": input_data},
                output={"success": success, "result_keys": list(result.keys())},
                metadata={
                    "mcp_name": mcp_name,
                    "execution_time_seconds": execution_time,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                },
                tags=["mcp", "verification", mcp_name]
            )
            logger.debug(f"Tracked MCP operation: {mcp_name}")
        except Exception as e:
            logger.error(f"Failed to track MCP operation: {e}")
    
    def track_user_feedback(self, case_id: str, feedback_type: str, feedback_data: Dict):
        """Track user feedback."""
        if not LANGFUSE_ENABLED or not langfuse:
            return
            
        try:
            _ = langfuse.trace(
                name="User Feedback",
                input={"case_id": case_id, "feedback_type": feedback_type},
                output=feedback_data,
                metadata={
                    "case_id": case_id,
                    "feedback_type": feedback_type,
                    "timestamp": datetime.now().isoformat()
                },
                tags=["feedback", feedback_type]
            )
            logger.debug(f"Tracked user feedback: {feedback_type}")
        except Exception as e:
            logger.error(f"Failed to track user feedback: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        if not LANGFUSE_ENABLED or not langfuse:
            return {"status": "disabled", "note": "Langfuse tracking is disabled"}
        
        return {
            "status": "enabled",
            "free_tier_limit": 50000,
            "note": "Check your Langfuse dashboard for detailed usage statistics",
            "recommendation": "Monitor monthly trace count to stay within free tier"
        }

# Global tracker instance
agent_tracker = OSINTAgentTracker()

# Convenience functions for compatibility
def track_agent_session_start(case_id: str, topic: str, model_id: str, temperature: float, 
                             long_term_memory: Optional[List[Dict]] = None) -> str:
    """Start tracking an agent session."""
    return agent_tracker.start_agent_trace(case_id, topic, model_id, temperature, long_term_memory)

def track_agent_session_end(final_state: Dict[str, Any], success: bool = True):
    """End tracking an agent session."""
    agent_tracker.end_agent_trace(final_state, success)

def get_langfuse_client():
    """Get the Langfuse client instance for custom tracking."""
    return langfuse

def is_langfuse_enabled() -> bool:
    """Check if Langfuse tracking is enabled."""
    return LANGFUSE_ENABLED and langfuse is not None
