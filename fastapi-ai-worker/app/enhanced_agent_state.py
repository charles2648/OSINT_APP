"""
Enhanced Agent State and Integration for Conversation Memory.

This module provides enhanced agent state management that integrates
conversation memory for improved context tracking and efficient reuse.
"""

from typing import Dict, List, Any, Optional
from typing_extensions import TypedDict

from .conversation_memory import ConversationMemoryManager, MessageRole, MessageType


class EnhancedAgentState(TypedDict):
    """Enhanced agent state with conversation memory integration."""
    
    # Original agent state fields
    topic: str
    case_id: str
    model_id: str
    temperature: float
    long_term_memory: List[Dict]
    search_queries: List[str]
    search_results: List[Dict]
    synthesized_findings: str
    num_steps: int
    mcp_verification_list: List[Dict]
    verified_data: Dict
    planner_reasoning: str
    synthesis_confidence: str
    information_gaps: List[str]
    search_quality_metrics: Dict
    query_performance: List[Dict]
    mcp_execution_results: List[Dict]
    verification_strategy: str
    final_confidence_assessment: str
    final_risk_indicators: List[str]
    final_verification_summary: str
    final_actionable_recommendations: List[str]
    final_information_reliability: str
    report_quality_metrics: Dict
    
    # Enhanced conversation and memory fields
    conversation_id: Optional[str]
    user_id: Optional[str]
    conversation_context: Dict[str, Any]
    cached_findings: Dict[str, Any]
    memory_insights: List[Dict[str, Any]]
    context_reuse_score: float
    token_optimization_applied: bool
    conversation_continuity: bool
    user_prompt_history: List[Dict[str, Any]]
    assistant_response_history: List[Dict[str, Any]]
    investigation_thread: List[str]  # Linked investigation IDs
    context_compression_applied: bool


class ConversationAwareAgentEnhancer:
    """
    Enhancer class that adds conversation memory capabilities to the existing agent.
    
    This class wraps around the existing agent workflow to provide:
    - Conversation tracking
    - Context reuse optimization
    - Token usage reduction
    - Memory-driven insights
    """
    
    def __init__(self, conversation_memory: ConversationMemoryManager):
        """
        Initialize the agent enhancer.
        
        Args:
            conversation_memory: Conversation memory manager instance
        """
        self.conversation_memory = conversation_memory
        
    async def enhance_agent_state(
        self,
        original_state: Dict[str, Any],
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> EnhancedAgentState:
        """
        Enhance the original agent state with conversation memory.
        
        Args:
            original_state: Original agent state from request
            conversation_id: Optional conversation ID for continuity
            user_id: Optional user ID for personalization
            
        Returns:
            Enhanced agent state with memory context
        """
        # Start or continue conversation
        if not conversation_id:
            conversation_id = await self.conversation_memory.start_conversation(
                user_id=user_id,
                initial_topic=original_state.get("topic"),
                session_tags=["osint_investigation"]
            )
        
        # Get relevant conversation context
        context = await self.conversation_memory.get_relevant_context(
            conversation_id=conversation_id,
            current_query=original_state.get("topic", ""),
            max_tokens=2000
        )
        
        # Check for investigation continuity
        continuation_context, should_use_cache = await self.conversation_memory.continue_investigation(
            conversation_id=conversation_id,
            case_id=original_state.get("case_id", ""),
            additional_query=original_state.get("topic", "")
        )
        
        # Calculate context reuse score
        reuse_score = await self._calculate_context_reuse_score(context, continuation_context)
        
        # Extract memory insights
        memory_insights = context.get("memory_insights", [])
        
        # Build enhanced state - manually construct to avoid TypedDict expansion issues
        enhanced_state = EnhancedAgentState(
            # Original state fields
            topic=original_state.get("topic", ""),
            case_id=original_state.get("case_id", ""),
            model_id=original_state.get("model_id", "gpt-4o"),
            temperature=original_state.get("temperature", 0.1),
            long_term_memory=original_state.get("long_term_memory", []),
            search_queries=original_state.get("search_queries", []),
            search_results=original_state.get("search_results", []),
            synthesized_findings=original_state.get("synthesized_findings", ""),
            num_steps=original_state.get("num_steps", 0),
            mcp_verification_list=original_state.get("mcp_verification_list", []),
            verified_data=original_state.get("verified_data", {}),
            planner_reasoning=original_state.get("planner_reasoning", ""),
            synthesis_confidence=original_state.get("synthesis_confidence", ""),
            information_gaps=original_state.get("information_gaps", []),
            search_quality_metrics=original_state.get("search_quality_metrics", {}),
            query_performance=original_state.get("query_performance", []),
            mcp_execution_results=original_state.get("mcp_execution_results", []),
            verification_strategy=original_state.get("verification_strategy", ""),
            final_confidence_assessment=original_state.get("final_confidence_assessment", ""),
            final_risk_indicators=original_state.get("final_risk_indicators", []),
            final_verification_summary=original_state.get("final_verification_summary", ""),
            final_actionable_recommendations=original_state.get("final_actionable_recommendations", []),
            final_information_reliability=original_state.get("final_information_reliability", ""),
            report_quality_metrics=original_state.get("report_quality_metrics", {}),
            
            # Add conversation context
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_context=context,
            cached_findings=continuation_context.get("cached_findings", {}),
            memory_insights=memory_insights,
            context_reuse_score=reuse_score,
            token_optimization_applied=should_use_cache,
            conversation_continuity=continuation_context.get("investigation_continuity", False),
            user_prompt_history=context.get("recent_conversation", []),
            assistant_response_history=[],
            investigation_thread=[original_state.get("case_id", "")],
            context_compression_applied=False
        )
        
        # Log the user's request
        await self.conversation_memory.add_message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=original_state.get("topic", ""),
            message_type=MessageType.INVESTIGATION_REQUEST,
            related_case_ids=[original_state.get("case_id", "")]
        )
        
        return enhanced_state
    
    async def optimize_planner_with_memory(
        self,
        state: EnhancedAgentState,
        planner_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize planner output using conversation memory insights.
        
        Args:
            state: Enhanced agent state
            planner_output: Original planner output
            
        Returns:
            Optimized planner output with memory insights
        """
        # Get memory insights for planning optimization
        memory_insights = state.get("memory_insights", [])
        cached_findings = state.get("cached_findings", {})
        
        optimized_output = planner_output.copy()
        
        # Enhance search queries with memory insights
        if memory_insights:
            recommended_strategies = []
            for insight in memory_insights:
                recommended_strategies.extend(insight.get("strategies", []))
            
            # Add memory-informed reasoning
            memory_context = f"\n\nMemory Context: Based on {len(memory_insights)} similar past investigations, "
            memory_context += f"recommended strategies include: {', '.join(recommended_strategies[:3])}"
            
            optimized_output["reasoning"] = planner_output.get("reasoning", "") + memory_context
        
        # Optimize queries if we have cached findings
        if cached_findings and state.get("conversation_continuity", False):
            # Modify queries to focus on gaps in cached findings
            existing_topics = []
            for case_id, findings in cached_findings.items():
                if isinstance(findings.get("findings"), dict):
                    summary = findings["findings"].get("summary", "")
                    existing_topics.extend(summary.lower().split())
            
            # Filter out redundant queries
            original_queries = planner_output.get("search_queries", [])
            optimized_queries = []
            
            for query in original_queries:
                query_words = set(query.lower().split())
                overlap = len(query_words.intersection(set(existing_topics)))
                
                if overlap < len(query_words) * 0.7:  # Less than 70% overlap
                    optimized_queries.append(query)
                else:
                    # Modify query to focus on new aspects
                    modified_query = f"{query} recent updates new information"
                    optimized_queries.append(modified_query)
            
            optimized_output["search_queries"] = optimized_queries
            
            # Add context about query optimization
            optimization_note = f"\n\nQuery Optimization: Modified {len(original_queries) - len(optimized_queries)} queries to avoid redundancy with cached findings."
            optimized_output["reasoning"] += optimization_note
        
        return optimized_output
    
    async def track_synthesis_with_memory(
        self,
        state: EnhancedAgentState,
        synthesis_output: str,
        confidence_score: float
    ) -> None:
        """
        Track synthesis output in conversation memory.
        
        Args:
            state: Enhanced agent state
            synthesis_output: Generated synthesis
            confidence_score: Confidence in synthesis
        """
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            return
        
        # Add assistant message for synthesis
        await self.conversation_memory.add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=synthesis_output,
            message_type=MessageType.FINDINGS_RESPONSE,
            confidence_score=confidence_score,
            related_case_ids=[state.get("case_id", "")]
        )
        
        # Cache findings for future use
        findings_data = {
            "summary": synthesis_output,
            "confidence": confidence_score,
            "search_queries": state.get("search_queries", []),
            "search_results": state.get("search_results", []),
            "timestamp": state.get("num_steps", 0)
        }
        
        await self.conversation_memory.cache_investigation_findings(
            conversation_id=conversation_id,
            case_id=state.get("case_id", ""),
            findings=findings_data,
            confidence_score=confidence_score
        )
    
    async def generate_memory_enhanced_recommendations(
        self,
        state: EnhancedAgentState
    ) -> List[str]:
        """
        Generate recommendations enhanced with memory insights.
        
        Args:
            state: Enhanced agent state
            
        Returns:
            List of memory-enhanced recommendations
        """
        recommendations = []
        memory_insights = state.get("memory_insights", [])
        
        # Add recommendations based on memory insights
        if memory_insights:
            for insight in memory_insights:
                tools = insight.get("tools", [])
                if tools:
                    recommendations.append(
                        f"Consider using {', '.join(tools[:2])} based on similar past investigations"
                    )
        
        # Add recommendations based on conversation continuity
        if state.get("conversation_continuity", False):
            recommendations.append(
                "This investigation builds on previous findings - focus on new aspects and verification"
            )
        
        # Add token optimization recommendations
        if state.get("token_optimization_applied", False):
            recommendations.append(
                "Leveraging cached findings to reduce processing time and improve efficiency"
            )
        
        # Add context-specific recommendations
        cached_findings = state.get("cached_findings", {})
        if cached_findings:
            confidence_scores = [
                entry.get("confidence_score", 0) 
                for entry in cached_findings.values()
            ]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            if avg_confidence < 0.7:
                recommendations.append(
                    "Previous findings had moderate confidence - prioritize verification with additional sources"
                )
        
        return recommendations
    
    async def finalize_conversation_investigation(
        self,
        state: EnhancedAgentState,
        final_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Finalize investigation with conversation memory integration.
        
        Args:
            state: Enhanced agent state
            final_output: Final investigation output
            
        Returns:
            Enhanced final output with memory context
        """
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            return final_output
        
        # Add final response to conversation
        await self.conversation_memory.add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=final_output.get("synthesized_findings", ""),
            message_type=MessageType.ANALYSIS_RESPONSE,
            confidence_score=float(final_output.get("final_confidence_assessment", "0.5")),
            related_case_ids=[state.get("case_id", "")]
        )
        
        # Get conversation summary for context
        conversation_summary = await self.conversation_memory.get_conversation_summary(
            conversation_id=conversation_id,
            include_findings=True
        )
        
        # Enhance final output with conversation context
        enhanced_output = final_output.copy()
        enhanced_output.update({
            "conversation_context": {
                "conversation_id": conversation_id,
                "total_messages": conversation_summary.get("metrics", {}).get("total_messages", 0),
                "investigation_continuity": state.get("conversation_continuity", False),
                "context_reuse_score": state.get("context_reuse_score", 0.0),
                "token_optimization": state.get("token_optimization_applied", False)
            },
            "memory_enhanced_recommendations": await self.generate_memory_enhanced_recommendations(state)
        })
        
        return enhanced_output
    
    async def _calculate_context_reuse_score(
        self,
        context: Dict[str, Any],
        continuation_context: Dict[str, Any]
    ) -> float:
        """Calculate how much context can be reused to optimize token usage."""
        score = 0.0
        
        # Score based on cached findings
        if continuation_context.get("cached_findings"):
            score += 0.4
        
        # Score based on relevant history
        relevant_history = context.get("relevant_history", [])
        if relevant_history:
            score += 0.3
        
        # Score based on memory insights
        memory_insights = context.get("memory_insights", [])
        if memory_insights:
            score += 0.3
        
        return min(score, 1.0)


# Factory function for creating conversation-aware agent enhancer
def create_conversation_aware_enhancer(
    conversation_memory: Optional[ConversationMemoryManager] = None
) -> ConversationAwareAgentEnhancer:
    """
    Create a conversation-aware agent enhancer.
    
    Args:
        conversation_memory: Optional conversation memory manager
        
    Returns:
        Configured ConversationAwareAgentEnhancer instance
    """
    if conversation_memory is None:
        from .conversation_memory import create_conversation_memory_manager
        conversation_memory = create_conversation_memory_manager()
    
    return ConversationAwareAgentEnhancer(conversation_memory)
