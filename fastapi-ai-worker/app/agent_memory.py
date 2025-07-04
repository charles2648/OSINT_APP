"""
Agent Memory Integration Layer for OSINT investigations.

This module provides the integration layer between the OSINT agent workflow and the vector memory system,
enabling seamless memory storage, retrieval, and learning capabilities.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .vector_memory import MemoryEntry, VectorMemoryManager, VECTOR_DEPS_AVAILABLE

logger = logging.getLogger(__name__)


class InvestigationContext(BaseModel):
    """Context for an ongoing OSINT investigation."""
    
    case_id: str = Field(description="Unique identifier for the investigation")
    topic: str = Field(description="Investigation topic or subject")
    start_time: float = Field(default_factory=time.time)
    search_queries: List[str] = Field(default_factory=list)
    mcp_tools_executed: List[str] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    successful_strategies: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    entity_relationships: Dict[str, List[str]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class MemoryInsight(BaseModel):
    """Insight retrieved from memory for current investigation."""
    
    similar_case_id: str = Field(description="ID of similar past case")
    similarity_score: float = Field(description="Similarity score")
    relevant_strategies: List[str] = Field(description="Relevant investigation strategies")
    recommended_tools: List[str] = Field(description="Recommended MCP tools")
    potential_challenges: List[str] = Field(description="Potential challenges to expect")
    confidence_indicators: Dict[str, Any] = Field(description="Confidence assessment indicators")
    reasoning: str = Field(description="Why this memory is relevant")


class AgentMemoryManager:
    """
    Memory integration manager for OSINT agents.
    
    This class provides high-level memory operations for the agent workflow,
    including storing investigation results, retrieving relevant memories,
    and generating actionable insights for new investigations.
    """
    
    def __init__(self, vector_memory: Optional[VectorMemoryManager] = None):
        """
        Initialize the agent memory manager.
        
        Args:
            vector_memory: Vector memory manager instance (optional, will create if not provided)
        """
        self.vector_memory = vector_memory
        self.current_context: Optional[InvestigationContext] = None
        self.memory_enabled = VECTOR_DEPS_AVAILABLE
        
        if not self.memory_enabled:
            logger.warning("Vector memory dependencies not available - memory features disabled")
        elif self.vector_memory is None:
            try:
                from .vector_memory import create_vector_memory_manager
                self.vector_memory = create_vector_memory_manager()
                logger.info("Created default vector memory manager")
            except Exception as e:
                logger.error(f"Failed to initialize vector memory: {e}")
                self.memory_enabled = False
    
    def start_investigation(self, case_id: str, topic: str, tags: Optional[List[str]] = None) -> InvestigationContext:
        """
        Start a new investigation and initialize context.
        
        Args:
            case_id: Unique identifier for the investigation
            topic: Investigation topic or subject
            tags: Optional categorization tags
            
        Returns:
            Investigation context for tracking
        """
        self.current_context = InvestigationContext(
            case_id=case_id,
            topic=topic,
            tags=tags or []
        )
        
        logger.info(f"Started investigation: {case_id} - {topic}")
        return self.current_context
    
    async def get_memory_insights(self, topic: str, context: Optional[Dict[str, Any]] = None) -> List[MemoryInsight]:
        """
        Retrieve memory insights for the current investigation topic.
        
        Args:
            topic: Investigation topic to search for
            context: Additional context for memory retrieval
            
        Returns:
            List of relevant memory insights
        """
        if not self.memory_enabled or not self.vector_memory:
            logger.debug("Memory system not available - returning empty insights")
            return []
        
        try:
            # Search for similar memories
            similar_memories = await self.vector_memory.search_similar_memories(
                query=topic,
                case_context=context,
                limit=5
            )
            
            insights = []
            for memory_result in similar_memories:
                memory = memory_result.entry
                
                # Extract insights from memory
                insight = MemoryInsight(
                    similar_case_id=memory.case_id,
                    similarity_score=memory_result.similarity_score,
                    relevant_strategies=memory.successful_strategies,
                    recommended_tools=memory.mcp_tools_used,
                    potential_challenges=memory.challenges_encountered,
                    confidence_indicators={
                        "avg_confidence": sum(memory.confidence_scores.values()) / len(memory.confidence_scores) if memory.confidence_scores else 0.0,
                        "verification_status": memory.verification_status,
                        "total_sources": memory.total_sources
                    },
                    reasoning=memory_result.relevance_explanation
                )
                
                insights.append(insight)
            
            logger.info(f"Retrieved {len(insights)} memory insights for topic: {topic}")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory insights: {e}")
            return []
    
    def record_search_query(self, query: str) -> None:
        """Record a search query in the current investigation context."""
        if self.current_context:
            self.current_context.search_queries.append(query)
            logger.debug(f"Recorded search query: {query}")
    
    def record_mcp_execution(self, tool_name: str) -> None:
        """Record MCP tool execution in the current investigation context."""
        if self.current_context:
            self.current_context.mcp_tools_executed.append(tool_name)
            logger.debug(f"Recorded MCP execution: {tool_name}")
    
    def record_finding(self, finding: Dict[str, Any], confidence: Optional[float] = None) -> None:
        """
        Record a finding in the current investigation context.
        
        Args:
            finding: The finding data
            confidence: Optional confidence score for the finding
        """
        if self.current_context:
            self.current_context.findings.append(finding)
            
            if confidence is not None:
                finding_id = finding.get("id", f"finding_{len(self.current_context.findings)}")
                self.current_context.confidence_scores[finding_id] = confidence
            
            logger.debug(f"Recorded finding with confidence {confidence}")
    
    def record_strategy_success(self, strategy: str) -> None:
        """Record a successful investigation strategy."""
        if self.current_context and strategy not in self.current_context.successful_strategies:
            self.current_context.successful_strategies.append(strategy)
            logger.debug(f"Recorded successful strategy: {strategy}")
    
    def record_challenge(self, challenge: str) -> None:
        """Record a challenge encountered during investigation."""
        if self.current_context and challenge not in self.current_context.challenges:
            self.current_context.challenges.append(challenge)
            logger.debug(f"Recorded challenge: {challenge}")
    
    def record_entity_relationship(self, entity: str, related_entities: List[str]) -> None:
        """Record entity relationships discovered during investigation."""
        if self.current_context:
            if entity not in self.current_context.entity_relationships:
                self.current_context.entity_relationships[entity] = []
            
            for related in related_entities:
                if related not in self.current_context.entity_relationships[entity]:
                    self.current_context.entity_relationships[entity].append(related)
            
            logger.debug(f"Recorded entity relationships for: {entity}")
    
    async def finalize_investigation(
        self,
        verification_status: str = "completed",
        additional_tags: Optional[List[str]] = None
    ) -> bool:
        """
        Finalize the current investigation and store it in memory.
        
        Args:
            verification_status: Final verification status
            additional_tags: Additional tags to add to the investigation
            
        Returns:
            True if successfully stored, False otherwise
        """
        if not self.current_context:
            logger.warning("No active investigation context to finalize")
            return False
        
        if not self.memory_enabled or not self.vector_memory:
            logger.debug("Memory system not available - skipping storage")
            return False
        
        try:
            # Calculate investigation duration
            investigation_duration = time.time() - self.current_context.start_time
            
            # Merge additional tags
            final_tags = list(set(self.current_context.tags + (additional_tags or [])))
            
            # Create memory entry
            memory_entry = MemoryEntry(
                case_id=self.current_context.case_id,
                topic=self.current_context.topic,
                timestamp=self.current_context.start_time,
                search_queries=self.current_context.search_queries,
                findings=self.current_context.findings,
                mcp_tools_used=self.current_context.mcp_tools_executed,
                confidence_scores=self.current_context.confidence_scores,
                successful_strategies=self.current_context.successful_strategies,
                challenges_encountered=self.current_context.challenges,
                entity_relationships=self.current_context.entity_relationships,
                investigation_duration=investigation_duration,
                total_sources=len(set(self.current_context.mcp_tools_executed)),
                verification_status=verification_status,
                tags=final_tags
            )
            
            # Store in vector memory
            success = await self.vector_memory.store_memory(memory_entry)
            
            if success:
                logger.info(f"Successfully stored investigation memory: {self.current_context.case_id}")
            else:
                logger.error(f"Failed to store investigation memory: {self.current_context.case_id}")
            
            # Clear current context
            self.current_context = None
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to finalize investigation: {e}")
            return False
    
    async def get_investigation_patterns(self, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get investigation patterns and insights from stored memories.
        
        Args:
            topic_filter: Optional filter for specific topic patterns
            
        Returns:
            Analysis of investigation patterns
        """
        if not self.memory_enabled or not self.vector_memory:
            return {"error": "Memory system not available"}
        
        try:
            patterns = await self.vector_memory.get_investigation_patterns(topic_filter)
            return patterns
        except Exception as e:
            logger.error(f"Failed to get investigation patterns: {e}")
            return {"error": str(e)}
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        if not self.memory_enabled or not self.vector_memory:
            return {
                "memory_enabled": False,
                "error": "Memory system not available"
            }
        
        try:
            stats = await self.vector_memory.get_memory_stats()
            return {
                "memory_enabled": True,
                **stats
            }
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {
                "memory_enabled": False,
                "error": str(e)
            }
    
    def get_current_context(self) -> Optional[InvestigationContext]:
        """Get the current investigation context."""
        return self.current_context
    
    async def search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search stored memories with a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching memory entries
        """
        if not self.memory_enabled or not self.vector_memory:
            return []
        
        try:
            results = await self.vector_memory.search_similar_memories(query, limit=limit)
            return [
                {
                    "case_id": result.entry.case_id,
                    "topic": result.entry.topic,
                    "timestamp": result.entry.timestamp,
                    "similarity_score": result.similarity_score,
                    "relevance_explanation": result.relevance_explanation,
                    "findings_count": len(result.entry.findings),
                    "tools_used": result.entry.mcp_tools_used,
                    "tags": result.entry.tags
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []


# Factory function for easy initialization
def create_agent_memory_manager() -> AgentMemoryManager:
    """
    Factory function to create an agent memory manager with default settings.
    
    Returns:
        Configured AgentMemoryManager instance
    """
    return AgentMemoryManager()
