"""
Enhanced Conversation Memory System for OSINT Agent.

This module extends the existing memory system to track user conversations,
chat history, and enable efficient context reuse across multiple interactions.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import timedelta
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from .agent_memory import AgentMemoryManager


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Types of conversation messages."""
    INVESTIGATION_REQUEST = "investigation_request"
    FOLLOW_UP_QUESTION = "follow_up_question"
    CLARIFICATION = "clarification"
    FINDINGS_RESPONSE = "findings_response"
    VERIFICATION_REQUEST = "verification_request"
    ANALYSIS_RESPONSE = "analysis_response"


@dataclass
class ConversationMessage:
    """Individual message in a conversation."""
    message_id: str
    role: MessageRole
    content: str
    message_type: MessageType
    timestamp: float
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    related_case_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.related_case_ids is None:
            self.related_case_ids = []
        if self.metadata is None:
            self.metadata = {}


class ConversationContext(BaseModel):
    """Context for an ongoing conversation session."""
    
    conversation_id: str = Field(description="Unique conversation identifier")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    start_time: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)
    messages: List[ConversationMessage] = Field(default_factory=list)
    active_investigations: List[str] = Field(default_factory=list)
    conversation_topics: List[str] = Field(default_factory=list)
    context_summary: str = Field("", description="AI-generated conversation summary")
    cached_findings: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_tags: List[str] = Field(default_factory=list)


class ConversationMemoryManager:
    """
    Enhanced memory manager for conversation tracking and context reuse.
    
    This class extends the existing investigation memory with conversation-level
    tracking, enabling efficient context reuse and reduced LLM token usage.
    """

    def __init__(self, base_memory_manager: AgentMemoryManager):
        """
        Initialize conversation memory manager.
        
        Args:
            base_memory_manager: Existing investigation memory manager
        """
        self.base_memory = base_memory_manager
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_cache: Dict[str, Dict] = {}
        self.context_compression_threshold = 5000  # tokens
        self.max_conversation_age = timedelta(hours=24)

    async def start_conversation(
        self, 
        user_id: Optional[str] = None,
        initial_topic: Optional[str] = None,
        session_tags: Optional[List[str]] = None
    ) -> str:
        """
        Start a new conversation session.
        
        Args:
            user_id: Optional user identifier
            initial_topic: Optional initial conversation topic
            session_tags: Optional session categorization tags
            
        Returns:
            Conversation ID for tracking
        """
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_topics=[initial_topic] if initial_topic else [],
            session_tags=session_tags or [],
            context_summary=""
        )
        
        self.active_conversations[conversation_id] = context
        print(f"🔍 [start_conversation] Stored conversation {conversation_id} in active_conversations")
        print(f"🔍 [start_conversation] Active conversations now: {list(self.active_conversations.keys())}")
        
        # Add system message for conversation start
        try:
            await self.add_message(
                conversation_id=conversation_id,
                role=MessageRole.SYSTEM,
                content=f"Conversation started for user {user_id or 'anonymous'} on topic: {initial_topic or 'general'}",
                message_type=MessageType.INVESTIGATION_REQUEST
            )
            print(f"🔍 [start_conversation] Successfully added system message")
        except Exception as e:
            print(f"❌ [start_conversation] Failed to add system message: {e}")
            raise
        
        return conversation_id

    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        message_type: MessageType,
        tokens_used: Optional[int] = None,
        processing_time: Optional[float] = None,
        confidence_score: Optional[float] = None,
        related_case_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a message to the conversation.
        
        Args:
            conversation_id: Conversation to add message to
            role: Message role (user/assistant/system)
            content: Message content
            message_type: Type of message
            tokens_used: Number of tokens used for this message
            processing_time: Time taken to process/generate message
            confidence_score: Confidence in the response (for assistant messages)
            related_case_ids: Related investigation case IDs
            metadata: Additional message metadata
            
        Returns:
            Message ID
        """
        if conversation_id not in self.active_conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        context = self.active_conversations[conversation_id]
        message_id = str(uuid.uuid4())
        
        message = ConversationMessage(
            message_id=message_id,
            role=role,
            content=content,
            message_type=message_type,
            timestamp=time.time(),
            tokens_used=tokens_used,
            processing_time=processing_time,
            confidence_score=confidence_score,
            related_case_ids=related_case_ids or [],
            metadata=metadata or {}
        )
        
        context.messages.append(message)
        context.last_activity = time.time()
        
        # Update conversation topics if it's a new investigation
        if message_type == MessageType.INVESTIGATION_REQUEST and role == MessageRole.USER:
            await self._extract_and_update_topics(context, content)
        
        # Check if conversation should be compressed
        await self._maybe_compress_conversation(context)
        
        return message_id

    async def get_relevant_context(
        self,
        conversation_id: str,
        current_query: str,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Get relevant conversation context for the current query.
        
        This method efficiently retrieves the most relevant parts of the conversation
        history to minimize token usage while maintaining context quality.
        
        Args:
            conversation_id: Conversation to get context from
            current_query: Current user query
            max_tokens: Maximum tokens to include in context
            
        Returns:
            Relevant context including messages, cached findings, and insights
        """
        if conversation_id not in self.active_conversations:
            return {"error": "Conversation not found"}
        
        context = self.active_conversations[conversation_id]
        
        # Get recent messages (always include)
        recent_messages = context.messages[-5:]  # Last 5 messages
        
        # Search for semantically relevant messages
        relevant_messages = await self._find_relevant_messages(
            context, current_query, exclude_recent=5
        )
        
        # Get cached findings relevant to current query
        relevant_findings = await self._get_relevant_cached_findings(
            context, current_query
        )
        
        # Get insights from past similar investigations
        memory_insights = await self.base_memory.get_memory_insights(
            topic=current_query,
            context={"conversation_id": conversation_id}
        )
        
        # Compile context with token management
        compiled_context = await self._compile_context(
            recent_messages=recent_messages,
            relevant_messages=relevant_messages,
            cached_findings=relevant_findings,
            memory_insights=memory_insights,
            max_tokens=max_tokens
        )
        
        return compiled_context

    async def cache_investigation_findings(
        self,
        conversation_id: str,
        case_id: str,
        findings: Dict[str, Any],
        confidence_score: float
    ) -> None:
        """
        Cache investigation findings for efficient reuse.
        
        Args:
            conversation_id: Conversation to cache findings for
            case_id: Investigation case ID
            findings: Investigation findings to cache
            confidence_score: Confidence in the findings
        """
        if conversation_id not in self.active_conversations:
            return
        
        context = self.active_conversations[conversation_id]
        
        # Create cache entry with timestamp and metadata
        cache_entry = {
            "case_id": case_id,
            "findings": findings,
            "confidence_score": confidence_score,
            "timestamp": time.time(),
            "access_count": 0,
            "topics": await self._extract_topics_from_findings(findings)
        }
        
        context.cached_findings[case_id] = cache_entry
        
        # Update active investigations
        if case_id not in context.active_investigations:
            context.active_investigations.append(case_id)

    async def get_conversation_summary(
        self,
        conversation_id: str,
        include_findings: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive conversation summary.
        
        Args:
            conversation_id: Conversation to summarize
            include_findings: Whether to include cached findings
            
        Returns:
            Conversation summary with key insights and context
        """
        if conversation_id not in self.active_conversations:
            return {"error": "Conversation not found"}
        
        context = self.active_conversations[conversation_id]
        
        # Calculate conversation metrics
        total_messages = len(context.messages)
        user_messages = [m for m in context.messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in context.messages if m.role == MessageRole.ASSISTANT]
        
        total_tokens = sum(m.tokens_used for m in context.messages if m.tokens_used)
        total_processing_time = sum(m.processing_time for m in context.messages if m.processing_time)
        
        summary: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "user_id": context.user_id,
            "duration": time.time() - context.start_time,
            "topics": context.conversation_topics,
            "metrics": {
                "total_messages": total_messages,
                "user_messages": len(user_messages),
                "assistant_messages": len(assistant_messages),
                "total_tokens": total_tokens,
                "total_processing_time": total_processing_time,
                "active_investigations": len(context.active_investigations)
            },
            "session_tags": context.session_tags
        }
        
        if include_findings:
            findings_summary = {}
            for case_id, entry in context.cached_findings.items():
                if isinstance(entry, dict):
                    findings_summary[case_id] = {
                        "confidence_score": entry.get("confidence_score", 0.0),
                        "timestamp": entry.get("timestamp", time.time()),
                        "topics": entry.get("topics", [])
                    }
                else:
                    findings_summary[case_id] = {
                        "confidence_score": 0.0,
                        "timestamp": time.time(),
                        "topics": []
                    }
            summary["cached_findings"] = findings_summary  # type: ignore
        
        return summary

    async def continue_investigation(
        self,
        conversation_id: str,
        case_id: str,
        additional_query: str
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Continue an existing investigation with additional context.
        
        Args:
            conversation_id: Conversation context
            case_id: Existing investigation to continue
            additional_query: Additional query or direction
            
        Returns:
            Tuple of (relevant_context, should_use_cache)
        """
        if conversation_id not in self.active_conversations:
            return {}, False
        
        context = self.active_conversations[conversation_id]
        
        # Check if we have cached findings for this investigation
        if case_id in context.cached_findings:
            cached_entry = context.cached_findings[case_id]
            cached_entry["access_count"] += 1
            
            # Determine if cached findings are still relevant
            cache_age = time.time() - cached_entry["timestamp"]
            cache_relevance = await self._assess_cache_relevance(
                cached_entry["findings"], additional_query
            )
            
            should_use_cache = cache_age < 3600 and cache_relevance > 0.7  # 1 hour, 70% relevance
            
            if should_use_cache:
                # Return enhanced context with cached findings
                return {
                    "cached_findings": cached_entry["findings"],
                    "confidence_score": cached_entry["confidence_score"],
                    "additional_query": additional_query,
                    "investigation_continuity": True,
                    "cache_metadata": {
                        "age_seconds": cache_age,
                        "relevance_score": cache_relevance,
                        "access_count": cached_entry["access_count"]
                    }
                }, True
        
        # If no cache or cache not relevant, return context for fresh investigation
        relevant_context = await self.get_relevant_context(
            conversation_id, additional_query
        )
        
        return relevant_context, False

    async def _extract_and_update_topics(self, context: ConversationContext, content: str) -> None:
        """Extract topics from content and update conversation topics."""
        # Simple keyword extraction - can be enhanced with NLP
        words = content.lower().split()
        potential_topics = [word for word in words if len(word) > 3]
        
        # Add unique topics
        for topic in potential_topics:
            if topic not in context.conversation_topics:
                context.conversation_topics.append(topic)

    async def _maybe_compress_conversation(self, context: ConversationContext) -> None:
        """Compress conversation if it exceeds token threshold."""
        total_tokens = sum(m.tokens_used for m in context.messages if m.tokens_used)
        
        if total_tokens > self.context_compression_threshold:
            # Keep recent messages and compress older ones
            recent_messages = context.messages[-10:]  # Keep last 10 messages
            older_messages = context.messages[:-10]
            
            # Generate summary of older messages
            if older_messages:
                summary_content = await self._generate_message_summary(older_messages)
                
                # Replace older messages with summary
                summary_message = ConversationMessage(
                    message_id=str(uuid.uuid4()),
                    role=MessageRole.SYSTEM,
                    content=f"[COMPRESSED HISTORY]: {summary_content}",
                    message_type=MessageType.ANALYSIS_RESPONSE,
                    timestamp=older_messages[0].timestamp,
                    metadata={"compression": True, "original_count": len(older_messages)}
                )
                
                context.messages = [summary_message] + recent_messages

    async def _find_relevant_messages(
        self, 
        context: ConversationContext, 
        query: str, 
        exclude_recent: int = 5
    ) -> List[ConversationMessage]:
        """Find semantically relevant messages from conversation history."""
        # Simple relevance based on keyword matching
        # Can be enhanced with embedding-based similarity
        query_words = set(query.lower().split())
        relevant_messages = []
        
        messages_to_search = context.messages[:-exclude_recent] if exclude_recent > 0 else context.messages
        
        for message in messages_to_search:
            message_words = set(message.content.lower().split())
            overlap = len(query_words.intersection(message_words))
            
            if overlap > 1:  # At least 2 word overlap
                relevant_messages.append(message)
        
        return relevant_messages[-3:]  # Return most recent 3 relevant messages

    async def _get_relevant_cached_findings(
        self, 
        context: ConversationContext, 
        query: str
    ) -> Dict[str, Any]:
        """Get cached findings relevant to current query."""
        relevant_findings = {}
        query_words = set(query.lower().split())
        
        for case_id, cache_entry in context.cached_findings.items():
            # Check topic overlap
            topic_words = set()
            for topic in cache_entry["topics"]:
                topic_words.update(topic.lower().split())
            
            overlap = len(query_words.intersection(topic_words))
            if overlap > 0:
                relevant_findings[case_id] = {
                    "findings": cache_entry["findings"],
                    "confidence_score": cache_entry["confidence_score"],
                    "relevance_score": overlap / len(query_words),
                    "age_seconds": time.time() - cache_entry["timestamp"]
                }
        
        return relevant_findings

    async def _compile_context(
        self,
        recent_messages: List[ConversationMessage],
        relevant_messages: List[ConversationMessage],
        cached_findings: Dict[str, Any],
        memory_insights: List[Any],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Compile context within token limits."""
        context = {
            "recent_conversation": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "type": msg.message_type.value,
                    "timestamp": msg.timestamp
                }
                for msg in recent_messages
            ],
            "relevant_history": [
                {
                    "role": msg.role.value,
                    "content": msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                    "type": msg.message_type.value,
                    "timestamp": msg.timestamp
                }
                for msg in relevant_messages
            ],
            "cached_findings": cached_findings,
            "memory_insights": [
                {
                    "case_id": insight.similar_case_id,
                    "similarity": insight.similarity_score,
                    "strategies": insight.relevant_strategies,
                    "tools": insight.recommended_tools
                }
                for insight in memory_insights[:3]  # Limit to top 3
            ]
        }
        
        return context

    async def _extract_topics_from_findings(self, findings: Dict[str, Any]) -> List[str]:
        """Extract topics from investigation findings."""
        topics = []
        
        # Extract from summary
        if "summary" in findings:
            words = findings["summary"].lower().split()
            topics.extend([word for word in words if len(word) > 4])
        
        # Extract from entities if available
        if "entities" in findings:
            topics.extend(findings["entities"])
        
        return list(set(topics))[:10]  # Limit to 10 unique topics

    async def _assess_cache_relevance(self, cached_findings: Dict[str, Any], new_query: str) -> float:
        """Assess relevance of cached findings to new query."""
        query_words = set(new_query.lower().split())
        
        # Check relevance based on content overlap
        finding_text = json.dumps(cached_findings).lower()
        finding_words = set(finding_text.split())
        
        overlap = len(query_words.intersection(finding_words))
        relevance = overlap / len(query_words) if query_words else 0
        
        return min(relevance, 1.0)

    async def _generate_message_summary(self, messages: List[ConversationMessage]) -> str:
        """Generate a summary of message history."""
        # Simple summary generation - can be enhanced with LLM
        topics = []
        user_queries = []
        
        for msg in messages:
            if msg.role == MessageRole.USER:
                user_queries.append(msg.content[:100])
            if msg.message_type == MessageType.INVESTIGATION_REQUEST:
                topics.append(msg.content[:50])
        
        summary = f"Previous conversation covered {len(topics)} topics including: {', '.join(topics[:3])}. "
        summary += f"User made {len(user_queries)} queries. "
        
        return summary[:200]  # Limit summary length


# Factory function for creating enhanced conversation memory manager
def create_conversation_memory_manager(base_memory: Optional[AgentMemoryManager] = None) -> ConversationMemoryManager:
    """
    Create a conversation memory manager with base memory integration.
    
    Args:
        base_memory: Optional base memory manager (will create if not provided)
        
    Returns:
        Configured ConversationMemoryManager instance
    """
    if base_memory is None:
        from .agent_memory import create_agent_memory_manager
        base_memory = create_agent_memory_manager()
    
    return ConversationMemoryManager(base_memory)
