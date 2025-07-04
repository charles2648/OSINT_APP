"""
Vector-based long-term memory system for OSINT agents using ChromaDB and Sentence Transformers.

This module provides persistent memory storage and semantic retrieval capabilities for the OSINT agent,
enabling it to learn from past investigations and apply insights to new cases.
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    import numpy as np
    VECTOR_DEPS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Vector memory dependencies not available: {e}")
    VECTOR_DEPS_AVAILABLE = False
    # Create placeholder classes for type hints
    class SentenceTransformer:
        pass
    class Settings:
        pass

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryEntry(BaseModel):
    """Structured memory entry for OSINT investigations."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str = Field(description="Unique identifier for the investigation case")
    topic: str = Field(description="Investigation topic or subject")
    timestamp: float = Field(default_factory=time.time)
    
    # Investigation content
    search_queries: List[str] = Field(default_factory=list, description="Search queries executed")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Key findings")
    mcp_tools_used: List[str] = Field(default_factory=list, description="MCP tools executed")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores")
    
    # Learning insights
    successful_strategies: List[str] = Field(default_factory=list, description="Successful investigation approaches")
    challenges_encountered: List[str] = Field(default_factory=list, description="Challenges and how they were resolved")
    entity_relationships: Dict[str, List[str]] = Field(default_factory=dict, description="Entity relationships discovered")
    
    # Metadata
    investigation_duration: float = Field(default=0.0, description="Duration in seconds")
    total_sources: int = Field(default=0, description="Number of sources consulted")
    verification_status: str = Field(default="pending", description="Verification status")
    tags: List[str] = Field(default_factory=list, description="Categorization tags")


class MemorySearchResult(BaseModel):
    """Result from memory search operations."""
    
    entry: MemoryEntry
    similarity_score: float = Field(ge=0.0, le=1.0, description="Semantic similarity score")
    relevance_explanation: str = Field(description="Why this memory is relevant")


class VectorMemoryManager:
    """
    Advanced vector-based memory system for OSINT investigations.
    
    Features:
    - Semantic similarity search using sentence transformers
    - Persistent storage with ChromaDB
    - Contextual memory retrieval
    - Investigation pattern learning
    - Entity relationship tracking
    """
    
    def __init__(
        self,
        db_path: str = "./data/memory_db",
        collection_name: str = "osint_investigations",
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ):
        """
        Initialize the vector memory manager.
        
        Args:
            db_path: Path to ChromaDB storage directory
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Minimum similarity score for relevant results
            max_results: Maximum number of results to return
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "Vector memory dependencies not available. "
                "Please install: pip install chromadb sentence-transformers numpy"
            )
        
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self._initialize_database()
        
        # Performance metrics
        self.metrics = {
            "total_memories": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_similarity": 0.0
        }
    
    def _initialize_database(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "OSINT investigation memories with semantic search"}
            )
            
            # Update metrics
            self.metrics["total_memories"] = self.collection.count()
            
            logger.info(f"Vector memory initialized with {self.metrics['total_memories']} existing memories")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector memory database: {e}")
            raise
    
    def _create_memory_text(self, entry: MemoryEntry) -> str:
        """
        Create searchable text representation of a memory entry.
        
        Args:
            entry: Memory entry to convert to text
            
        Returns:
            Concatenated text for semantic search
        """
        text_components = [
            f"Topic: {entry.topic}",
            f"Search queries: {', '.join(entry.search_queries)}",
            f"Key findings: {', '.join([str(f) for f in entry.findings])}",
            f"Tools used: {', '.join(entry.mcp_tools_used)}",
            f"Successful strategies: {', '.join(entry.successful_strategies)}",
            f"Challenges: {', '.join(entry.challenges_encountered)}",
            f"Tags: {', '.join(entry.tags)}"
        ]
        
        return " | ".join(filter(None, text_components))
    
    async def store_memory(self, entry: MemoryEntry) -> bool:
        """
        Store a new memory entry in the vector database.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Create searchable text
            memory_text = self._create_memory_text(entry)
            
            # Generate embedding
            embedding = self.embedding_model.encode(memory_text).tolist()
            
            # Prepare metadata
            metadata = {
                "case_id": entry.case_id,
                "topic": entry.topic,
                "timestamp": entry.timestamp,
                "investigation_duration": entry.investigation_duration,
                "total_sources": entry.total_sources,
                "verification_status": entry.verification_status,
                "tags": json.dumps(entry.tags),
                "mcp_tools": json.dumps(entry.mcp_tools_used)
            }
            
            # Store in ChromaDB
            self.collection.add(
                ids=[entry.id],
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[metadata]
            )
            
            # Update metrics
            self.metrics["total_memories"] += 1
            
            logger.info(f"Stored memory entry: {entry.id} for case: {entry.case_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory entry {entry.id}: {e}")
            return False
    
    async def search_similar_memories(
        self,
        query: str,
        case_context: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[MemorySearchResult]:
        """
        Search for memories similar to the given query.
        
        Args:
            query: Search query for semantic similarity
            case_context: Additional context for filtering results
            limit: Maximum number of results to return
            
        Returns:
            List of similar memory entries with similarity scores
        """
        try:
            # Use provided limit or default
            search_limit = limit or self.max_results
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(search_limit, self.collection.count()) if self.collection.count() > 0 else 1,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            search_results = []
            
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1.0 - distance
                    
                    # Filter by similarity threshold
                    if similarity_score < self.similarity_threshold:
                        continue
                    
                    # Reconstruct memory entry from metadata
                    try:
                        memory_entry = MemoryEntry(
                            id=doc_id,
                            case_id=metadata.get("case_id", ""),
                            topic=metadata.get("topic", ""),
                            timestamp=metadata.get("timestamp", time.time()),
                            investigation_duration=metadata.get("investigation_duration", 0.0),
                            total_sources=metadata.get("total_sources", 0),
                            verification_status=metadata.get("verification_status", "pending"),
                            tags=json.loads(metadata.get("tags", "[]")),
                            mcp_tools_used=json.loads(metadata.get("mcp_tools", "[]"))
                        )
                        
                        # Generate relevance explanation
                        relevance_explanation = self._generate_relevance_explanation(
                            query, memory_entry, similarity_score
                        )
                        
                        search_results.append(MemorySearchResult(
                            entry=memory_entry,
                            similarity_score=similarity_score,
                            relevance_explanation=relevance_explanation
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Failed to reconstruct memory entry {doc_id}: {e}")
                        continue
            
            # Sort by similarity score
            search_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Update metrics
            if search_results:
                self.metrics["successful_retrievals"] += 1
                self.metrics["average_similarity"] = np.mean([r.similarity_score for r in search_results])
            else:
                self.metrics["failed_retrievals"] += 1
            
            logger.info(f"Found {len(search_results)} similar memories for query: {query[:100]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search similar memories: {e}")
            self.metrics["failed_retrievals"] += 1
            return []
    
    def _generate_relevance_explanation(
        self,
        query: str,
        memory: MemoryEntry,
        similarity_score: float
    ) -> str:
        """
        Generate explanation for why a memory is relevant to the query.
        
        Args:
            query: Original search query
            memory: Retrieved memory entry
            similarity_score: Similarity score
            
        Returns:
            Human-readable explanation of relevance
        """
        explanations = []
        
        # Topic similarity
        if memory.topic.lower() in query.lower() or query.lower() in memory.topic.lower():
            explanations.append(f"similar topic ({memory.topic})")
        
        # Tool overlap
        query_words = set(query.lower().split())
        tool_words = set(" ".join(memory.mcp_tools_used).lower().split())
        if query_words.intersection(tool_words):
            explanations.append("similar investigation tools")
        
        # Strategy patterns
        if memory.successful_strategies:
            explanations.append("successful investigation strategies")
        
        # High similarity
        if similarity_score > 0.9:
            explanations.append("very high semantic similarity")
        elif similarity_score > 0.8:
            explanations.append("high semantic similarity")
        
        if not explanations:
            explanations.append(f"semantic similarity ({similarity_score:.2f})")
        
        return f"Relevant due to: {', '.join(explanations)}"
    
    async def get_investigation_patterns(self, topic_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze stored memories to identify investigation patterns.
        
        Args:
            topic_filter: Optional filter for specific topic patterns
            
        Returns:
            Analysis of investigation patterns and insights
        """
        try:
            # Retrieve all memories or filtered subset
            if topic_filter:
                memories = await self.search_similar_memories(topic_filter, limit=100)
                memory_entries = [result.entry for result in memories]
            else:
                # Get all memories from ChromaDB
                all_results = self.collection.get(include=["metadatas"])
                memory_entries = []
                
                for metadata in all_results["metadatas"]:
                    try:
                        memory_entries.append(MemoryEntry(
                            case_id=metadata.get("case_id", ""),
                            topic=metadata.get("topic", ""),
                            timestamp=metadata.get("timestamp", time.time()),
                            tags=json.loads(metadata.get("tags", "[]")),
                            mcp_tools_used=json.loads(metadata.get("mcp_tools", "[]"))
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to parse memory metadata: {e}")
                        continue
            
            # Analyze patterns
            patterns = {
                "total_investigations": len(memory_entries),
                "common_topics": self._analyze_common_topics(memory_entries),
                "popular_tools": self._analyze_tool_usage(memory_entries),
                "success_strategies": self._analyze_success_patterns(memory_entries),
                "investigation_trends": self._analyze_temporal_trends(memory_entries),
                "tag_distribution": self._analyze_tag_distribution(memory_entries)
            }
            
            logger.info(f"Generated investigation patterns for {len(memory_entries)} memories")
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze investigation patterns: {e}")
            return {}
    
    def _analyze_common_topics(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Analyze most common investigation topics."""
        topic_counts = {}
        for memory in memories:
            topic = memory.topic.lower()
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return [
            {"topic": topic, "count": count, "percentage": count / len(memories) * 100}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_tool_usage(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Analyze most frequently used MCP tools."""
        tool_counts = {}
        for memory in memories:
            for tool in memory.mcp_tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        total_usage = sum(tool_counts.values())
        
        return [
            {"tool": tool, "count": count, "percentage": count / total_usage * 100 if total_usage > 0 else 0}
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_success_patterns(self, memories: List[MemoryEntry]) -> List[str]:
        """Analyze common success strategies."""
        strategies = []
        for memory in memories:
            strategies.extend(memory.successful_strategies)
        
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return [
            strategy for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _analyze_temporal_trends(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze investigation trends over time."""
        if not memories:
            return {}
        
        timestamps = [memory.timestamp for memory in memories]
        durations = [memory.investigation_duration for memory in memories if memory.investigation_duration > 0]
        
        return {
            "earliest_investigation": min(timestamps),
            "latest_investigation": max(timestamps),
            "average_duration": np.mean(durations) if durations else 0,
            "investigations_per_day": len(memories) / max(1, (max(timestamps) - min(timestamps)) / 86400)
        }
    
    def _analyze_tag_distribution(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Analyze distribution of investigation tags."""
        tag_counts = {}
        for memory in memories:
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        total_tags = sum(tag_counts.values())
        
        return [
            {"tag": tag, "count": count, "percentage": count / total_tags * 100 if total_tags > 0 else 0}
            for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            collection_stats = {
                "total_memories": self.collection.count(),
                "collection_name": self.collection_name,
                "database_path": str(self.db_path)
            }
            
            return {
                **collection_stats,
                **self.metrics,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                "similarity_threshold": self.similarity_threshold,
                "max_results": self.max_results
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return self.metrics
    
    async def clear_memories(self, confirm: bool = False) -> bool:
        """
        Clear all stored memories (use with caution).
        
        Args:
            confirm: Must be True to proceed with deletion
            
        Returns:
            True if memories were cleared, False otherwise
        """
        if not confirm:
            logger.warning("Memory clearing requires explicit confirmation")
            return False
        
        try:
            # Reset the collection
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(self.collection_name)
            
            # Reset metrics
            self.metrics = {
                "total_memories": 0,
                "successful_retrievals": 0,
                "failed_retrievals": 0,
                "average_similarity": 0.0
            }
            
            logger.info("All memories cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False


# Factory function for easy initialization
def create_vector_memory_manager(
    db_path: str = "./data/memory_db",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> VectorMemoryManager:
    """
    Factory function to create a vector memory manager with optimal settings.
    
    Args:
        db_path: Path to the memory database
        embedding_model: Sentence transformer model to use
        
    Returns:
        Configured VectorMemoryManager instance
    """
    return VectorMemoryManager(
        db_path=db_path,
        collection_name="osint_investigations",
        embedding_model=embedding_model,
        similarity_threshold=0.7,
        max_results=10
    )
