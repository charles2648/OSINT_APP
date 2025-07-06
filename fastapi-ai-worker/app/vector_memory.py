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
    # Create placeholder classes for type hints when imports fail
    if not globals().get('SentenceTransformer'):
        class SentenceTransformer:
            pass
    if not globals().get('Settings'):
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
        similarity_threshold: float = 0.05,  # Lowered for OSINT use
        min_similarity_threshold: float = 0.01,
        max_results: int = 10
    ):
        """
        Initialize the vector memory manager.
        
        Args:
            db_path: Path to ChromaDB storage directory
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            similarity_threshold: Minimum similarity score for relevant results (default 0.05, optimized for OSINT)
            min_similarity_threshold: Absolute minimum allowed for threshold (default 0.01)
            max_results: Maximum number of results to return
        """
        if not VECTOR_DEPS_AVAILABLE:
            raise ImportError(
                "Vector memory dependencies not available. "
                "Please install: pip install chromadb sentence-transformers numpy"
            )
        if similarity_threshold < min_similarity_threshold:
            logger.warning(f"Similarity threshold {similarity_threshold} is very low; using min {min_similarity_threshold}.")
            similarity_threshold = min_similarity_threshold
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.similarity_threshold = similarity_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.max_results = max_results
        
        # Initialize performance metrics first
        self.metrics = {
            "total_memories": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "average_similarity": 0.0
        }
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self._initialize_database()
    
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
        Create a more natural language searchable text representation of a memory entry.
        """
        findings_text = "; ".join([
            f"Finding: {json.dumps(f)}" for f in entry.findings
        ])
        strategies_text = ", ".join(entry.successful_strategies)
        challenges_text = ", ".join(entry.challenges_encountered)
        tags_text = ", ".join(entry.tags)
        tools_text = ", ".join(entry.mcp_tools_used)
        queries_text = ", ".join(entry.search_queries)
        
        # Safely handle entity relationships
        entity_text = ""
        if hasattr(entry, 'entity_relationships') and entry.entity_relationships:
            entity_pairs = []
            for k, v in entry.entity_relationships.items():
                if isinstance(v, list):
                    entity_pairs.append(f"{k}: {', '.join(v)}")
                else:
                    entity_pairs.append(f"{k}: {v}")
            entity_text = ", ".join(entity_pairs)
        
        return (
            f"Investigation on '{entry.topic}'. "
            f"Search queries: {queries_text}. "
            f"Key findings: {findings_text}. "
            f"Tools used: {tools_text}. "
            f"Successful strategies: {strategies_text}. "
            f"Challenges: {challenges_text}. "
            f"Tags: {tags_text}. "
            f"Entities: {entity_text}."
        )
    
    async def store_memory(self, entry: MemoryEntry) -> bool:
        """
        Store a new memory entry in the vector database.
        
        Args:
            entry: Memory entry to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        # Validate entry
        if not entry.case_id or not entry.topic:
            logger.warning(f"Refusing to store memory with empty case_id or topic: {entry}")
            return False
        
        try:
            # Check for existing entry to prevent duplicates
            existing = self.collection.get(ids=[entry.id])
            if existing["ids"]:
                logger.warning(f"Memory entry {entry.id} already exists, skipping storage")
                return False
            
            # Create searchable text
            memory_text = self._create_memory_text(entry)
            
            # Generate embedding
            embedding = self.embedding_model.encode(memory_text).tolist()
            
            # Prepare metadata - store complete entry as JSON for accurate reconstruction
            metadata: Dict[str, Any] = {
                "case_id": entry.case_id,
                "topic": entry.topic,
                "timestamp": entry.timestamp,
                "investigation_duration": entry.investigation_duration,
                "total_sources": entry.total_sources,
                "verification_status": entry.verification_status,
                "tags": json.dumps(entry.tags),
                "mcp_tools": json.dumps(entry.mcp_tools_used),
                # Store complete entry data for accurate reconstruction
                "full_entry": entry.model_dump_json()
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
            search_limit = limit or self.max_results
            
            # Check if collection is empty
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.info("No memories stored yet, returning empty results")
                return []
            
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(search_limit, collection_count),
                include=["documents", "metadatas", "distances"]
            )
            search_results = []
            
            if results["ids"] and results["ids"][0]:  # Check if we have results
                documents_list = results.get("documents")
                metadatas_list = results.get("metadatas")  
                distances_list = results.get("distances")
                ids_list = results.get("ids")
                
                if documents_list and metadatas_list and distances_list and ids_list:
                    documents = documents_list[0] or []
                    metadatas = metadatas_list[0] or []
                    distances = distances_list[0] or []
                    ids = ids_list[0] or []
                    
                    for i, (doc_id, document, metadata, distance) in enumerate(zip(
                        ids, documents, metadatas, distances
                    )):
                        # Convert L2 distance to cosine similarity (for normalized embeddings)
                        # ChromaDB uses L2 distance: L2_distance = 2 * (1 - cosine_similarity)
                        # So: cosine_similarity = 1 - L2_distance/2
                        similarity_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                        
                        # Filter by similarity threshold
                        if similarity_score < self.similarity_threshold:
                            continue
                        
                        # Reconstruct memory entry from stored data
                        try:
                            if "full_entry" in metadata:
                                # Use complete stored entry data for accurate reconstruction
                                full_entry_str = metadata.get("full_entry")
                                if isinstance(full_entry_str, str):
                                    memory_entry = MemoryEntry.model_validate_json(full_entry_str)
                                else:
                                    # Handle unexpected type gracefully
                                    logger.warning(f"Expected string for full_entry, got {type(full_entry_str)}")
                                    continue
                            else:
                                # Fallback to partial reconstruction for older entries
                                memory_entry = MemoryEntry(
                                    id=doc_id,
                                    case_id=str(metadata.get("case_id", "")),
                                    topic=str(metadata.get("topic", "")),
                                    timestamp=float(metadata.get("timestamp") or time.time()),
                                    investigation_duration=float(metadata.get("investigation_duration") or 0.0),
                                    total_sources=int(metadata.get("total_sources") or 0),
                                    verification_status=str(metadata.get("verification_status", "pending")),
                                    tags=json.loads(str(metadata.get("tags", "[]"))),
                                    mcp_tools_used=json.loads(str(metadata.get("mcp_tools", "[]")))
                                )
                            
                            # Apply case context filtering if provided
                            if case_context:
                                should_include = True
                                for filter_key, filter_value in case_context.items():
                                    if hasattr(memory_entry, filter_key):
                                        entry_value = getattr(memory_entry, filter_key)
                                        if entry_value != filter_value:
                                            should_include = False
                                            break
                                    elif filter_key in metadata:
                                        if metadata[filter_key] != filter_value:
                                            should_include = False
                                            break
                                
                                if not should_include:
                                    continue
                            
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
                scores = [r.similarity_score for r in search_results]
                self.metrics["average_similarity"] = float(np.mean(scores)) if scores else 0.0
            else:
                self.metrics["failed_retrievals"] += 1
            
            logger.info(f"Found {len(search_results)} similar memories for query: {query[:100]}...")
            if not search_results and self.similarity_threshold > self.min_similarity_threshold:
                # Fallback: retry with min threshold (temporarily)
                logger.warning(f"No results, retrying with min_similarity_threshold {self.min_similarity_threshold}")
                original_threshold = self.similarity_threshold
                self.similarity_threshold = self.min_similarity_threshold
                try:
                    fallback_results = await self.search_similar_memories(query, case_context, limit)
                    return fallback_results
                finally:
                    # Restore original threshold
                    self.similarity_threshold = original_threshold
            if not search_results:
                logger.warning(f"No similar memories found for query '{query}'. Even after fallback.")
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
                
                metadatas = all_results.get("metadatas") if all_results else None
                if metadatas:
                    for metadata in metadatas:
                        try:
                            if "full_entry" in metadata:
                                # Use complete stored entry data for accurate reconstruction
                                full_entry_str = metadata.get("full_entry")
                                if isinstance(full_entry_str, str):
                                    memory_entry = MemoryEntry.model_validate_json(full_entry_str)
                                else:
                                    continue
                            else:
                                # Fallback to partial reconstruction for older entries
                                memory_entry = MemoryEntry(
                                    case_id=str(metadata.get("case_id", "")),
                                    topic=str(metadata.get("topic", "")),
                                    timestamp=float(metadata.get("timestamp") or time.time()),
                                    tags=json.loads(str(metadata.get("tags", "[]"))),
                                    mcp_tools_used=json.loads(str(metadata.get("mcp_tools", "[]")))
                                )
                            memory_entries.append(memory_entry)
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
        if not memories:
            return []
            
        topic_counts: Dict[str, int] = {}
        for memory in memories:
            topic = memory.topic.lower()
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return [
            {"topic": topic, "count": count, "percentage": count / len(memories) * 100}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_tool_usage(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Analyze most frequently used MCP tools."""
        tool_counts: Dict[str, int] = {}
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
        
        strategy_counts: Dict[str, int] = {}
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
            "average_duration": float(np.mean(durations)) if durations else 0.0,
            "investigations_per_day": len(memories) / max(1, (max(timestamps) - min(timestamps)) / 86400)
        }
    
    def _analyze_tag_distribution(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Analyze distribution of investigation tags."""
        tag_counts: Dict[str, int] = {}
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
            
            # Get embedding dimension safely
            try:
                embedding_dim: Optional[int] = self.embedding_model.get_sentence_embedding_dimension()
            except AttributeError:
                # Fallback for older versions
                try:
                    embedding_dim = len(self.embedding_model.encode("test"))
                except Exception:
                    embedding_dim = None
            
            return {
                **collection_stats,
                **self.metrics,
                "embedding_model_dimension": embedding_dim,
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
