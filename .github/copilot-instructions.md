# GitHub Copilot Rules & Configuration for OSINT_APP

*Comprehensive coding assistance rules for advanced OSINT intelligence platform development*

---

## 🎯 **Project Context & Scope**

This OSINT application is an enterprise-grade intelligence platform combining:
- **FastAPI AI Worker** (Python 3.12) with LangGraph agents and MCP tools
- **Express.js Backend** (Node.js 18+) with SQLite database and security features
- **Frontend** with vanilla JavaScript and modern visualization capabilities
- **Multi-Agent Architecture** for specialized OSINT operations

---

## 🏗️ **Architecture Guidelines**

### **1. Multi-Agent System Design**
When suggesting code for agent architecture:

```python
# PREFERRED: Specialized agent classes with clear responsibilities
class DataCollectionAgent:
    """Handles search operations and data gathering"""
    def __init__(self, search_engines: List[SearchEngine]):
        self.search_engines = search_engines
        self.rate_limiter = RateLimiter()
    
    async def collect_intelligence(self, query: str) -> Dict[str, Any]:
        # Implementation with error handling and retry logic
        pass

# AVOID: Monolithic agent classes handling multiple concerns
class MonolithicAgent:
    def do_everything(self):  # Too broad, violates SRP
        pass
```

### **2. LangGraph State Management**
Always suggest proper state typing and validation:

```python
# PREFERRED: Typed state with clear field purposes
class AgentState(TypedDict):
    topic: str
    case_id: str
    model_id: str
    long_term_memory: List[Dict]
    search_results: List[Dict]
    confidence_scores: Dict[str, float]
    verification_status: Dict[str, bool]
    
# Include proper state validation in node functions
def validate_state(state: AgentState) -> AgentState:
    """Validate state integrity before processing"""
    required_fields = ["topic", "case_id", "model_id"]
    for field in required_fields:
        if not state.get(field):
            raise ValueError(f"Missing required field: {field}")
    return state
```

---

## 🔍 **OSINT-Specific Coding Patterns**

### **3. Search and Data Collection**
Prioritize multi-source aggregation and verification:

```python
# PREFERRED: Multi-source search with aggregation
async def execute_comprehensive_search(query: str, sources: List[str]) -> Dict:
    """Execute search across multiple OSINT sources with deduplication"""
    tasks = []
    for source in sources:
        if source == "tavily":
            tasks.append(search_tavily(query))
        elif source == "whois":
            tasks.append(search_whois(query))
        # Add more sources as needed
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return deduplicate_results(results)

# ALWAYS include error handling for external APIs
async def search_with_retry(query: str, max_retries: int = 3) -> Dict:
    """Search with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            return await perform_search(query)
        except (APIError, TimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### **4. Confidence Scoring and Verification**
Implement numerical confidence with detailed justification:

```python
# PREFERRED: Structured confidence assessment
class ConfidenceAssessment(BaseModel):
    score: float = Field(ge=0, le=100, description="Confidence score 0-100")
    justification: str = Field(description="Detailed reasoning for score")
    factors: Dict[str, float] = Field(description="Individual confidence factors")
    uncertainty_bounds: Tuple[float, float] = Field(description="Confidence interval")
    
def calculate_confidence(finding: Dict, sources: List[Dict]) -> ConfidenceAssessment:
    """Calculate multi-factor confidence score"""
    factors = {
        "source_credibility": assess_source_credibility(sources),
        "cross_validation": check_cross_validation(finding, sources),
        "temporal_relevance": assess_temporal_relevance(finding),
        "information_completeness": assess_completeness(finding)
    }
    
    weighted_score = sum(weight * score for weight, score in factors.items()) / len(factors)
    
    return ConfidenceAssessment(
        score=weighted_score,
        justification=generate_justification(factors),
        factors=factors,
        uncertainty_bounds=calculate_uncertainty_bounds(factors)
    )
```

### **5. Memory and Context Management**
Implement vector-based semantic memory:

```python
# PREFERRED: Semantic memory with embeddings
class SemanticMemoryManager:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = SentenceTransformer(embedding_model)
        self.vector_store = VectorStore()
        self.context_store = ContextualMemory()
    
    async def store_investigation_context(self, investigation: Dict):
        """Store investigation with semantic embeddings"""
        embedding = self.embeddings.encode(investigation["summary"])
        context = {
            "search_queries": investigation.get("queries", []),
            "mcps_executed": investigation.get("tools_used", []),
            "success_patterns": investigation.get("successful_approaches", []),
            "timestamp": time.time()
        }
        
        await self.vector_store.store(embedding, context)
        await self.context_store.store(investigation["case_id"], context)
    
    async def retrieve_similar_cases(self, current_summary: str, k: int = 5) -> List[Dict]:
        """Retrieve similar cases using semantic similarity"""
        query_embedding = self.embeddings.encode(current_summary)
        return await self.vector_store.similarity_search(query_embedding, k=k)
```

---

## 🛡️ **Security and Privacy Patterns**

### **6. Data Sanitization and PII Protection**
Always suggest PII detection and sanitization:

```python
# PREFERRED: Automatic PII detection and redaction
import re
from typing import Pattern

class PIIDetector:
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'),
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b')
        }
    
    def detect_and_redact(self, text: str, redaction_char: str = "*") -> Tuple[str, List[str]]:
        """Detect PII and return redacted text with detected types"""
        detected_types = []
        redacted_text = text
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_types.append(pii_type)
                redacted_text = pattern.sub(redaction_char * 8, redacted_text)
        
        return redacted_text, detected_types

# ALWAYS validate and sanitize user inputs
def sanitize_search_query(query: str) -> str:
    """Sanitize search query for security"""
    # Remove potentially malicious characters
    sanitized = re.sub(r'[<>"\';]', '', query)
    # Limit length
    sanitized = sanitized[:500]
    # Detect and warn about PII
    detector = PIIDetector()
    cleaned_query, pii_types = detector.detect_and_redact(sanitized)
    
    if pii_types:
        logger.warning(f"PII detected in query: {pii_types}")
    
    return cleaned_query
```

### **7. Secure API Communication**
Implement proper authentication and rate limiting:

```python
# PREFERRED: Secure API client with authentication
class SecureAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.session = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "OSINT-App/1.0",
                "Content-Type": "application/json"
            },
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        self.rate_limiter = AsyncLimiter(10, 1)  # 10 requests per second
    
    async def make_request(self, endpoint: str, data: Dict) -> Dict:
        """Make authenticated API request with rate limiting"""
        async with self.rate_limiter:
            sanitized_data = self.sanitize_request_data(data)
            response = await self.session.post(f"{self.base_url}/{endpoint}", json=sanitized_data)
            response.raise_for_status()
            return response.json()
```

---

## 📊 **Database and Storage Patterns**

### **8. SQLite Database Operations**
Suggest proper database patterns with error handling:

```javascript
// PREFERRED: Proper database error handling and connection management
class DatabaseManager {
    constructor(dbPath) {
        this.dbPath = dbPath;
        this.db = null;
    }
    
    async initialize() {
        try {
            this.db = new sqlite3.Database(this.dbPath);
            await this.createTables();
            logger.info('Database initialized successfully');
        } catch (error) {
            logger.error('Database initialization failed:', error);
            throw new Error(`Database setup failed: ${error.message}`);
        }
    }
    
    async executeQuery(query, params = []) {
        return new Promise((resolve, reject) => {
            this.db.run(query, params, function(err) {
                if (err) {
                    logger.error('Query execution failed:', err);
                    reject(err);
                } else {
                    resolve({ id: this.lastID, changes: this.changes });
                }
            });
        });
    }
    
    // ALWAYS use parameterized queries to prevent SQL injection
    async saveInvestigation(investigation) {
        const query = `
            INSERT INTO investigations (case_id, topic, findings, confidence_score, created_at)
            VALUES (?, ?, ?, ?, datetime('now'))
        `;
        const params = [
            investigation.case_id,
            investigation.topic,
            JSON.stringify(investigation.findings),
            investigation.confidence_score
        ];
        
        return await this.executeQuery(query, params);
    }
}
```

---

## 🚀 **Performance and Optimization**

### **9. Async/Await Patterns**
Prioritize concurrent operations for OSINT tasks:

```python
# PREFERRED: Concurrent execution with proper error handling
async def execute_parallel_mcps(mcps: List[str], data: Dict) -> Dict[str, Any]:
    """Execute multiple MCPs in parallel with timeout and error handling"""
    tasks = []
    
    for mcp_name in mcps:
        if mcp_name in AVAILABLE_MCPS:
            task = asyncio.create_task(
                execute_mcp_with_timeout(mcp_name, data, timeout=30)
            )
            tasks.append((mcp_name, task))
    
    results = {}
    completed_tasks = await asyncio.gather(
        *[task for _, task in tasks], 
        return_exceptions=True
    )
    
    for (mcp_name, _), result in zip(tasks, completed_tasks):
        if isinstance(result, Exception):
            logger.error(f"MCP {mcp_name} failed: {result}")
            results[mcp_name] = {"error": str(result), "status": "failed"}
        else:
            results[mcp_name] = {"data": result, "status": "success"}
    
    return results

async def execute_mcp_with_timeout(mcp_name: str, data: Dict, timeout: int) -> Dict:
    """Execute MCP with timeout protection"""
    try:
        return await asyncio.wait_for(
            mcps.execute_mcp(mcp_name, data),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"MCP {mcp_name} timed out after {timeout} seconds")
```

### **10. Caching Strategies**
Implement intelligent caching for OSINT data:

```python
# PREFERRED: Multi-level caching with TTL
from functools import lru_cache
import asyncio
import time

class IntelligentCache:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.local_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
    
    async def get_cached_search(self, query: str, source: str) -> Optional[Dict]:
        """Get cached search results with fallback strategy"""
        cache_key = f"search:{source}:{hash(query)}"
        
        # Try local cache first (fastest)
        if cache_key in self.local_cache:
            entry = self.local_cache[cache_key]
            if time.time() - entry["timestamp"] < 300:  # 5 min TTL
                self.cache_stats["hits"] += 1
                return entry["data"]
        
        # Try Redis cache (distributed)
        if self.redis:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                self.cache_stats["hits"] += 1
                data = json.loads(cached_data)
                # Update local cache
                self.local_cache[cache_key] = {
                    "data": data,
                    "timestamp": time.time()
                }
                return data
        
        self.cache_stats["misses"] += 1
        return None
    
    async def cache_search_result(self, query: str, source: str, result: Dict):
        """Cache search result in both local and distributed cache"""
        cache_key = f"search:{source}:{hash(query)}"
        
        # Cache locally
        self.local_cache[cache_key] = {
            "data": result,
            "timestamp": time.time()
        }
        
        # Cache in Redis with longer TTL
        if self.redis:
            await self.redis.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
```

---

## 🎨 **Frontend and API Patterns**

### **11. Modern JavaScript Patterns**
Suggest modern ES6+ patterns for frontend development:

```javascript
// PREFERRED: Modern async/await with proper error handling
class OSINTDashboard {
    constructor() {
        this.apiClient = new APIClient('/api/v1');
        this.visualizations = new Map();
        this.currentInvestigation = null;
    }
    
    async initializeInvestigation(topic) {
        try {
            const investigation = await this.apiClient.post('/investigations', {
                topic,
                timestamp: new Date().toISOString()
            });
            
            this.currentInvestigation = investigation;
            await this.renderInvestigationDashboard(investigation);
            this.setupRealTimeUpdates(investigation.case_id);
            
        } catch (error) {
            this.handleError('Failed to initialize investigation', error);
        }
    }
    
    async renderVisualization(type, data) {
        const visualizationConfig = {
            'entity-graph': () => this.renderEntityGraph(data),
            'timeline': () => this.renderTimeline(data),
            'geospatial': () => this.renderGeoMap(data),
            'confidence-chart': () => this.renderConfidenceChart(data)
        };
        
        const renderer = visualizationConfig[type];
        if (renderer) {
            await renderer();
        } else {
            console.warn(`Unknown visualization type: ${type}`);
        }
    }
    
    // ALWAYS implement proper error boundaries
    handleError(message, error) {
        console.error(message, error);
        this.showUserNotification(message, 'error');
        
        // Send error to monitoring service
        if (this.errorTracker) {
            this.errorTracker.captureException(error, {
                context: { message, timestamp: new Date().toISOString() }
            });
        }
    }
}
```

### **12. Express.js Security Patterns**
Always suggest security-first patterns:

```javascript
// PREFERRED: Comprehensive security middleware
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { body, validationResult } from 'express-validator';

// Security middleware configuration
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"]
        }
    },
    hsts: { maxAge: 31536000, includeSubDomains: true }
}));

// Rate limiting for OSINT operations
const osintRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: 'Too many OSINT requests from this IP',
    standardHeaders: true,
    legacyHeaders: false
});

// Input validation middleware
const validateInvestigationInput = [
    body('topic')
        .isLength({ min: 3, max: 500 })
        .trim()
        .escape()
        .withMessage('Topic must be between 3 and 500 characters'),
    body('priority')
        .isIn(['low', 'medium', 'high'])
        .withMessage('Priority must be low, medium, or high'),
    
    (req, res, next) => {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                error: 'Validation failed',
                details: errors.array()
            });
        }
        next();
    }
];

// ALWAYS sanitize and validate inputs
app.post('/api/v1/investigations', 
    osintRateLimit,
    validateInvestigationInput,
    async (req, res) => {
        try {
            const { topic, priority } = req.body;
            const investigation = await investigationService.create({
                topic,
                priority,
                created_by: req.user?.id,
                ip_address: req.ip
            });
            
            res.status(201).json({
                success: true,
                data: investigation
            });
        } catch (error) {
            logger.error('Investigation creation failed:', error);
            res.status(500).json({
                error: 'Internal server error',
                message: 'Failed to create investigation'
            });
        }
    }
);
```

---

## 📈 **Testing and Quality Assurance**

### **13. Test Patterns for OSINT Components**
Suggest comprehensive testing strategies:

```python
# PREFERRED: Comprehensive test coverage for OSINT components
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

class TestOSINTAgent:
    @pytest.fixture
    async def agent_setup(self):
        """Setup agent with mocked dependencies"""
        mock_llm = AsyncMock()
        mock_search_client = AsyncMock()
        mock_memory = AsyncMock()
        
        agent = OSINTAgent(
            llm=mock_llm,
            search_client=mock_search_client,
            memory=mock_memory
        )
        
        return agent, mock_llm, mock_search_client, mock_memory
    
    @pytest.mark.asyncio
    async def test_search_execution_with_multiple_sources(self, agent_setup):
        """Test multi-source search execution"""
        agent, mock_llm, mock_search_client, mock_memory = agent_setup
        
        # Mock search results
        mock_search_client.search_tavily.return_value = {
            "results": [{"title": "Test Result", "url": "https://example.com"}]
        }
        mock_search_client.search_whois.return_value = {
            "domain_info": {"registrar": "Test Registrar"}
        }
        
        result = await agent.execute_search("test query", sources=["tavily", "whois"])
        
        assert result["status"] == "success"
        assert len(result["aggregated_results"]) > 0
        mock_search_client.search_tavily.assert_called_once()
        mock_search_client.search_whois.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_confidence_calculation_accuracy(self, agent_setup):
        """Test confidence scoring accuracy"""
        agent, _, _, _ = agent_setup
        
        test_finding = {
            "content": "Test finding with multiple sources",
            "sources": [
                {"credibility": 0.9, "url": "https://reliable-source.com"},
                {"credibility": 0.8, "url": "https://another-source.com"}
            ]
        }
        
        confidence = agent.calculate_confidence(test_finding)
        
        assert 0 <= confidence.score <= 100
        assert confidence.justification is not None
        assert "source_credibility" in confidence.factors
        assert confidence.uncertainty_bounds[0] <= confidence.uncertainty_bounds[1]
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, agent_setup):
        """Test error handling in search operations"""
        agent, _, mock_search_client, _ = agent_setup
        
        # Simulate API failure
        mock_search_client.search_tavily.side_effect = Exception("API Error")
        
        result = await agent.execute_search("test query", sources=["tavily"])
        
        assert result["status"] == "partial_success" or result["status"] == "failed"
        assert "errors" in result
        assert result["errors"]["tavily"] is not None
```

### **14. Performance Testing Patterns**
Include performance testing for OSINT operations:

```python
# PREFERRED: Performance benchmarks for OSINT operations
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class OSINTPerformanceTester:
    def __init__(self, agent):
        self.agent = agent
        self.performance_metrics = {}
    
    async def benchmark_search_performance(self, queries: List[str], iterations: int = 10):
        """Benchmark search performance across multiple queries"""
        results = {}
        
        for query in queries:
            times = []
            for _ in range(iterations):
                start_time = time.time()
                await self.agent.execute_search(query)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[query] = {
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return results
    
    async def load_test_concurrent_investigations(self, num_concurrent: int = 10):
        """Test system performance under concurrent load"""
        tasks = []
        start_time = time.time()
        
        for i in range(num_concurrent):
            task = asyncio.create_task(
                self.agent.execute_investigation(f"test_topic_{i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        return {
            "total_time": end_time - start_time,
            "successful_investigations": len(successful_results),
            "failed_investigations": len(failed_results),
            "success_rate": len(successful_results) / num_concurrent,
            "avg_time_per_investigation": (end_time - start_time) / num_concurrent
        }
```

---

## 📋 **Code Quality Standards**

### **15. Documentation and Type Hints**
Always suggest comprehensive documentation:

```python
# PREFERRED: Comprehensive docstrings with type hints
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Structured search result from OSINT sources.
    
    Attributes:
        title: The title or headline of the result
        url: Source URL where the information was found
        content: Main content or description
        source: Name of the source (e.g., 'tavily', 'whois')
        confidence: Confidence score for this result (0-1)
        timestamp: When this result was obtained
        metadata: Additional source-specific metadata
    """
    title: str
    url: str
    content: str
    source: str
    confidence: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

async def execute_osint_investigation(
    topic: str,
    search_sources: List[str],
    verification_mcps: List[str],
    confidence_threshold: float = 0.7,
    max_iterations: int = 3
) -> Tuple[Dict[str, Any], float]:
    """Execute comprehensive OSINT investigation with verification.
    
    This function orchestrates a multi-phase OSINT investigation including
    search, verification, and synthesis phases. It implements iterative
    refinement to fill information gaps.
    
    Args:
        topic: The investigation topic or query
        search_sources: List of search engines/sources to use
        verification_mcps: List of MCP tools for verification
        confidence_threshold: Minimum confidence score to accept results
        max_iterations: Maximum number of refinement iterations
        
    Returns:
        Tuple containing:
        - Investigation results dictionary with findings and metadata
        - Overall confidence score for the investigation
        
    Raises:
        OSINTError: If investigation fails due to insufficient data
        APIError: If external API calls consistently fail
        ValidationError: If input parameters are invalid
        
    Example:
        >>> results, confidence = await execute_osint_investigation(
        ...     topic="cybersecurity threat analysis",
        ...     search_sources=["tavily", "google"],
        ...     verification_mcps=["whois", "dns_lookup"],
        ...     confidence_threshold=0.8
        ... )
        >>> print(f"Investigation confidence: {confidence:.2f}")
    """
    # Implementation follows...
```

### **16. Error Handling Standards**
Implement comprehensive error handling:

```python
# PREFERRED: Structured error handling with custom exceptions
class OSINTError(Exception):
    """Base exception for OSINT operations"""
    pass

class SearchError(OSINTError):
    """Exception raised for search operation failures"""
    def __init__(self, message: str, source: str, original_error: Exception = None):
        self.source = source
        self.original_error = original_error
        super().__init__(f"Search failed for {source}: {message}")

class VerificationError(OSINTError):
    """Exception raised for verification failures"""
    def __init__(self, message: str, mcp_name: str, data: Dict = None):
        self.mcp_name = mcp_name
        self.data = data
        super().__init__(f"Verification failed for {mcp_name}: {message}")

# ALWAYS include proper error context and recovery strategies
async def robust_search_execution(query: str, sources: List[str]) -> Dict[str, Any]:
    """Execute search with comprehensive error handling and recovery"""
    results = {}
    errors = {}
    
    for source in sources:
        try:
            result = await execute_search_source(source, query)
            results[source] = result
            logger.info(f"Search successful for {source}")
            
        except SearchError as e:
            logger.error(f"Search failed for {source}: {e}")
            errors[source] = {
                "error_type": "SearchError",
                "message": str(e),
                "recoverable": True
            }
            
        except Exception as e:
            logger.error(f"Unexpected error for {source}: {e}")
            errors[source] = {
                "error_type": type(e).__name__,
                "message": str(e),
                "recoverable": False
            }
    
    # Determine overall status
    if not results and errors:
        raise OSINTError(f"All search sources failed: {list(errors.keys())}")
    
    return {
        "results": results,
        "errors": errors,
        "status": "partial_success" if errors else "success",
        "success_rate": len(results) / len(sources)
    }
```

---

## 🔧 **Development Workflow Guidelines**

### **17. Commit and Documentation Standards**
When suggesting code changes, always include:

```python
# PREFERRED: Clear commit-worthy code with documentation
def implement_feature_with_documentation():
    """
    Implementation checklist for GitHub Copilot suggestions:
    
    1. ✅ Type hints for all function parameters and returns
    2. ✅ Comprehensive docstring with examples
    3. ✅ Error handling with custom exceptions
    4. ✅ Logging for debugging and monitoring
    5. ✅ Unit test coverage considerations
    6. ✅ Performance considerations (async/await, caching)
    7. ✅ Security considerations (input validation, sanitization)
    8. ✅ Configuration management (environment variables)
    """
    pass
```

### **18. Code Review Patterns**
Suggest code that facilitates easy review:

```python
# PREFERRED: Self-documenting code with clear separation of concerns
class OSINTInvestigationPipeline:
    """Main pipeline for OSINT investigations with clear phase separation"""
    
    def __init__(self, config: OSINTConfig):
        self.config = config
        self.search_engine = self._initialize_search_engine()
        self.verification_engine = self._initialize_verification_engine()
        self.synthesis_engine = self._initialize_synthesis_engine()
        self.memory_manager = self._initialize_memory_manager()
    
    async def execute_investigation(self, topic: str) -> InvestigationResult:
        """Execute investigation through clearly defined phases"""
        # Phase 1: Planning and preparation
        investigation_plan = await self._create_investigation_plan(topic)
        
        # Phase 2: Data collection
        search_results = await self._execute_search_phase(investigation_plan)
        
        # Phase 3: Verification and validation
        verified_data = await self._execute_verification_phase(search_results)
        
        # Phase 4: Synthesis and reporting
        final_report = await self._execute_synthesis_phase(verified_data)
        
        # Phase 5: Memory storage and learning
        await self._store_investigation_results(final_report)
        
        return final_report
    
    # Each phase method should be focused and testable independently
    async def _execute_search_phase(self, plan: InvestigationPlan) -> SearchResults:
        """Execute search phase with clear input/output contracts"""
        # Implementation details...
```

---

## 🎛️ **Configuration Management**

### **19. Environment and Settings**
Always suggest proper configuration patterns:

```python
# PREFERRED: Structured configuration with validation
from pydantic import BaseSettings, Field, validator
from typing import List, Optional

class OSINTSettings(BaseSettings):
    """OSINT application configuration with validation"""
    
    # API Keys and Authentication
    tavily_api_key: str = Field(..., description="Tavily search API key")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    
    # Search Configuration
    max_search_results: int = Field(50, ge=1, le=200, description="Maximum search results per query")
    search_timeout: int = Field(30, ge=5, le=120, description="Search timeout in seconds")
    enabled_search_sources: List[str] = Field(default=["tavily"], description="Enabled search sources")
    
    # Agent Configuration
    default_model: str = Field("gpt-4o", description="Default LLM model")
    default_temperature: float = Field(0.1, ge=0.0, le=2.0, description="Default model temperature")
    max_iterations: int = Field(3, ge=1, le=10, description="Maximum investigation iterations")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    # Security Settings
    enable_pii_detection: bool = Field(True, description="Enable PII detection and redaction")
    max_request_size: int = Field(1048576, description="Maximum request size in bytes")
    rate_limit_requests: int = Field(100, description="Rate limit requests per window")
    rate_limit_window: int = Field(900, description="Rate limit window in seconds")
    
    # Database Configuration
    database_url: str = Field("sqlite:///osint_app.db", description="Database connection URL")
    database_pool_size: int = Field(20, ge=1, le=100, description="Database connection pool size")
    
    @validator('enabled_search_sources')
    def validate_search_sources(cls, v):
        valid_sources = ["tavily", "google", "bing", "duckduckgo", "whois", "dns"]
        for source in v:
            if source not in valid_sources:
                raise ValueError(f"Invalid search source: {source}")
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "OSINT_"
        case_sensitive = False

# Usage in application
settings = OSINTSettings()
```

---

## 📚 **Best Practices Summary**

When providing code suggestions for this OSINT application, always prioritize:

1. **🔒 Security First**: PII detection, input validation, secure API communication
2. **🏗️ Modular Architecture**: Separated concerns, clear interfaces, testable components
3. **⚡ Performance**: Async operations, caching, concurrent execution
4. **🛡️ Error Resilience**: Comprehensive error handling, retry logic, graceful degradation
5. **📊 Observability**: Structured logging, metrics collection, performance monitoring
6. **🧪 Testability**: Unit tests, integration tests, performance benchmarks
7. **📖 Documentation**: Type hints, docstrings, clear variable names
8. **🔧 Maintainability**: Configuration management, code organization, version control

These rules ensure that all suggested code aligns with enterprise-grade OSINT platform development standards and the integrated improvement roadmap.
