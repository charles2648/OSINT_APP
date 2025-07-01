# Gemini CLI Configuration for OSINT_APP

*Comprehensive AI assistant configuration for advanced OSINT intelligence platform development*

---

## 🎯 **Project Context & Specialized Domain**

You are an AI assistant specialized in OSINT (Open Source Intelligence) platform development. This project is an enterprise-grade intelligence system with the following architecture:

### **System Architecture**
- **FastAPI AI Worker** (Python 3.12): LangGraph multi-agent system with MCP tool integration
- **Express.js Backend** (Node.js 18+): RESTful API with SQLite database and security middleware
- **Modern Frontend**: Vanilla JavaScript with interactive visualizations
- **Multi-Agent Framework**: Specialized agents for data collection, verification, synthesis, and quality assessment

### **Core Technologies Stack**
```yaml
Backend:
  - FastAPI 0.115.0+ with Pydantic models
  - LangGraph 0.2.50+ for agent orchestration
  - LangChain ecosystem for LLM integration
  - Tavily for OSINT search operations
  - SQLite with proper parameterized queries

Frontend:
  - Vanilla JavaScript (ES6+)
  - Modern CSS with responsive design
  - D3.js for data visualization
  - WebSocket for real-time updates

AI/ML:
  - OpenAI GPT models (gpt-4o, gpt-4o-mini)
  - Anthropic Claude models (claude-3-5-sonnet)
  - Dynamic model selection based on task complexity
  - Vector embeddings for semantic memory
```

---

## 🧠 **OSINT-Specific Knowledge Requirements**

### **1. OSINT Methodology Understanding**
Always consider these OSINT principles when providing assistance:

#### **Intelligence Cycle Integration**
- **Planning & Direction**: Strategic research planning, requirement definition
- **Collection**: Multi-source data gathering with source diversity
- **Processing**: Data normalization, deduplication, and structuring
- **Analysis**: Pattern recognition, entity extraction, relationship mapping
- **Dissemination**: Professional intelligence products and actionable insights

#### **Source Evaluation Framework**
```python
# Apply this framework when suggesting source assessment code
class SourceCredibilityAssessment:
    """OSINT source evaluation criteria"""
    
    def assess_source_reliability(self, source_data: Dict) -> float:
        """
        Evaluate source using OSINT reliability standards:
        - Accuracy: Historical accuracy of information
        - Timeliness: Recency and update frequency
        - Relevance: Direct relevance to investigation topic
        - Bias: Political, commercial, or ideological bias
        - Verifiability: Ability to cross-reference information
        """
        factors = {
            'accuracy_score': self.calculate_historical_accuracy(source_data),
            'timeliness_score': self.assess_information_freshness(source_data),
            'relevance_score': self.measure_topic_relevance(source_data),
            'bias_score': self.detect_bias_indicators(source_data),
            'verifiability_score': self.assess_cross_reference_potential(source_data)
        }
        
        return self.weighted_reliability_score(factors)
```

### **2. Threat Intelligence Integration**
When suggesting code for threat intelligence features:

#### **IOC (Indicators of Compromise) Patterns**
```python
# Always consider these IOC categories in threat analysis suggestions
IOC_PATTERNS = {
    'network_indicators': {
        'ip_addresses': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        'domains': r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,})\b',
        'urls': r'https?://[^\s<>"\']+',
        'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    },
    'file_indicators': {
        'md5_hashes': r'\b[a-fA-F0-9]{32}\b',
        'sha1_hashes': r'\b[a-fA-F0-9]{40}\b',
        'sha256_hashes': r'\b[a-fA-F0-9]{64}\b',
        'file_paths': r'[A-Za-z]:\\[^<>:"|?*\r\n]*|/[^<>:"|?*\r\n]*'
    },
    'behavioral_indicators': {
        'registry_keys': r'HKEY_[A-Z_]+\\[^<>:"|?*\r\n]*',
        'service_names': r'[A-Za-z][A-Za-z0-9_\-]*',
        'process_names': r'[A-Za-z][A-Za-z0-9_\-]*\.exe'
    }
}
```

#### **MITRE ATT&CK Framework Integration**
```python
# When suggesting threat analysis code, reference MITRE ATT&CK
class MITREAttackMapper:
    """Map observed behaviors to MITRE ATT&CK framework"""
    
    TACTICS = {
        'initial_access': 'TA0001',
        'execution': 'TA0002',
        'persistence': 'TA0003',
        'privilege_escalation': 'TA0004',
        'defense_evasion': 'TA0005',
        'credential_access': 'TA0006',
        'discovery': 'TA0007',
        'lateral_movement': 'TA0008',
        'collection': 'TA0009',
        'command_and_control': 'TA0011',
        'exfiltration': 'TA0010',
        'impact': 'TA0040'
    }
    
    def map_behaviors_to_techniques(self, observed_behaviors: List[str]) -> Dict[str, List[str]]:
        """Map observed behaviors to specific MITRE techniques"""
        # Implementation should reference current MITRE ATT&CK knowledge
        pass
```

---

## 🔍 **Code Generation Guidelines**

### **3. Multi-Agent Architecture Patterns**
When suggesting agent-related code, always consider specialized agent roles:

#### **Agent Specialization Framework**
```python
# PREFERRED: Specialized agent classes with clear responsibilities
from abc import ABC, abstractmethod
from typing import Protocol

class OSINTAgent(ABC):
    """Base class for specialized OSINT agents"""
    
    @abstractmethod
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-specific task"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass

class DataCollectionAgent(OSINTAgent):
    """Specialized agent for multi-source data collection"""
    
    def __init__(self, search_engines: List[SearchEngine], rate_limiter: RateLimiter):
        self.search_engines = search_engines
        self.rate_limiter = rate_limiter
        self.collection_metrics = CollectionMetrics()
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive data collection with source diversity"""
        query = task_data['query']
        sources = task_data.get('sources', ['tavily', 'whois'])
        
        # Parallel execution with rate limiting
        collection_results = await self.execute_parallel_collection(query, sources)
        
        # Apply deduplication and quality assessment
        processed_results = await self.process_collection_results(collection_results)
        
        return {
            'collected_data': processed_results,
            'collection_metrics': self.collection_metrics.get_stats(),
            'source_coverage': self.assess_source_coverage(sources),
            'data_quality_score': self.calculate_quality_score(processed_results)
        }

class VerificationAgent(OSINTAgent):
    """Specialized agent for fact-checking and source validation"""
    
    def __init__(self, verification_mcps: List[str], cross_reference_db: CrossReferenceDB):
        self.verification_mcps = verification_mcps
        self.cross_reference_db = cross_reference_db
        self.verification_history = VerificationHistory()
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive verification using multiple methods"""
        findings = task_data['findings']
        verification_strategy = task_data.get('strategy', 'comprehensive')
        
        # Multi-method verification
        verification_results = await self.execute_verification_methods(findings)
        
        # Cross-reference validation
        cross_ref_results = await self.cross_reference_validation(findings)
        
        # Generate verification confidence score
        confidence_assessment = self.calculate_verification_confidence(
            verification_results, cross_ref_results
        )
        
        return {
            'verification_results': verification_results,
            'cross_reference_results': cross_ref_results,
            'confidence_assessment': confidence_assessment,
            'verification_recommendations': self.generate_recommendations(findings)
        }
```

### **4. Search and Intelligence Fusion**
When suggesting search-related code, implement comprehensive aggregation:

#### **Multi-Source Search Architecture**
```python
# PREFERRED: Comprehensive search with intelligence fusion
class IntelligenceFusionEngine:
    """Advanced multi-source intelligence aggregation with fusion algorithms"""
    
    def __init__(self):
        self.search_sources = {
            'web_search': [TavilySearchEngine(), GoogleSearchEngine(), BingSearchEngine()],
            'social_media': [TwitterAPI(), LinkedInAPI(), RedditAPI()],
            'technical_sources': [ShodanAPI(), VirusTotalAPI(), WhoisAPI()],
            'news_sources': [NewsAPI(), AllSidesAPI(), RSSAggregator()],
            'threat_intel': [OTXAPI(), MISPClient(), ThreatCrowdAPI()]
        }
        self.fusion_algorithms = IntelligenceFusionAlgorithms()
        self.source_credibility = SourceCredibilityManager()
    
    async def execute_comprehensive_search(
        self, 
        query: str, 
        source_categories: List[str] = None,
        fusion_strategy: str = 'weighted_consensus'
    ) -> FusedIntelligenceResult:
        """Execute search across multiple categories with intelligent fusion"""
        
        # Select appropriate sources based on query analysis
        selected_sources = self.select_optimal_sources(query, source_categories)
        
        # Execute parallel searches with error handling
        search_results = await self.execute_parallel_searches(query, selected_sources)
        
        # Apply intelligence fusion algorithms
        fused_results = await self.fusion_algorithms.fuse_intelligence(
            search_results, strategy=fusion_strategy
        )
        
        # Calculate aggregated confidence scores
        confidence_metrics = self.calculate_fusion_confidence(fused_results)
        
        # Identify information gaps and conflicts
        gap_analysis = self.analyze_information_gaps(fused_results)
        conflict_analysis = self.detect_source_conflicts(search_results)
        
        return FusedIntelligenceResult(
            fused_data=fused_results,
            confidence_metrics=confidence_metrics,
            source_attribution=self.generate_source_attribution(search_results),
            gap_analysis=gap_analysis,
            conflict_analysis=conflict_analysis,
            recommendation_priority=self.prioritize_findings(fused_results)
        )
    
    def select_optimal_sources(self, query: str, categories: List[str] = None) -> Dict[str, List]:
        """Select optimal sources based on query characteristics and requirements"""
        query_analysis = self.analyze_query_intent(query)
        
        if query_analysis['type'] == 'person_investigation':
            return {
                'social_media': self.search_sources['social_media'],
                'web_search': self.search_sources['web_search'][:2]  # Limit for efficiency
            }
        elif query_analysis['type'] == 'technical_investigation':
            return {
                'technical_sources': self.search_sources['technical_sources'],
                'threat_intel': self.search_sources['threat_intel']
            }
        elif query_analysis['type'] == 'event_investigation':
            return {
                'news_sources': self.search_sources['news_sources'],
                'social_media': self.search_sources['social_media'],
                'web_search': self.search_sources['web_search']
            }
        else:
            # Default comprehensive search
            return {category: sources for category, sources in self.search_sources.items()}
```

### **5. Advanced Memory and Learning Systems**
When suggesting memory-related code, implement vector-based semantic systems:

#### **Semantic Memory Architecture**
```python
# PREFERRED: Advanced semantic memory with learning capabilities
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

class SemanticIntelligenceMemory:
    """Advanced semantic memory system for OSINT investigations"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_store = VectorStore(dimension=384)  # Model-specific dimension
        self.context_graph = InvestigationContextGraph()
        self.learning_engine = ContinuousLearningEngine()
        self.pattern_recognizer = OSINTPatternRecognizer()
    
    async def store_investigation_context(
        self, 
        investigation: InvestigationResult,
        success_indicators: Dict[str, float]
    ) -> str:
        """Store investigation with comprehensive context and success patterns"""
        
        # Generate semantic embeddings for different aspects
        embeddings = {
            'topic_embedding': self.embedding_model.encode(investigation.topic),
            'findings_embedding': self.embedding_model.encode(investigation.findings_summary),
            'methodology_embedding': self.embedding_model.encode(investigation.methodology_used)
        }
        
        # Create comprehensive context record
        context_record = {
            'investigation_id': investigation.case_id,
            'topic': investigation.topic,
            'methodology': {
                'search_queries': investigation.search_queries,
                'mcps_executed': investigation.mcps_executed,
                'agent_coordination': investigation.agent_coordination_pattern,
                'time_to_completion': investigation.execution_time
            },
            'success_patterns': {
                'confidence_achieved': investigation.final_confidence_score,
                'user_satisfaction': success_indicators.get('user_satisfaction', 0.0),
                'information_completeness': success_indicators.get('completeness', 0.0),
                'source_diversity': success_indicators.get('source_diversity', 0.0)
            },
            'learned_insights': {
                'effective_queries': investigation.most_effective_queries,
                'optimal_mcps': investigation.most_valuable_mcps,
                'time_efficiency': investigation.time_efficiency_metrics,
                'challenge_points': investigation.identified_challenges
            },
            'embeddings': embeddings,
            'timestamp': investigation.completion_timestamp
        }
        
        # Store in vector database for similarity retrieval
        await self.vector_store.store(
            vector=embeddings['topic_embedding'],
            metadata=context_record,
            id=investigation.case_id
        )
        
        # Update context graph for relationship mapping
        await self.context_graph.add_investigation_node(context_record)
        
        # Feed to learning engine for pattern extraction
        await self.learning_engine.process_new_experience(context_record)
        
        return investigation.case_id
    
    async def retrieve_relevant_experiences(
        self, 
        current_topic: str, 
        similarity_threshold: float = 0.7,
        max_results: int = 5
    ) -> List[RelevantExperience]:
        """Retrieve relevant past experiences for current investigation"""
        
        # Generate embedding for current topic
        topic_embedding = self.embedding_model.encode(current_topic)
        
        # Search for similar investigations
        similar_investigations = await self.vector_store.similarity_search(
            query_vector=topic_embedding,
            threshold=similarity_threshold,
            limit=max_results
        )
        
        # Extract actionable insights from similar cases
        relevant_experiences = []
        for similar_case in similar_investigations:
            experience = RelevantExperience(
                case_id=similar_case['metadata']['investigation_id'],
                similarity_score=similar_case['similarity_score'],
                recommended_queries=similar_case['metadata']['learned_insights']['effective_queries'],
                recommended_mcps=similar_case['metadata']['learned_insights']['optimal_mcps'],
                expected_challenges=similar_case['metadata']['learned_insights']['challenge_points'],
                success_probability=self.predict_success_probability(
                    current_topic, similar_case['metadata']
                ),
                adaptation_suggestions=self.generate_adaptation_suggestions(
                    current_topic, similar_case['metadata']
                )
            )
            relevant_experiences.append(experience)
        
        return sorted(relevant_experiences, key=lambda x: x.similarity_score, reverse=True)
    
    async def extract_investigation_patterns(self) -> OSINTPatterns:
        """Extract patterns from investigation history for system improvement"""
        return await self.pattern_recognizer.analyze_historical_patterns(
            self.vector_store.get_all_records()
        )
```

### **6. Quality Assessment and Confidence Scoring**
When suggesting quality assessment code, implement comprehensive evaluation:

#### **Multi-Dimensional Quality Assessment**
```python
# PREFERRED: Comprehensive quality assessment framework
class IntelligenceQualityAssessor:
    """Advanced quality assessment for OSINT intelligence products"""
    
    def __init__(self):
        self.bias_detector = BiasDetectionEngine()
        self.completeness_analyzer = CompletenessAnalyzer()
        self.credibility_assessor = CredibilityAssessor()
        self.temporal_analyzer = TemporalRelevanceAnalyzer()
        self.cross_validator = CrossValidationEngine()
    
    async def assess_intelligence_quality(
        self, 
        investigation_result: InvestigationResult,
        assessment_criteria: QualityAssessmentCriteria
    ) -> IntelligenceQualityReport:
        """Comprehensive quality assessment of intelligence products"""
        
        # Multi-dimensional quality assessment
        quality_dimensions = {
            'accuracy': await self.assess_accuracy(investigation_result),
            'completeness': await self.assess_completeness(investigation_result),
            'credibility': await self.assess_credibility(investigation_result),
            'timeliness': await self.assess_timeliness(investigation_result),
            'relevance': await self.assess_relevance(investigation_result),
            'bias_neutrality': await self.assess_bias_neutrality(investigation_result),
            'verifiability': await self.assess_verifiability(investigation_result)
        }
        
        # Calculate weighted overall quality score
        overall_score = self.calculate_weighted_quality_score(
            quality_dimensions, assessment_criteria.weights
        )
        
        # Generate detailed quality justification
        quality_justification = self.generate_quality_justification(quality_dimensions)
        
        # Identify improvement opportunities
        improvement_recommendations = self.identify_improvement_opportunities(
            quality_dimensions, assessment_criteria.thresholds
        )
        
        # Calculate confidence bounds
        confidence_interval = self.calculate_confidence_interval(
            quality_dimensions, investigation_result.source_diversity
        )
        
        return IntelligenceQualityReport(
            overall_score=overall_score,
            dimension_scores=quality_dimensions,
            confidence_interval=confidence_interval,
            quality_justification=quality_justification,
            improvement_recommendations=improvement_recommendations,
            assessment_metadata={
                'assessment_timestamp': time.time(),
                'assessment_criteria': assessment_criteria,
                'assessor_version': self.get_version_info()
            }
        )
    
    async def assess_accuracy(self, investigation_result: InvestigationResult) -> AccuracyAssessment:
        """Assess accuracy through cross-validation and fact-checking"""
        
        # Cross-reference findings across multiple sources
        cross_ref_results = await self.cross_validator.validate_findings(
            investigation_result.findings
        )
        
        # Check against known fact databases
        fact_check_results = await self.cross_validator.fact_check_against_databases(
            investigation_result.findings
        )
        
        # Analyze internal consistency
        consistency_analysis = self.analyze_internal_consistency(
            investigation_result.findings
        )
        
        # Calculate accuracy score
        accuracy_score = self.calculate_accuracy_score(
            cross_ref_results, fact_check_results, consistency_analysis
        )
        
        return AccuracyAssessment(
            score=accuracy_score,
            cross_reference_validation=cross_ref_results,
            fact_check_results=fact_check_results,
            consistency_analysis=consistency_analysis,
            accuracy_confidence=self.calculate_accuracy_confidence(
                cross_ref_results, fact_check_results
            )
        )
```

---

## 🛡️ **Security and Privacy Integration**

### **7. OSINT Operational Security (OPSEC)**
When suggesting security-related code, always include OPSEC considerations:

#### **OPSEC Framework Implementation**
```python
# PREFERRED: Comprehensive OPSEC framework for OSINT operations
class OSINTOperationalSecurity:
    """OPSEC framework for secure OSINT operations"""
    
    def __init__(self):
        self.anonymization_engine = AnonymizationEngine()
        self.proxy_manager = ProxyRotationManager()
        self.attribution_tracker = AttributionTracker()
        self.exposure_analyzer = ExposureAnalyzer()
    
    async def evaluate_operation_security(
        self, 
        search_query: str, 
        target_sources: List[str],
        sensitivity_level: str
    ) -> OPSECAssessment:
        """Evaluate OPSEC risks before executing search operations"""
        
        # Analyze query for attribution risks
        attribution_risks = await self.analyze_attribution_risks(search_query)
        
        # Assess source exposure risks
        source_risks = await self.assess_source_exposure_risks(target_sources)
        
        # Evaluate data sensitivity
        sensitivity_assessment = await self.assess_data_sensitivity(search_query)
        
        # Calculate overall OPSEC risk score
        risk_score = self.calculate_opsec_risk_score(
            attribution_risks, source_risks, sensitivity_assessment
        )
        
        # Generate security recommendations
        security_recommendations = self.generate_security_recommendations(
            risk_score, sensitivity_level, target_sources
        )
        
        return OPSECAssessment(
            risk_score=risk_score,
            attribution_risks=attribution_risks,
            source_risks=source_risks,
            sensitivity_level=sensitivity_assessment,
            security_recommendations=security_recommendations,
            recommended_anonymization=self.recommend_anonymization_level(risk_score),
            proxy_requirements=self.determine_proxy_requirements(risk_score)
        )
    
    async def execute_secure_search(
        self, 
        search_query: str, 
        opsec_assessment: OPSECAssessment
    ) -> SecureSearchResult:
        """Execute search with appropriate OPSEC measures"""
        
        # Apply query anonymization if required
        if opsec_assessment.recommended_anonymization > 0.5:
            anonymized_query = await self.anonymization_engine.anonymize_query(
                search_query, level=opsec_assessment.recommended_anonymization
            )
        else:
            anonymized_query = search_query
        
        # Configure proxy if required
        proxy_config = None
        if opsec_assessment.proxy_requirements['required']:
            proxy_config = await self.proxy_manager.get_secure_proxy(
                country=opsec_assessment.proxy_requirements.get('country'),
                anonymity_level=opsec_assessment.proxy_requirements.get('anonymity_level')
            )
        
        # Execute search with security measures
        search_result = await self.execute_protected_search(
            query=anonymized_query,
            proxy_config=proxy_config,
            attribution_protection=opsec_assessment.attribution_risks['protection_level']
        )
        
        # Sanitize results for sensitive data
        sanitized_results = await self.sanitize_search_results(
            search_result, opsec_assessment.sensitivity_level
        )
        
        return SecureSearchResult(
            results=sanitized_results,
            security_metadata={
                'opsec_measures_applied': opsec_assessment.security_recommendations,
                'anonymization_level': opsec_assessment.recommended_anonymization,
                'proxy_used': proxy_config is not None,
                'attribution_protection': opsec_assessment.attribution_risks['protection_level']
            }
        )
```

### **8. Privacy-Preserving Analytics**
When suggesting analytics code, implement privacy-first design:

#### **Differential Privacy Implementation**
```python
# PREFERRED: Privacy-preserving analytics for OSINT operations
import numpy as np
from typing import Dict, List, Any

class PrivacyPreservingAnalytics:
    """Privacy-preserving analytics engine for OSINT metrics"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Privacy failure probability
        self.noise_generator = DifferentialPrivacyNoiseGenerator(epsilon, delta)
        self.data_minimizer = DataMinimizationEngine()
    
    async def analyze_investigation_patterns(
        self, 
        investigations: List[InvestigationResult],
        privacy_level: str = 'high'
    ) -> PrivateAnalyticsResult:
        """Analyze investigation patterns with differential privacy"""
        
        # Apply data minimization
        minimized_data = await self.data_minimizer.minimize_dataset(
            investigations, privacy_level
        )
        
        # Calculate private statistics
        private_stats = {
            'success_rate': self.calculate_private_success_rate(minimized_data),
            'average_confidence': self.calculate_private_average_confidence(minimized_data),
            'source_distribution': self.calculate_private_source_distribution(minimized_data),
            'time_patterns': self.calculate_private_time_patterns(minimized_data)
        }
        
        # Add calibrated noise for differential privacy
        noisy_stats = self.add_differential_privacy_noise(private_stats)
        
        # Generate privacy-preserving insights
        insights = self.generate_private_insights(noisy_stats)
        
        return PrivateAnalyticsResult(
            statistics=noisy_stats,
            insights=insights,
            privacy_guarantees={
                'epsilon': self.epsilon,
                'delta': self.delta,
                'privacy_level': privacy_level
            },
            data_retention_policy=self.get_retention_policy(privacy_level)
        )
    
    def calculate_private_success_rate(self, data: List[Dict]) -> float:
        """Calculate success rate with differential privacy"""
        true_success_rate = sum(1 for inv in data if inv['success']) / len(data)
        
        # Add Laplace noise for differential privacy
        noise = self.noise_generator.laplace_noise(sensitivity=1/len(data))
        
        # Clamp result to valid range [0, 1]
        return np.clip(true_success_rate + noise, 0, 1)
```

---

## 📊 **Visualization and Dashboard Guidance**

### **9. Interactive Intelligence Dashboards**
When suggesting frontend code, focus on intelligence visualization:

#### **OSINT Dashboard Framework**
```javascript
// PREFERRED: Modern intelligence dashboard with interactive features
class OSINTIntelligenceDashboard {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.visualizations = new Map();
        this.dataManager = new IntelligenceDataManager();
        this.interactionController = new DashboardInteractionController();
        this.realTimeUpdater = new RealTimeIntelligenceUpdater();
    }
    
    async initializeDashboard(investigationId) {
        try {
            // Load investigation data
            const investigationData = await this.dataManager.loadInvestigation(investigationId);
            
            // Initialize core visualizations
            await this.initializeEntityGraph(investigationData.entities);
            await this.initializeTimelineVisualization(investigationData.timeline);
            await this.initializeGeospatialMap(investigationData.locations);
            await this.initializeConfidenceMetrics(investigationData.confidence);
            await this.initializeSourceNetwork(investigationData.sources);
            
            // Setup real-time updates
            this.realTimeUpdater.subscribe(investigationId, this.handleDataUpdate.bind(this));
            
            // Initialize interaction handlers
            this.setupInteractionHandlers();
            
        } catch (error) {
            this.handleDashboardError('Dashboard initialization failed', error);
        }
    }
    
    async initializeEntityGraph(entityData) {
        """Create interactive entity relationship graph using D3.js"""
        const graphContainer = this.createVisualizationContainer('entity-graph');
        
        const entityGraph = new EntityRelationshipGraph(graphContainer, {
            width: 800,
            height: 600,
            nodeTypes: ['person', 'organization', 'location', 'event', 'document'],
            linkTypes: ['associated_with', 'located_at', 'participated_in', 'contains'],
            interactionMode: 'exploration',
            clustering: {
                enabled: true,
                algorithm: 'community_detection',
                minClusterSize: 3
            },
            filters: {
                confidenceThreshold: 0.7,
                temporalRange: 'last_30_days',
                entityTypes: ['all']
            }
        });
        
        // Load entity data with confidence-based styling
        entityGraph.loadData(entityData);
        
        // Setup entity selection and detail view
        entityGraph.onEntitySelect((entity) => {
            this.displayEntityDetails(entity);
            this.highlightRelatedEntities(entity);
        });
        
        this.visualizations.set('entityGraph', entityGraph);
    }
    
    async initializeTimelineVisualization(timelineData) {
        """Create interactive timeline for temporal analysis"""
        const timelineContainer = this.createVisualizationContainer('timeline');
        
        const timeline = new IntelligenceTimeline(timelineContainer, {
            timeRange: 'auto',
            granularity: 'adaptive',
            eventTypes: ['search', 'verification', 'discovery', 'synthesis'],
            confidenceVisualization: 'heatmap',
            interactiveZoom: true,
            brushSelection: true,
            annotations: {
                enabled: true,
                collaborative: true
            }
        });
        
        // Load timeline data with confidence indicators
        timeline.loadEvents(timelineData);
        
        // Setup temporal filtering
        timeline.onTimeRangeSelect((startTime, endTime) => {
            this.filterDashboardByTimeRange(startTime, endTime);
        });
        
        this.visualizations.set('timeline', timeline);
    }
    
    async initializeGeospatialMap(locationData) {
        """Create interactive geospatial intelligence map"""
        const mapContainer = this.createVisualizationContainer('geospatial-map');
        
        const geoMap = new GeospatialIntelligenceMap(mapContainer, {
            baseLayer: 'satellite',
            clustering: {
                enabled: true,
                maxZoom: 12,
                radiusFunction: (count) => Math.sqrt(count) * 10
            },
            heatmap: {
                enabled: true,
                radius: 20,
                opacity: 0.7,
                confidenceWeighting: true
            },
            markerTypes: ['location', 'event', 'network_node', 'asset'],
            interactionMode: 'analysis'
        });
        
        // Load location data with intelligence context
        geoMap.loadLocations(locationData);
        
        // Setup geographic correlation analysis
        geoMap.onLocationSelect((location) => {
            this.analyzeGeographicCorrelations(location);
        });
        
        this.visualizations.set('geoMap', geoMap);
    }
    
    // ALWAYS implement error boundaries and graceful degradation
    handleDashboardError(message, error) {
        console.error(message, error);
        
        // Display user-friendly error message
        this.displayErrorNotification(message);
        
        // Send error to monitoring service
        if (this.errorTracker) {
            this.errorTracker.captureException(error, {
                context: { 
                    dashboard_state: this.getCurrentDashboardState(),
                    user_action: this.getLastUserAction()
                }
            });
        }
        
        // Attempt graceful degradation
        this.enableFallbackMode();
    }
}
```

---

## 🔧 **Development and Testing Guidelines**

### **10. OSINT-Specific Testing Patterns**
When suggesting test code, include OSINT-specific scenarios:

#### **Comprehensive OSINT Testing Framework**
```python
# PREFERRED: OSINT-specific test patterns with realistic scenarios
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

class TestOSINTInvestigationPipeline:
    """Comprehensive test suite for OSINT investigation pipeline"""
    
    @pytest.fixture
    async def investigation_setup(self):
        """Setup realistic OSINT investigation test environment"""
        mock_search_engines = {
            'tavily': AsyncMock(),
            'whois': AsyncMock(),
            'shodan': AsyncMock()
        }
        
        mock_verification_mcps = {
            'dns_lookup': AsyncMock(),
            'ip_geolocation': AsyncMock(),
            'domain_reputation': AsyncMock()
        }
        
        # Mock realistic search results
        mock_search_engines['tavily'].search.return_value = {
            'results': [
                {
                    'title': 'Company XYZ Annual Report 2024',
                    'url': 'https://company-xyz.com/reports/2024',
                    'content': 'Company XYZ reported revenue growth...',
                    'confidence': 0.9
                },
                {
                    'title': 'Industry Analysis: XYZ Sector Trends',
                    'url': 'https://industry-news.com/xyz-analysis',
                    'content': 'The XYZ sector shows significant...',
                    'confidence': 0.8
                }
            ]
        }
        
        # Mock realistic verification results
        mock_verification_mcps['dns_lookup'].execute.return_value = {
            'domain': 'company-xyz.com',
            'ip_addresses': ['192.168.1.100'],
            'nameservers': ['ns1.example.com', 'ns2.example.com'],
            'verification_status': 'verified'
        }
        
        investigation_pipeline = OSINTInvestigationPipeline(
            search_engines=mock_search_engines,
            verification_mcps=mock_verification_mcps
        )
        
        return investigation_pipeline, mock_search_engines, mock_verification_mcps
    
    @pytest.mark.asyncio
    async def test_multi_source_investigation_accuracy(self, investigation_setup):
        """Test accuracy of multi-source investigation results"""
        pipeline, search_engines, verification_mcps = investigation_setup
        
        # Execute investigation
        result = await pipeline.execute_investigation(
            topic="Company XYZ corporate intelligence",
            sources=['tavily', 'whois'],
            verification_mcps=['dns_lookup', 'domain_reputation']
        )
        
        # Verify search engine calls
        search_engines['tavily'].search.assert_called_once()
        search_engines['whois'].search.assert_called_once()
        
        # Verify verification MCP calls
        verification_mcps['dns_lookup'].execute.assert_called_once()
        
        # Assert investigation quality
        assert result['status'] == 'success'
        assert result['confidence_score'] >= 0.7
        assert len(result['findings']) > 0
        assert 'source_attribution' in result
        
        # Verify confidence calculation accuracy
        confidence_factors = result['confidence_breakdown']
        assert 'source_credibility' in confidence_factors
        assert 'cross_validation' in confidence_factors
        assert 'information_completeness' in confidence_factors
    
    @pytest.mark.asyncio
    async def test_investigation_with_conflicting_sources(self, investigation_setup):
        """Test handling of conflicting information from different sources"""
        pipeline, search_engines, verification_mcps = investigation_setup
        
        # Setup conflicting search results
        search_engines['tavily'].search.return_value = {
            'results': [{'content': 'Company revenue: $100M', 'confidence': 0.8}]
        }
        search_engines['whois'].search.return_value = {
            'results': [{'content': 'Company revenue: $200M', 'confidence': 0.9}]
        }
        
        result = await pipeline.execute_investigation(
            topic="Company financial information",
            sources=['tavily', 'whois']
        )
        
        # Verify conflict detection
        assert 'source_conflicts' in result
        assert len(result['source_conflicts']) > 0
        
        # Verify confidence adjustment for conflicts
        assert result['confidence_score'] < 0.8  # Should be reduced due to conflicts
        
        # Verify conflict resolution recommendations
        assert 'conflict_resolution_recommendations' in result
    
    @pytest.mark.asyncio
    async def test_investigation_performance_under_load(self, investigation_setup):
        """Test investigation performance under concurrent load"""
        pipeline, _, _ = investigation_setup
        
        # Create multiple concurrent investigations
        investigation_tasks = []
        for i in range(10):
            task = asyncio.create_task(
                pipeline.execute_investigation(f"test_topic_{i}")
            )
            investigation_tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*investigation_tasks, return_exceptions=True)
        end_time = time.time()
        
        # Verify performance benchmarks
        total_time = end_time - start_time
        assert total_time < 30  # Should complete within 30 seconds
        
        # Verify all investigations completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # At least 80% success rate
        
        # Verify resource utilization efficiency
        avg_time_per_investigation = total_time / len(successful_results)
        assert avg_time_per_investigation < 5  # Each investigation under 5 seconds
    
    @pytest.mark.asyncio
    async def test_privacy_and_security_compliance(self, investigation_setup):
        """Test privacy and security compliance in investigations"""
        pipeline, _, _ = investigation_setup
        
        # Test PII detection and redaction
        sensitive_query = "Investigate John Doe SSN 123-45-6789 email john@example.com"
        
        result = await pipeline.execute_investigation(sensitive_query)
        
        # Verify PII was detected and handled
        assert 'pii_detected' in result['security_metadata']
        assert result['security_metadata']['pii_detected'] == True
        
        # Verify sensitive data was redacted in logs
        assert 'SSN' not in result['processed_query']
        assert '123-45-6789' not in result['processed_query']
        
        # Verify OPSEC measures were applied
        assert 'opsec_measures' in result['security_metadata']
        assert result['security_metadata']['opsec_measures']['anonymization_applied'] == True
```

---

## 📋 **Response Formatting Guidelines**

### **11. Code Suggestion Format**
When providing code suggestions, always include:

1. **Context Explanation**: Brief explanation of the OSINT concept being implemented
2. **Security Considerations**: Any security or privacy implications
3. **Performance Notes**: Async patterns, caching considerations, rate limiting
4. **Error Handling**: Comprehensive error handling patterns
5. **Testing Guidance**: Suggested test patterns for the code
6. **Documentation**: Type hints, docstrings, and usage examples

### **12. Response Structure Template**
```markdown
## [Feature/Concept Name]

**OSINT Context**: [Brief explanation of how this relates to OSINT operations]

**Implementation Approach**:
```python
# [Your code suggestion with comprehensive comments]
```

**Security Considerations**:
- [List security implications and mitigations]

**Performance Optimization**:
- [Async patterns, caching, rate limiting considerations]

**Testing Strategy**:
- [Key test scenarios for this functionality]

**Integration Notes**:
- [How this integrates with the existing multi-agent architecture]
```

---

## 🎯 **Specialized Response Guidelines**

When responding to requests about this OSINT project, prioritize:

1. **🔍 Intelligence-First Thinking**: Always consider the intelligence value and analytical rigor
2. **🛡️ Security-by-Design**: Include OPSEC, privacy, and security considerations
3. **⚡ Performance at Scale**: Design for concurrent investigations and real-time operations
4. **🧠 Multi-Agent Coordination**: Consider how components fit into the agent ecosystem
5. **📊 Quality Assurance**: Include confidence scoring and verification mechanisms
6. **🔄 Continuous Learning**: Design for improvement and pattern recognition
7. **📈 Professional Standards**: Code should meet enterprise intelligence platform standards

This configuration ensures that all assistance provided aligns with advanced OSINT platform development standards and the integrated improvement roadmap defined in the project documentation.
