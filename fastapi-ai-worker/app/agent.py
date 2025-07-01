# Contains the advanced LangGraph agent with memory and MCP integration.

import os
import json
import time
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from tavily import TavilyClient  # type: ignore[import-untyped]

from .llm_selector import get_llm_instance
from .langfuse_tracker import agent_tracker
from . import mcps

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    print("⚠️ Warning: TAVILY_API_KEY not found in environment variables. Search functionality will be limited.")
    tavily_client = None
else:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

class AgentState(TypedDict):
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
    planner_reasoning: str  # Strategic reasoning from planner
    synthesis_confidence: str  # Confidence level of synthesis
    information_gaps: List[str]  # Identified information gaps
    search_quality_metrics: Dict  # Search quality assessment
    query_performance: List[Dict]  # Individual query performance data
    mcp_execution_results: List[Dict]  # Detailed MCP execution results
    verification_strategy: str  # MCP verification strategy
    final_confidence_assessment: str  # Final confidence assessment
    final_risk_indicators: List[str]  # Final risk indicators
    final_verification_summary: str  # Final verification summary
    final_actionable_recommendations: List[str]  # Final actionable recommendations
    final_information_reliability: str  # Final information reliability assessment
    report_quality_metrics: Dict  # Report quality assessment metrics
class PlannerSchema(BaseModel):
    search_queries: List[str] = Field(
        description="A list of 3-5 strategic OSINT search queries, each 5-15 words long, targeting different information sources and angles",
        min_length=3
    )
    reasoning: str = Field(
        description="Brief explanation of the research strategy and how these queries complement each other",
        min_length=50,
        max_length=500,
        default=""
    )

async def planner_node(state: AgentState):
    with agent_tracker.track_node_execution("planner", {"topic": state["topic"], "memory_size": len(state.get("long_term_memory", []))}) as span:
        llm = get_llm_instance(state["model_id"], state["temperature"])
        memory_str = json.dumps(state.get("long_term_memory", []), indent=2)
        
        # Enhanced prompt following Anthropic/OpenAI best practices
        prompt = f"""<role>
You are a professional OSINT (Open Source Intelligence) analyst with expertise in systematic information gathering, verification, and intelligence synthesis. Your role is to create strategic research plans that maximize information discovery while maintaining operational security.
</role>

<task>
Analyze the research topic and create a comprehensive OSINT research plan with targeted search queries.
</task>

<research_topic>
{state['topic']}
</research_topic>

<historical_context>
Previous approved research findings and patterns:
{memory_str if memory_str.strip() != '[]' else 'No previous findings available for this research topic.'}
</historical_context>

<methodology>
Apply the OSINT intelligence cycle:
1. Planning & Direction: Identify information requirements
2. Collection: Determine optimal search strategies
3. Processing: Consider data source reliability
4. Analysis: Plan for information correlation
5. Dissemination: Structure for actionable intelligence

Consider these OSINT principles:
- Start broad, then narrow focus
- Use multiple source types (social media, public records, news, technical data)
- Plan for cross-reference verification
- Consider temporal aspects (recent vs historical data)
- Account for geographic and linguistic variations
</methodology>

<instructions>
Create 3-5 targeted search queries following these criteria:

1. **Specificity**: Use precise terminology, names, identifiers, and technical terms
2. **Diversity**: Cover different angles - technical, social, business, historical
3. **Actionability**: Each query should yield concrete, verifiable information
4. **Intelligence Value**: Focus on queries that provide decision-making insight
5. **Source Variety**: Design queries to hit different types of sources

For each query, consider:
- What specific information gap does this address?
- What type of sources are most likely to contain this information?
- How does this complement the other queries?
- What verification opportunities does this create?
</instructions>

<format_requirements>
- Generate exactly 3-5 search queries
- Each query must be 5-15 words long
- Use specific terminology, not generic phrases
- Include relevant identifiers (domains, usernames, companies, locations, dates)
- Avoid duplicate information paths
- Structure queries for web search engines
</format_requirements>

<examples>
For topic "Suspicious domain xyz-corp.com":
- "xyz-corp.com WHOIS registration history owner details"
- "xyz-corp.com malware phishing reports security analysis"
- "xyz-corp company business registration corporate filings"
- "xyz-corp.com SSL certificate authority issuer details"

For topic "Social media account @suspicioususer":
- "@suspicioususer Twitter account creation date early posts"
- "suspicioususer Instagram Facebook cross-platform presence"
- "suspicioususer email phone number data breaches leaks"
</examples>

<thinking>
Let me analyze the research topic and historical context:

1. What is the core subject of investigation?
2. What information gaps exist based on previous findings?
3. What are the most critical intelligence requirements?
4. Which search approaches will yield the highest value information?
5. How can I ensure comprehensive coverage without redundancy?
</thinking>

Generate your research plan with 3-5 strategic search queries that will maximize information discovery for this OSINT investigation."""
        structured_llm = llm.with_structured_output(PlannerSchema)
        response = await structured_llm.ainvoke(prompt)
        
        # Track the LLM call with enhanced metadata
        agent_tracker.track_llm_call("planner", state["model_id"], prompt, response, state["temperature"])
        
        # Extract search_queries and reasoning from response
        if hasattr(response, 'search_queries'):
            search_queries = response.search_queries
            reasoning = getattr(response, 'reasoning', '')
        elif isinstance(response, dict):
            search_queries = response.get('search_queries', [])
            reasoning = response.get('reasoning', '')
        else:
            search_queries = []
            reasoning = ''
        
        # Enhanced validation with quality checks
        if not search_queries or len(search_queries) < 3:
            # Fallback queries with more sophisticated OSINT approach
            topic = state['topic']
            search_queries = [
                f'"{topic}" background information public records',
                f'"{topic}" social media presence online activity',
                f'"{topic}" news articles recent developments',
                f'"{topic}" related entities connections associates'
            ]
            reasoning = "Fallback queries generated due to insufficient structured response"
        
        # Quality validation - check query length and specificity
        validated_queries = []
        for query in search_queries:
            words = query.split()
            if 5 <= len(words) <= 15 and len(query.strip()) > 10:
                validated_queries.append(query.strip())
            else:
                # Enhance generic queries
                enhanced_query = f'"{state["topic"]}" {query.strip()}'[:100]
                validated_queries.append(enhanced_query)
        
        search_queries = validated_queries[:5]  # Limit to max 5 queries
        
        if span:
            span.update(output={
                "search_queries_count": len(search_queries), 
                "search_queries": search_queries,
                "reasoning": reasoning,
                "query_quality_score": sum(1 for q in search_queries if 5 <= len(q.split()) <= 15) / len(search_queries),
                "has_reasoning": bool(reasoning)
            })
        
        return {
            "search_queries": search_queries, 
            "num_steps": 1,
            "planner_reasoning": reasoning
        }

async def search_node(state: AgentState):
    with agent_tracker.track_node_execution("search", {"queries_count": len(state["search_queries"])}) as span:
        search_queries = state["search_queries"]
        topic = state["topic"]
        
        # Enhanced search execution with professional OSINT practices
        all_results = []
        query_metadata = []
        successful_queries = 0
        failed_queries = 0
        
        for i, query in enumerate(search_queries, 1):
            query_start_time = time.time()
            query_meta = {
                "query": query,
                "query_number": i,
                "word_count": len(query.split()),
                "query_type": _classify_query_type(query, topic)
            }
            
            try:
                # Dynamic search depth based on query complexity and importance
                search_depth = _determine_search_depth(query, topic)
                max_results = _determine_max_results(query, i, len(search_queries))
                
                # Execute search with enhanced parameters
                if not tavily_client:
                    raise Exception("Tavily API client not available. Please check TAVILY_API_KEY environment variable.")
                
                response = tavily_client.search(
                    query=query,
                    search_depth=search_depth,
                    max_results=max_results,
                    include_domains=None,  # Could be enhanced with domain filtering
                    exclude_domains=None   # Could be enhanced with domain exclusion
                )
                
                # Process and enhance results
                query_results = response.get("results", [])
                enhanced_results = []
                
                for result in query_results:
                    enhanced_result = {
                        **result,
                        "query_source": query,
                        "query_number": i,
                        "relevance_score": _calculate_relevance_score(result, query, topic),
                        "source_type": _classify_source_type(result.get("url", "")),
                        "content_length": len(result.get("content", "")),
                        "has_date": bool(result.get("published_date")),
                        "credibility_indicators": _assess_source_credibility(result)
                    }
                    enhanced_results.append(enhanced_result)
                
                # Sort results by relevance for this query
                enhanced_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                all_results.extend(enhanced_results)
                
                query_meta.update({
                    "status": "success",
                    "results_count": len(enhanced_results),
                    "execution_time": time.time() - query_start_time,
                    "average_relevance": sum(r.get("relevance_score", 0) for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0,
                    "source_diversity": len(set(r.get("source_type", "unknown") for r in enhanced_results))
                })
                successful_queries += 1
                
            except Exception as e:
                error_result = {
                    "query": query,
                    "query_number": i,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed",
                    "relevance_score": 0,
                    "source_type": "error"
                }
                all_results.append(error_result)
                
                query_meta.update({
                    "status": "failed",
                    "error": str(e),
                    "execution_time": time.time() - query_start_time,
                    "results_count": 0
                })
                failed_queries += 1
            
            query_metadata.append(query_meta)
        
        # Post-processing: Remove duplicates and rank by overall relevance
        deduplicated_results = _remove_duplicate_results(all_results)
        final_results = _rank_and_filter_results(deduplicated_results, topic, max_total_results=20)
        
        # Quality assessment
        quality_metrics = _assess_search_quality(final_results, search_queries, topic)
        
        # Track the search operation with enhanced metadata
        total_execution_time = 0.0
        for q in query_metadata:
            exec_time = q.get('execution_time', 0)
            if exec_time is not None and isinstance(exec_time, (int, float, str)):
                try:
                    total_execution_time += float(exec_time)
                except (ValueError, TypeError):
                    continue
        overall_success = successful_queries > 0
        agent_tracker.track_search_operation(
            query="; ".join(search_queries), 
            results=final_results, 
            execution_time=total_execution_time,
            success=overall_success
        )
        
        if span:
            span.update(output={
                "total_results": len(final_results),
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "deduplication_removed": len(all_results) - len(deduplicated_results),
                "final_ranking_kept": len(final_results),
                "average_relevance_score": quality_metrics["average_relevance"],
                "source_diversity_score": quality_metrics["source_diversity"],
                "credibility_score": quality_metrics["credibility_score"],
                "coverage_completeness": quality_metrics["coverage_score"],
                "query_metadata": query_metadata,
                "quality_assessment": quality_metrics
            })
        
        return {
            "search_results": final_results, 
            "num_steps": state["num_steps"] + 1,
            "search_quality_metrics": quality_metrics,
            "query_performance": query_metadata
        }

def _classify_query_type(query: str, topic: str) -> str:
    """Classify the type of OSINT query for better tracking."""
    query_lower = query.lower()
    if any(term in query_lower for term in ["whois", "registration", "domain", "dns"]):
        return "technical_infrastructure"
    elif any(term in query_lower for term in ["social", "twitter", "facebook", "instagram", "linkedin"]):
        return "social_media"
    elif any(term in query_lower for term in ["news", "article", "report", "press"]):
        return "news_media"
    elif any(term in query_lower for term in ["company", "business", "corporate", "filing"]):
        return "business_records"
    elif any(term in query_lower for term in ["security", "malware", "threat", "vulnerability"]):
        return "threat_intelligence"
    else:
        return "general_research"

def _determine_search_depth(query: str, topic: str) -> str:
    """Determine appropriate search depth based on query characteristics."""
    # Use advanced search for technical and business queries that need more comprehensive results
    if any(term in query.lower() for term in ["whois", "registration", "corporate", "filing", "security"]):
        return "advanced"
    else:
        return "basic"

def _determine_max_results(query: str, query_position: int, total_queries: int) -> int:
    """Determine maximum results based on query importance and position."""
    # First query gets more results, technical queries get more results
    base_results = 5 if query_position == 1 else 3
    
    # Increase for technical and business queries
    if any(term in query.lower() for term in ["whois", "corporate", "security", "malware"]):
        base_results += 2
    
    return min(base_results, 8)  # Cap at 8 results per query

def _calculate_relevance_score(result: Dict, query: str, topic: str) -> float:
    """Calculate relevance score for a search result."""
    score = 0.0
    
    # Check title relevance
    title = result.get("title", "").lower()
    content = result.get("content", "").lower()
    url = result.get("url", "").lower()
    
    query_terms = query.lower().split()
    topic_terms = topic.lower().split()
    
    # Title relevance (highest weight)
    for term in query_terms:
        if term in title:
            score += 0.3
    
    # Content relevance
    for term in query_terms:
        if term in content:
            score += 0.2
    
    # URL relevance
    for term in topic_terms:
        if term in url:
            score += 0.1
    
    # Boost for authoritative sources
    if any(domain in url for domain in ["gov", "edu", ".org", "reuters", "bbc", "cnn"]):
        score += 0.2
    
    # Boost for recent content
    if result.get("published_date"):
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

def _classify_source_type(url: str) -> str:
    """Classify the type of source based on URL."""
    url_lower = url.lower()
    
    if any(domain in url_lower for domain in ["twitter.com", "facebook.com", "instagram.com", "linkedin.com"]):
        return "social_media"
    elif any(domain in url_lower for domain in [".gov", "government"]):
        return "government"
    elif any(domain in url_lower for domain in [".edu", "university", "academic"]):
        return "academic"
    elif any(domain in url_lower for domain in ["news", "reuters", "bbc", "cnn", "times"]):
        return "news_media"
    elif any(domain in url_lower for domain in ["github.com", "stackoverflow.com"]):
        return "technical"
    elif any(domain in url_lower for domain in [".org"]):
        return "organization"
    else:
        return "commercial"

def _assess_source_credibility(result: Dict) -> Dict:
    """Assess credibility indicators for a source."""
    url = result.get("url", "").lower()
    title = result.get("title", "")
    content = result.get("content", "")
    
    indicators = {
        "has_https": url.startswith("https://"),
        "is_established_domain": any(domain in url for domain in ["gov", "edu", "reuters", "bbc", "cnn"]),
        "has_date": bool(result.get("published_date")),
        "content_length_adequate": len(content) > 200,
        "title_quality": len(title.split()) > 3,
        "potential_spam": any(spam_indicator in content.lower() for spam_indicator in ["click here", "buy now", "limited time"])
    }
    
    credibility_score = sum(1 for v in indicators.values() if v) / len(indicators)
    indicators["overall_credibility_score"] = credibility_score
    
    return indicators

def _remove_duplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate results based on URL and content similarity."""
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
        elif not url and result not in unique_results:  # Handle error results
            unique_results.append(result)
    
    return unique_results

def _rank_and_filter_results(results: List[Dict], topic: str, max_total_results: int = 20) -> List[Dict]:
    """Rank results by relevance and filter to top results."""
    # Separate successful results from errors
    successful_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]
    
    # Sort successful results by relevance score
    successful_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Take top results, ensuring source diversity
    final_results: list[dict] = []
    source_types_seen = set()
    
    for result in successful_results:
        source_type = result.get("source_type", "unknown")
        
        # Prioritize diverse sources
        if len(final_results) < max_total_results:
            if source_type not in source_types_seen or len(final_results) < max_total_results // 2:
                final_results.append(result)
                source_types_seen.add(source_type)
    
    # Add error results at the end for debugging
    final_results.extend(error_results)
    
    return final_results[:max_total_results]

def _assess_search_quality(results: List[Dict], queries: List[str], topic: str) -> Dict:
    """Assess overall quality of search results."""
    successful_results = [r for r in results if "error" not in r]
    
    if not successful_results:
        return {
            "average_relevance": 0.0,
            "source_diversity": 0.0,
            "credibility_score": 0.0,
            "coverage_score": 0.0,
            "total_sources": 0,
            "quality_grade": "F"
        }
    
    # Calculate metrics
    avg_relevance = sum(r.get("relevance_score", 0) for r in successful_results) / len(successful_results)
    
    source_types = set(r.get("source_type", "unknown") for r in successful_results)
    source_diversity = len(source_types) / 7  # 7 possible source types
    
    credibility_scores = [r.get("credibility_indicators", {}).get("overall_credibility_score", 0) for r in successful_results]
    avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0
    
    # Coverage score based on query fulfillment
    query_coverage = len(set(r.get("query_source", "") for r in successful_results)) / len(queries)
    
    # Overall quality grade
    overall_score = (avg_relevance * 0.4 + source_diversity * 0.2 + avg_credibility * 0.2 + query_coverage * 0.2)
    
    if overall_score >= 0.8:
        grade = "A"
    elif overall_score >= 0.6:
        grade = "B"
    elif overall_score >= 0.4:
        grade = "C"
    elif overall_score >= 0.2:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "average_relevance": round(avg_relevance, 3),
        "source_diversity": round(source_diversity, 3),
        "credibility_score": round(avg_credibility, 3),
        "coverage_score": round(query_coverage, 3),
        "total_sources": len(successful_results),
        "unique_source_types": len(source_types),
        "quality_grade": grade,
        "overall_score": round(overall_score, 3)
    }

class SynthesisSchema(BaseModel):
    synthesized_findings: str = Field(
        description="A comprehensive intelligence report with executive summary, detailed analysis, source assessment, and verification opportunities",
        min_length=200
    )
    confidence_level: str = Field(
        description="Overall confidence in the findings: High, Medium, or Low",
        default="Medium"
    )
    key_information_gaps: List[str] = Field(
        description="List of important information gaps that need further investigation",
        default=[]
    )

async def synthesis_node(state: AgentState):
    with agent_tracker.track_node_execution("synthesis", {"results_count": len(state["search_results"])}) as span:
        llm = get_llm_instance(state["model_id"], state["temperature"])
        results_str = json.dumps(state["search_results"], indent=2)
        planner_reasoning = state.get("planner_reasoning", "")
        
        # Enhanced synthesis prompt with planner context
        prompt = f"""<role>
You are a professional OSINT analyst specializing in intelligence synthesis and report generation.
</role>

<task>
Synthesize the collected search results into a comprehensive, structured intelligence report for the topic: '{state['topic']}'.
</task>

<research_strategy>
Original research strategy: {planner_reasoning}
</research_strategy>

<source_data>
{results_str}
</source_data>

<synthesis_requirements>
1. **Executive Summary**: Key findings and their significance
2. **Detailed Analysis**: Organize findings by information category
3. **Source Assessment**: Evaluate reliability and credibility of sources
4. **Information Gaps**: Identify areas needing further investigation
5. **Verification Opportunities**: Highlight claims that can be fact-checked
6. **Intelligence Assessment**: Confidence levels and analytical judgments

Structure your report with clear sections and maintain analytical objectivity.
Use specific details from the sources while avoiding speculation.
</synthesis_requirements>

Create a comprehensive intelligence report based on the search results."""
        
        structured_llm = llm.with_structured_output(SynthesisSchema)
        response = await structured_llm.ainvoke(prompt)
        
        # Track the LLM call
        agent_tracker.track_llm_call("synthesis", state["model_id"], prompt, response, state["temperature"])
        
        # Extract synthesized_findings and additional metadata from response
        if hasattr(response, 'synthesized_findings'):
            synthesized_findings = response.synthesized_findings
            confidence_level = getattr(response, 'confidence_level', 'Medium')
            key_gaps = getattr(response, 'key_information_gaps', [])
        elif isinstance(response, dict):
            synthesized_findings = response.get('synthesized_findings', '')
            confidence_level = response.get('confidence_level', 'Medium')
            key_gaps = response.get('key_information_gaps', [])
        else:
            synthesized_findings = ''
            confidence_level = 'Low'
            key_gaps = []
        
        if span:
            span.update(output={
                "findings_length": len(synthesized_findings),
                "confidence_level": confidence_level,
                "information_gaps_count": len(key_gaps),
                "used_planner_reasoning": bool(planner_reasoning),
                "report_structure_score": len([s for s in ['Executive Summary', 'Analysis', 'Sources', 'Gaps'] if s.lower() in synthesized_findings.lower()]) / 4
            })
        
        return {
            "synthesized_findings": synthesized_findings, 
            "num_steps": state["num_steps"] + 1,
            "synthesis_confidence": confidence_level,
            "information_gaps": key_gaps
        }

class MCPIdentificationSchema(BaseModel):
    mcp_tasks: List[Dict] = Field(
        description="A list of verification tasks for MCPs with detailed reasoning. Format: [{'mcp_name': 'tool_name', 'input': 'data_to_verify', 'priority': 'high|medium|low', 'reasoning': 'why this verification is important'}]",
        default=[]
    )
    verification_strategy: str = Field(
        description="Overall strategy for fact verification and cross-referencing",
        min_length=50,
        max_length=300,
        default=""
    )

async def mcp_identification_node(state: AgentState):
    with agent_tracker.track_node_execution("mcp_identifier", {"findings_length": len(state.get("synthesized_findings", ""))}) as span:
        llm = get_llm_instance(state["model_id"], state["temperature"])
        
        # Enhanced prompt for comprehensive MCP identification
        prompt = f"""<role>
You are a professional OSINT verification specialist. Your role is to identify all verifiable claims, entities, and technical artifacts from intelligence reports that can be fact-checked using specialized verification tools (MCPs).
</role>

<task>
Analyze the intelligence report and identify all elements that can be verified using available MCP (Mission Critical Protocol) tools.
</task>

<report_to_analyze>
{state['synthesized_findings']}
</report_to_analyze>

<available_mcp_tools>
1. **get_domain_whois**: Verify domain registration details, ownership, creation dates
   - Input: domain name (e.g., "example.com")
   - Verifies: registrant info, creation/expiry dates, nameservers, registrar

2. **check_ip_reputation**: Analyze IP address reputation and threat intelligence
   - Input: IPv4/IPv6 address (e.g., "192.168.1.1")
   - Verifies: malware associations, geographic location, ISP details

3. **verify_email_breach**: Check if email addresses appear in known data breaches
   - Input: email address (e.g., "user@example.com")
   - Verifies: breach history, compromise dates, exposed data types

4. **analyze_url_safety**: Assess URL safety and security indicators
   - Input: full URL (e.g., "https://example.com/path")
   - Verifies: malware status, phishing indicators, reputation scores

5. **get_ssl_certificate**: Extract and verify SSL certificate details
   - Input: domain name (e.g., "example.com")
   - Verifies: certificate authority, validity period, encryption details

6. **check_social_media_account**: Verify social media account details and activity
   - Input: social media handle/username (e.g., "@username")
   - Verifies: account creation, activity patterns, profile information

7. **analyze_file_hash**: Analyze file hashes for malware and threat intelligence
   - Input: file hash (MD5, SHA1, SHA256)
   - Verifies: malware signatures, threat classifications, detection rates

8. **analyze_phone_number**: Validate and analyze phone numbers for geographic and carrier information
   - Input: phone number (e.g., "+1-555-123-4567")
   - Verifies: country, carrier, number type, validity

9. **investigate_crypto_address**: Investigate cryptocurrency addresses for blockchain analysis
   - Input: crypto address (Bitcoin, Ethereum, etc.)
   - Verifies: currency type, transaction history, risk indicators

10. **analyze_dns_records**: Comprehensive DNS record analysis and subdomain enumeration
    - Input: domain name (e.g., "example.com")
    - Verifies: DNS records, security configurations, subdomains

11. **extract_image_metadata**: Extract metadata and EXIF data from images
    - Input: base64 encoded image data
    - Verifies: location data, device information, timestamps

12. **analyze_network_port**: Analyze specific network ports for service detection
    - Input: IP address and port number (e.g., "192.168.1.1", 80)
    - Verifies: service identification, security assessment

13. **check_paste_site_exposure**: Check for data exposure on paste sites
    - Input: search term or identifier
    - Verifies: potential data leaks, sensitive information exposure
</available_mcp_tools>

<identification_instructions>
1. **Systematic Extraction**: Scan the report for:
   - Domain names (any format: example.com, sub.domain.org)
   - IP addresses (IPv4/IPv6 formats)
   - Email addresses (user@domain.com format)
   - URLs (http/https links)
   - Social media handles (@username, platform/username)
   - File hashes (MD5, SHA1, SHA256 strings)

2. **Prioritization**: Assign priority levels:
   - **High**: Critical to case, mentioned multiple times, central to findings
   - **Medium**: Supporting evidence, mentioned once, circumstantial
   - **Low**: Tangential, background information, less reliable sources

3. **Verification Strategy**: Consider:
   - Which verifications will provide the most valuable intelligence
   - What cross-referencing opportunities exist
   - How verification results could impact case conclusions
   - Which claims need the strongest evidence support

4. **Quality Focus**: Only include:
   - Well-formed, valid inputs (proper domain/IP/email/phone/crypto formats)
   - Items that could meaningfully impact the investigation
   - Entities where verification adds intelligence value
</identification_instructions>

<output_format>
For each identified verification opportunity, provide:
- mcp_name: The specific tool to use
- input: The exact data to verify (clean, properly formatted)
- priority: high/medium/low based on investigative importance
- reasoning: Why this verification is important to the case

For network port analysis, use format "ip:port" (e.g., "192.168.1.1:80")

Also provide an overall verification strategy explaining how these checks will enhance the intelligence picture.
</output_format>

<examples>
For a report mentioning "suspicious activity from badsite.com, IP 1.2.3.4, and phone +1-555-123-4567":
```json
{
  "mcp_tasks": [
    {
      "mcp_name": "get_domain_whois",
      "input": "badsite.com",
      "priority": "high",
      "reasoning": "Central domain in investigation - ownership and registration details critical for attribution"
    },
    {
      "mcp_name": "check_ip_reputation",
      "input": "1.2.3.4", 
      "priority": "high",
      "reasoning": "Associated IP address - reputation check will reveal malicious activity history"
    },
    {
      "mcp_name": "analyze_phone_number",
      "input": "+1-555-123-4567",
      "priority": "medium",
      "reasoning": "Phone number verification will provide geographic and carrier attribution"
    },
    {
      "mcp_name": "analyze_dns_records",
      "input": "badsite.com",
      "priority": "medium",
      "reasoning": "DNS analysis will reveal infrastructure patterns and security configurations"
    }
  ],
  "verification_strategy": "Cross-reference domain ownership with IP geolocation and phone carrier data to establish comprehensive operational infrastructure profile"
}
```
</examples>

Analyze the report and identify all verification opportunities that will strengthen the intelligence assessment."""
        
        structured_llm = llm.with_structured_output(MCPIdentificationSchema)
        response = await structured_llm.ainvoke(prompt)
        
        # Track the LLM call
        agent_tracker.track_llm_call("mcp_identifier", state["model_id"], prompt, response, state["temperature"])
        
        # Extract mcp_tasks and strategy from response
        if hasattr(response, 'mcp_tasks'):
            mcp_tasks = response.mcp_tasks
            verification_strategy = getattr(response, 'verification_strategy', '')
        elif isinstance(response, dict):
            mcp_tasks = response.get('mcp_tasks', [])
            verification_strategy = response.get('verification_strategy', '')
        else:
            mcp_tasks = []
            verification_strategy = ''
        
        # Enhanced validation and prioritization
        validated_tasks = []
        for task in mcp_tasks:
            if isinstance(task, dict) and task.get('mcp_name') and task.get('input'):
                # Validate input format based on MCP type
                if _validate_mcp_input(task['mcp_name'], task['input']):
                    validated_tasks.append({
                        **task,
                        'priority': task.get('priority', 'medium'),
                        'reasoning': task.get('reasoning', 'Verification identified by automated analysis')
                    })
        
        # Sort by priority: high -> medium -> low
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        validated_tasks.sort(key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        
        if span:
            span.update(output={
                "mcp_tasks_count": len(validated_tasks),
                "mcp_tasks": validated_tasks,
                "verification_strategy": verification_strategy,
                "high_priority_tasks": len([t for t in validated_tasks if t.get('priority') == 'high']),
                "task_types": list(set(t.get('mcp_name') for t in validated_tasks))
            })
        
        return {
            "mcp_verification_list": validated_tasks, 
            "num_steps": state["num_steps"] + 1,
            "verification_strategy": verification_strategy
        }

def _validate_mcp_input(mcp_name: str, input_data: str) -> bool:
    """Validate input format for different MCP tools."""
    import re
    
    input_data = input_data.strip()
    
    if mcp_name == "get_domain_whois":
        # Basic domain validation
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return re.match(domain_pattern, input_data) is not None
    
    elif mcp_name == "check_ip_reputation":
        # IPv4 and IPv6 validation
        ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        ipv6_pattern = r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$'
        return bool(re.match(ipv4_pattern, input_data) or re.match(ipv6_pattern, input_data))
    
    elif mcp_name == "verify_email_breach":
        # Email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_pattern, input_data) is not None
    
    elif mcp_name == "analyze_url_safety":
        # URL validation
        url_pattern = r'^https?:\/\/[^\s/$.?#].[^\s]*$'
        return re.match(url_pattern, input_data) is not None
    
    elif mcp_name == "get_ssl_certificate":
        # Domain validation (same as WHOIS)
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return re.match(domain_pattern, input_data) is not None
    
    elif mcp_name == "check_social_media_account":
        # Social media handle validation
        handle_pattern = r'^@?[a-zA-Z0-9_]{1,50}$'
        return re.match(handle_pattern, input_data) is not None
    
    elif mcp_name == "analyze_file_hash":
        # Hash validation (MD5, SHA1, SHA256)
        md5_pattern = r'^[a-fA-F0-9]{32}$'
        sha1_pattern = r'^[a-fA-F0-9]{40}$'
        sha256_pattern = r'^[a-fA-F0-9]{64}$'
        return bool(re.match(md5_pattern, input_data) or 
                    re.match(sha1_pattern, input_data) or 
                    re.match(sha256_pattern, input_data))
    
    elif mcp_name == "analyze_phone_number":
        # Phone number validation (international format)
        phone_pattern = r'^[\+]?[1-9]\d{1,14}$'
        return bool(re.match(phone_pattern, input_data.replace('-', '').replace(' ', '').replace('(', '').replace(')', '')))
    
    elif mcp_name == "investigate_crypto_address":
        # Cryptocurrency address validation (basic patterns)
        crypto_patterns = [
            r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$',  # Bitcoin legacy
            r'^bc1[a-z0-9]{39,59}$',  # Bitcoin bech32
            r'^0x[a-fA-F0-9]{40}$',  # Ethereum
            r'^[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}$',  # Litecoin
            r'^4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}$',  # Monero
            r'^r[0-9a-zA-Z]{24,34}$'  # Ripple
        ]
        return any(re.match(pattern, input_data) for pattern in crypto_patterns)
    
    elif mcp_name == "analyze_dns_records":
        # Domain validation (same as WHOIS)
        domain_pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$'
        return re.match(domain_pattern, input_data) is not None
    
    elif mcp_name == "extract_image_metadata":
        # Base64 image data validation
        try:
            # Check if it's valid base64
            import base64
            # Remove data:image/...;base64, prefix if present
            if ',' in input_data:
                input_data = input_data.split(',')[1]
            base64.b64decode(input_data)
            return True
        except Exception:
            return False
    
    elif mcp_name == "analyze_network_port":
        # Should be in format "ip:port" or separate validation
        if ':' in input_data:
            ip, port = input_data.split(':', 1)
            ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
            try:
                port_num = int(port)
                return bool(re.match(ipv4_pattern, ip) and 1 <= port_num <= 65535)
            except Exception:
                return False
        return False
    
    elif mcp_name == "check_paste_site_exposure":
        # Any non-empty search term is valid
        return len(input_data.strip()) > 0
    
    return True  # Default to valid for unknown MCP types

async def mcp_execution_node(state: AgentState):
    with agent_tracker.track_node_execution("mcp_executor", {"tasks_count": len(state.get("mcp_verification_list", []))}) as span:
        mcp_tasks = state.get("mcp_verification_list", [])
        verified_data = {}
        execution_results = []
        successful_verifications = 0
        failed_verifications = 0
        
        for task in mcp_tasks:
            mcp_name = task.get('mcp_name')
            input_data = task.get('input')
            priority = task.get('priority', 'medium')
            reasoning = task.get('reasoning', '')
            
            if not mcp_name or not input_data:
                continue
                
            verification_start_time = time.time()
            
            try:
                # Route to appropriate MCP function
                if mcp_name == 'get_domain_whois':
                    result = mcps.get_domain_whois(input_data)
                elif mcp_name == 'check_ip_reputation':
                    result = mcps.check_ip_reputation(input_data)
                elif mcp_name == 'verify_email_breach':
                    result = mcps.verify_email_breach(input_data)
                elif mcp_name == 'analyze_url_safety':
                    result = mcps.analyze_url_safety(input_data)
                elif mcp_name == 'get_ssl_certificate':
                    result = mcps.get_ssl_certificate(input_data)
                elif mcp_name == 'check_social_media_account':
                    result = mcps.check_social_media_account(input_data)
                elif mcp_name == 'analyze_file_hash':
                    result = mcps.analyze_file_hash(input_data)
                elif mcp_name == 'analyze_phone_number':
                    result = mcps.analyze_phone_number(input_data)
                elif mcp_name == 'investigate_crypto_address':
                    result = mcps.investigate_crypto_address(input_data)
                elif mcp_name == 'analyze_dns_records':
                    result = mcps.analyze_dns_records(input_data)
                elif mcp_name == 'extract_image_metadata':
                    result = mcps.extract_image_metadata(input_data)
                elif mcp_name == 'analyze_network_port':
                    # Parse IP:port format
                    if ':' in input_data:
                        ip, port_str = input_data.split(':', 1)
                        try:
                            port = int(port_str)
                            result = mcps.analyze_network_port(ip, port)
                        except ValueError:
                            result = {"error": f"Invalid port number: {port_str}"}
                    else:
                        result = {"error": "Input must be in format 'ip:port'"}
                elif mcp_name == 'check_paste_site_exposure':
                    result = mcps.check_paste_site_exposure(input_data)
                else:
                    result = {"error": f"Unknown MCP tool: {mcp_name}"}
                
                # Store result with metadata
                verification_key = f"{mcp_name}:{input_data}"
                verified_data[verification_key] = {
                    "mcp_name": mcp_name,
                    "input": input_data,
                    "result": result,
                    "priority": priority,
                    "reasoning": reasoning,
                    "execution_time": time.time() - verification_start_time,
                    "success": "error" not in result,
                    "timestamp": time.time()
                }
                
                # Track individual MCP operation
                agent_tracker.track_mcp_operation(mcp_name, input_data, result, time.time() - verification_start_time)
                
                execution_results.append({
                    "mcp_name": mcp_name,
                    "input": input_data,
                    "success": "error" not in result,
                    "execution_time": time.time() - verification_start_time
                })
                
                if "error" not in result:
                    successful_verifications += 1
                else:
                    failed_verifications += 1
                    
            except Exception as e:
                error_result = {"error": f"Exception during {mcp_name} execution: {str(e)}"}
                verification_key = f"{mcp_name}:{input_data}"
                verified_data[verification_key] = {
                    "mcp_name": mcp_name,
                    "input": input_data,
                    "result": error_result,
                    "priority": priority,
                    "reasoning": reasoning,
                    "execution_time": time.time() - verification_start_time,
                    "success": False,
                    "timestamp": time.time(),
                    "exception": str(e)
                }
                
                execution_results.append({
                    "mcp_name": mcp_name,
                    "input": input_data,
                    "success": False,
                    "execution_time": time.time() - verification_start_time,
                    "error": str(e)
                })
                
                failed_verifications += 1
        
        if span:
            span.update(output={
                "total_verifications": len(mcp_tasks),
                "successful_verifications": successful_verifications,
                "failed_verifications": failed_verifications,
                "verification_types": list(set(t.get('mcp_name') for t in mcp_tasks)),
                "high_priority_completed": len([r for r in execution_results if r.get('success') and 
                                              any(t.get('priority') == 'high' and t.get('mcp_name') == r.get('mcp_name') 
                                                  for t in mcp_tasks)]),
                "execution_summary": execution_results
            })
        
        return {
            "verified_data": verified_data, 
            "num_steps": state["num_steps"] + 1,
            "mcp_execution_results": execution_results
        }

class FinalReportSchema(BaseModel):
    final_report: str = Field(
        description="A comprehensive, structured intelligence report with executive summary, detailed findings, verification results, risk assessment, and actionable recommendations",
        min_length=500
    )
    confidence_assessment: str = Field(
        description="Overall confidence level in the intelligence assessment: High, Medium, Low with justification",
        default="Medium"
    )
    risk_indicators: List[str] = Field(
        description="List of identified risk indicators and threat levels",
        default=[]
    )
    verification_summary: str = Field(
        description="Summary of MCP verification results and their impact on the assessment",
        min_length=100,
        default=""
    )
    actionable_recommendations: List[str] = Field(
        description="Specific, actionable recommendations based on the intelligence findings",
        default=[]
    )
    information_reliability: str = Field(
        description="Assessment of source reliability and information credibility using intelligence standards",
        default="Moderate"
    )

async def update_report_node(state: AgentState):
    with agent_tracker.track_node_execution("final_updater", {"verified_data_count": len(state.get("verified_data", {}))}) as span:
        llm = get_llm_instance(state["model_id"], state["temperature"])
        
        # Gather comprehensive context for report generation
        verified_data = state.get("verified_data", {})
        mcp_execution_results = state.get("mcp_execution_results", [])
        # Analyze verification results for intelligence assessment
        verification_analysis = _analyze_verification_results(verified_data, mcp_execution_results)
        
        # Gather comprehensive context for report generation
        search_quality_metrics = state.get("search_quality_metrics", {})
        synthesis_confidence = state.get("synthesis_confidence", "Medium")
        verification_strategy = state.get("verification_strategy", "")
        information_gaps = state.get("information_gaps", [])
        planner_reasoning = state.get("planner_reasoning", "")
        
        # Enhanced prompt for comprehensive intelligence report generation
        prompt = f"""<role>
You are a senior intelligence analyst specializing in OSINT (Open Source Intelligence) report generation. Your expertise includes threat assessment, risk analysis, and actionable intelligence production for decision-makers.
</role>

<task>
Generate a comprehensive, professional intelligence report that integrates all collected data, verification results, and analytical assessments into a structured intelligence product suitable for executive briefing.
</task>

<investigation_context>
**Original Intelligence Requirement**: {state['topic']}
**Research Strategy**: {planner_reasoning}
**Search Quality Assessment**: {search_quality_metrics.get('quality_grade', 'Unknown')} grade with {search_quality_metrics.get('total_sources', 0)} sources analyzed
**Synthesis Confidence**: {synthesis_confidence}
</investigation_context>

<draft_intelligence_assessment>
{state['synthesized_findings']}
</draft_intelligence_assessment>

<verification_results>
**Verification Strategy**: {verification_strategy}
**Verification Summary**: {verification_analysis.get('summary', 'No verification data available')}
**Successful Verifications**: {verification_analysis.get('successful_count', 0)} of {verification_analysis.get('total_count', 0)}
**High-Priority Verifications**: {verification_analysis.get('high_priority_results', [])}

**Detailed Verification Data**:
{json.dumps(verified_data, indent=2) if verified_data else 'No verification data available'}
</verification_results>

<analytical_context>
**Information Gaps**: {information_gaps}
**Source Reliability**: {search_quality_metrics.get('credibility_score', 'Unknown')}
**Coverage Completeness**: {search_quality_metrics.get('coverage_score', 'Unknown')}
**Verification Impact**: {verification_analysis.get('impact_assessment', 'Minimal impact')}
</analytical_context>

<report_requirements>
Generate a comprehensive intelligence report with the following structure:

1. **EXECUTIVE SUMMARY** (2-3 paragraphs)
   - Key findings and their significance
   - Primary threats or opportunities identified
   - Critical decisions or actions required

2. **INTELLIGENCE ASSESSMENT** (detailed analysis)
   - Comprehensive analysis of all findings
   - Integration of verified and unverified information
   - Threat/risk level assessment with justification
   - Attribution and actor analysis where applicable

3. **VERIFICATION ANALYSIS** (MCP results integration)
   - Summary of verification activities conducted
   - Impact of verified data on overall assessment
   - Reliability grading of key claims
   - Technical indicators confirmed or refuted

4. **SOURCE ASSESSMENT** (reliability evaluation)
   - Source diversity and reliability analysis
   - Information credibility assessment using NATO standards:
     * A (Completely reliable) to F (Reliability cannot be judged)
     * 1 (Confirmed) to 6 (Truth cannot be judged)
   - Potential bias or limitation identification

5. **RISK INDICATORS** (threat assessment)
   - Specific risk factors identified
   - Threat level classification (Critical/High/Medium/Low)
   - Indicators of compromise or malicious activity
   - Potential impact assessment

6. **GAPS AND LIMITATIONS**
   - Information gaps requiring further investigation
   - Analytical limitations and assumptions
   - Recommendations for additional collection

7. **ACTIONABLE RECOMMENDATIONS**
   - Specific, time-bound actions recommended
   - Mitigation strategies for identified risks
   - Further investigation priorities
   - Monitoring and follow-up requirements

8. **CONFIDENCE ASSESSMENT**
   - Overall confidence level with detailed justification
   - Factors supporting and limiting confidence
   - Alternative scenarios or interpretations
</report_requirements>

<analytical_standards>
- Use intelligence community analytical standards
- Clearly distinguish between facts, assessments, and judgments
- Provide confidence levels for key assessments
- Include alternative hypotheses where appropriate
- Use precise, professional language suitable for executive briefing
- Ensure all claims are supported by evidence or clearly marked as assessments
- Apply critical thinking and avoid analytical bias
</analytical_standards>

<formatting_requirements>
- Use clear section headers and bullet points for readability
- Include specific data points and metrics where available
- Highlight critical findings and urgent recommendations
- Maintain professional tone throughout
- Structure for executive-level consumption
</formatting_requirements>

Generate the comprehensive intelligence report that integrates all available information and provides actionable intelligence for decision-makers."""

        structured_llm = llm.with_structured_output(FinalReportSchema)
        response = await structured_llm.ainvoke(prompt)
        
        # Track the LLM call with enhanced metadata
        agent_tracker.track_llm_call("final_updater", state["model_id"], prompt, response, state["temperature"])
        
        # Extract comprehensive response data
        if hasattr(response, 'final_report'):
            final_report = response.final_report
            confidence_assessment = getattr(response, 'confidence_assessment', 'Medium')
            risk_indicators = getattr(response, 'risk_indicators', [])
            verification_summary = getattr(response, 'verification_summary', '')
            actionable_recommendations = getattr(response, 'actionable_recommendations', [])
            information_reliability = getattr(response, 'information_reliability', 'Moderate')
        elif isinstance(response, dict):
            final_report = response.get('final_report', '')
            confidence_assessment = response.get('confidence_assessment', 'Medium')
            risk_indicators = response.get('risk_indicators', [])
            verification_summary = response.get('verification_summary', '')
            actionable_recommendations = response.get('actionable_recommendations', [])
            information_reliability = response.get('information_reliability', 'Moderate')
        else:
            final_report = str(response) if response else 'Report generation failed'
            confidence_assessment = 'Low'
            risk_indicators = []
            verification_summary = ''
            actionable_recommendations = []
            information_reliability = 'Poor'
        
        # Enhanced quality assessment of the final report
        report_quality_metrics = _assess_report_quality(
            final_report, verified_data, search_quality_metrics, 
            len(state.get("search_results", [])), verification_analysis
        )
        
        if span:
            span.update(output={
                "final_report_length": len(final_report),
                "confidence_assessment": confidence_assessment,
                "risk_indicators_count": len(risk_indicators),
                "recommendations_count": len(actionable_recommendations),
                "verification_summary_length": len(verification_summary),
                "information_reliability": information_reliability,
                "report_quality_score": report_quality_metrics.get("overall_score", 0),
                "sections_detected": report_quality_metrics.get("sections_count", 0),
                "integration_quality": report_quality_metrics.get("integration_score", 0)
            })
        
        return {
            "synthesized_findings": final_report,
            "final_confidence_assessment": confidence_assessment,
            "final_risk_indicators": risk_indicators,
            "final_verification_summary": verification_summary,
            "final_actionable_recommendations": actionable_recommendations,
            "final_information_reliability": information_reliability,
            "report_quality_metrics": report_quality_metrics,
            "num_steps": state["num_steps"] + 1
        }

def _analyze_verification_results(verified_data: Dict, execution_results: List[Dict]) -> Dict:
    """Analyze MCP verification results to provide intelligence assessment context."""
    if not verified_data and not execution_results:
        return {
            "summary": "No verification activities conducted",
            "total_count": 0,
            "successful_count": 0,
            "high_confidence_count": 0,
            "high_priority_results": [],
            "impact_assessment": "No verification impact"
        }
    
    successful_verifications = []
    failed_verifications = []
    high_priority_results = []
    verification_types = set()
    
    # Analyze verification data
    for key, verification in verified_data.items():
        verification_types.add(verification.get('mcp_name', 'unknown'))
        
        if verification.get('success', False):
            successful_verifications.append(verification)
            
            # Check for high-impact results
            if verification.get('priority') == 'high':
                high_priority_results.append({
                    "type": verification.get('mcp_name'),
                    "input": verification.get('input'),
                    "reasoning": verification.get('reasoning', ''),
                    "key_findings": _extract_key_findings(verification.get('result', {}))
                })
        else:
            failed_verifications.append(verification)
    
    # Calculate impact assessment
    impact_level = "Minimal impact"
    if len(high_priority_results) >= 3:
        impact_level = "High impact - multiple critical verifications completed"
    elif len(high_priority_results) >= 1:
        impact_level = "Medium impact - key verifications provide actionable intelligence"
    elif len(successful_verifications) >= 3:
        impact_level = "Moderate impact - supporting verifications enhance reliability"
    
    # Generate summary
    total_count = len(verified_data)
    successful_count = len(successful_verifications)
    
    if total_count > 0:
        success_rate = (successful_count / total_count) * 100
        summary = f"Completed {total_count} verification tasks with {success_rate:.0f}% success rate. "
        summary += f"Verified {len(verification_types)} different types of entities. "
        
        if high_priority_results:
            summary += f"High-priority verifications provided critical intelligence on {len(high_priority_results)} key entities."
        else:
            summary += "Supporting verifications enhance overall assessment reliability."
    else:
        summary = "No verification data available for analysis."
    
    return {
        "summary": summary,
        "total_count": total_count,
        "successful_count": successful_count,
        "failed_count": len(failed_verifications),
        "success_rate": (successful_count / total_count * 100) if total_count > 0 else 0,
        "high_confidence_count": len(high_priority_results),
        "high_priority_results": high_priority_results,
        "verification_types": list(verification_types),
        "impact_assessment": impact_level
    }

def _extract_key_findings(result: Dict) -> List[str]:
    """Extract key findings from MCP verification results."""
    findings: List[str] = []
    
    if not isinstance(result, dict) or "error" in result:
        return findings
    
    mcp_type = result.get("mcp_type", "")
    
    # Domain WHOIS findings
    if mcp_type == "domain_whois":
        if result.get("registrar"):
            findings.append(f"Registrar: {result['registrar']}")
        if result.get("creation_date"):
            findings.append(f"Created: {result['creation_date']}")
        if result.get("registrant_country"):
            findings.append(f"Country: {result['registrant_country']}")
    
    # IP reputation findings
    elif mcp_type == "ip_reputation":
        if result.get("geolocation", {}).get("country"):
            findings.append(f"Location: {result['geolocation']['country']}")
        if result.get("geolocation", {}).get("isp"):
            findings.append(f"ISP: {result['geolocation']['isp']}")
        if result.get("open_ports"):
            findings.append(f"Open ports: {', '.join(map(str, result['open_ports']))}")
    
    # Email breach findings  
    elif mcp_type == "email_breach":
        if result.get("potential_exposure"):
            findings.append("Potential breach exposure detected")
        if result.get("domain"):
            findings.append(f"Domain: {result['domain']}")
    
    return findings[:3]  # Limit to most important findings

def _assess_report_quality(final_report: str, verified_data: Dict, search_metrics: Dict, 
                          source_count: int, verification_analysis: Dict) -> Dict:
    """Assess the quality and completeness of the final intelligence report."""
    
    # Check for required sections
    required_sections = [
        "executive summary", "intelligence assessment", "verification", 
        "source assessment", "risk", "recommendations", "confidence"
    ]
    
    sections_found = 0
    for section in required_sections:
        if section.lower() in final_report.lower():
            sections_found += 1
    
    section_completeness = sections_found / len(required_sections)
    
    # Calculate overall quality score
    length_score = min(len(final_report) / 2000, 1.0)  # Optimal length around 2000 chars
    verification_score = min(verification_analysis.get('success_rate', 0) / 100, 1.0)
    source_score = min(source_count / 10, 1.0)  # Good if 10+ sources
    
    overall_score = (
        section_completeness * 0.4 +
        length_score * 0.3 +
        verification_score * 0.2 +
        source_score * 0.1
    )
    
    # Quality assessment
    if overall_score >= 0.8:
        quality_grade = "Excellent"
    elif overall_score >= 0.6:
        quality_grade = "Good"
    elif overall_score >= 0.4:
        quality_grade = "Satisfactory"
    else:
        quality_grade = "Needs Improvement"
    
    return {
        "overall_score": round(overall_score, 3),
        "quality_grade": quality_grade,
        "sections_count": sections_found,
        "section_completeness": round(section_completeness, 3),
        "report_length": len(final_report),
        "sources_utilized": source_count
    }

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("search", search_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("mcp_identifier", mcp_identification_node)
workflow.add_node("mcp_executor", mcp_execution_node)
workflow.add_node("final_updater", update_report_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "search")
workflow.add_edge("search", "synthesis")
workflow.add_edge("synthesis", "mcp_identifier")
workflow.add_edge("mcp_identifier", "mcp_executor")
workflow.add_edge("mcp_executor", "final_updater")
workflow.add_edge("final_updater", END)

agent_executor = workflow.compile()