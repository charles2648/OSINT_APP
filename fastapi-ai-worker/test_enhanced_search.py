# Test script for the enhanced search node functionality

import asyncio
import json
import pytest
from typing import Dict, List
from unittest.mock import Mock, patch

def create_mock_tavily_response(query: str, num_results: int = 3) -> Dict:
    """Create a mock Tavily API response for testing."""
    base_results = [
        {
            "title": f"Test Result 1 for {query}",
            "url": "https://example.com/result1",
            "content": f"This is test content for {query}. Contains relevant information about the topic with sufficient detail to be useful.",
            "published_date": "2024-01-15T10:30:00Z",
            "score": 0.95
        },
        {
            "title": f"Technical Analysis of {query}",
            "url": "https://security-blog.edu/analysis",
            "content": f"Technical analysis and research findings related to {query}. Academic source with detailed methodology.",
            "published_date": "2024-01-10T14:20:00Z",
            "score": 0.87
        },
        {
            "title": f"News Report: {query} Investigation",
            "url": "https://news.reuters.com/investigation",
            "content": f"Breaking news report covering recent developments in {query} case. Verified by multiple sources.",
            "published_date": "2024-01-20T09:15:00Z",
            "score": 0.92
        },
        {
            "title": f"Government Report on {query}",
            "url": "https://cyber.gov/reports/2024/analysis",
            "content": f"Official government analysis and findings regarding {query}. Authoritative source with verified data.",
            "published_date": "2024-01-18T16:45:00Z",
            "score": 0.98
        },
        {
            "title": f"Social Media Discussion: {query}",
            "url": "https://twitter.com/cybersec/status/123456",
            "content": f"Social media posts and discussions about {query}. Community-generated content with varying reliability.",
            "published_date": "2024-01-22T11:30:00Z",
            "score": 0.65
        }
    ]
    
    return {
        "results": base_results[:num_results]
    }

@pytest.mark.asyncio
async def test_enhanced_search_node():
    """Test the enhanced search node with various scenarios."""
    
    print("🔍 Testing Enhanced Search Node")
    print("=" * 50)
    
    # Import here to avoid issues if module isn't available
    try:
        from app.agent import search_node, _classify_query_type, _calculate_relevance_score, _assess_search_quality
    except ImportError:
        print("❌ Could not import search functions - they may not be implemented yet")
        return
    
    # Test cases with different types of queries
    test_cases = [
        {
            "name": "Domain Investigation",
            "state": {
                "topic": "malicious domain badsite.com",
                "search_queries": [
                    "badsite.com WHOIS registration history owner details",
                    "badsite.com malware reports security analysis",
                    "badsite.com business registration corporate records"
                ],
                "num_steps": 1
            }
        },
        {
            "name": "Social Media Investigation",
            "state": {
                "topic": "suspicious Twitter account @badactor",
                "search_queries": [
                    "@badactor Twitter account creation early posts",
                    "badactor social media cross-platform presence",
                    "badactor email phone number breach data"
                ],
                "num_steps": 1
            }
        },
        {
            "name": "Corporate Investigation",
            "state": {
                "topic": "TechCorp Inc data breach incident",
                "search_queries": [
                    "TechCorp Inc data breach 2024 incident report",
                    "TechCorp security team leadership changes",
                    "TechCorp Inc SEC filings security disclosures"
                ],
                "num_steps": 1
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Topic: {test_case['state']['topic']}")
        print(f"Queries: {len(test_case['state']['search_queries'])}")
        
        # Test query classification
        print(f"\n🏷️  Query Classifications:")
        for j, query in enumerate(test_case['state']['search_queries'], 1):
            query_type = _classify_query_type(query, test_case['state']['topic'])
            print(f"  {j}. {query}")
            print(f"     Type: {query_type}")
        
        # Mock the Tavily client for testing
        with patch('app.agent.tavily_client') as mock_tavily:
            mock_tavily.search.side_effect = lambda query, **kwargs: create_mock_tavily_response(query, kwargs.get('max_results', 3))
            
            try:
                # Test the search node
                result = await search_node(test_case['state'])
                
                print(f"\n✅ Search Results:")
                print(f"  • Total results: {len(result['search_results'])}")
                print(f"  • Search quality metrics available: {bool(result.get('search_quality_metrics'))}")
                print(f"  • Query performance data available: {bool(result.get('query_performance'))}")
                
                if result.get('search_quality_metrics'):
                    metrics = result['search_quality_metrics']
                    print(f"  • Quality grade: {metrics.get('quality_grade', 'N/A')}")
                    print(f"  • Average relevance: {metrics.get('average_relevance', 0):.3f}")
                    print(f"  • Source diversity: {metrics.get('source_diversity', 0):.3f}")
                    print(f"  • Credibility score: {metrics.get('credibility_score', 0):.3f}")
                
                # Test individual result quality
                successful_results = [r for r in result['search_results'] if 'error' not in r]
                if successful_results:
                    print(f"\n📊 Sample Result Analysis:")
                    sample_result = successful_results[0]
                    relevance = sample_result.get('relevance_score', 0)
                    source_type = sample_result.get('source_type', 'unknown')
                    print(f"  • Title: {sample_result.get('title', 'N/A')[:60]}...")
                    print(f"  • Relevance score: {relevance:.3f}")
                    print(f"  • Source type: {source_type}")
                    print(f"  • Has credibility indicators: {bool(sample_result.get('credibility_indicators'))}")
                
            except Exception as e:
                print(f"❌ Error testing search node: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\n🧪 Testing Utility Functions")
    print("-" * 30)
    
    # Test utility functions directly
    test_queries = [
        "example.com WHOIS registration details",
        "@username Twitter social media analysis", 
        "TechCorp business registration filing",
        "malware security threat analysis"
    ]
    
    print("Query Type Classification:")
    for query in test_queries:
        q_type = _classify_query_type(query, "test topic")
        print(f"  • '{query}' → {q_type}")
    
    # Test relevance scoring
    print(f"\nRelevance Score Testing:")
    test_result = {
        "title": "Example.com Domain Analysis Report",
        "content": "Comprehensive analysis of example.com domain registration and security status",
        "url": "https://security.gov/reports/example-analysis"
    }
    
    test_query = "example.com security analysis"
    test_topic = "example.com investigation"
    
    relevance_score = _calculate_relevance_score(test_result, test_query, test_topic)
    print(f"  • Sample relevance score: {relevance_score:.3f}")
    
    # Test quality assessment
    print(f"\nQuality Assessment Testing:")
    mock_results = [
        {"relevance_score": 0.9, "source_type": "government", "credibility_indicators": {"overall_credibility_score": 0.95}, "query_source": "query1"},
        {"relevance_score": 0.7, "source_type": "news_media", "credibility_indicators": {"overall_credibility_score": 0.8}, "query_source": "query2"},
        {"relevance_score": 0.6, "source_type": "social_media", "credibility_indicators": {"overall_credibility_score": 0.5}, "query_source": "query3"}
    ]
    
    quality_metrics = _assess_search_quality(mock_results, ["query1", "query2", "query3"], "test topic")
    print(f"  • Quality grade: {quality_metrics['quality_grade']}")
    print(f"  • Overall score: {quality_metrics['overall_score']:.3f}")
    
    print(f"\n🏁 Enhanced Search Node Testing Completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_search_node())
