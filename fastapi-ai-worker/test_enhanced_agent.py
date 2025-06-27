# Test script for the enhanced OSINT agent with improved planner prompts

import asyncio
import json
from app.agent import agent_executor

async def test_mcp_vs_function_calls():
    """Test to demonstrate difference between current MCPs and true MCP protocol."""
    
    print("\n" + "=" * 60)
    print("🔬 MCP vs Function Calls Comparison")
    print("=" * 60)
    
    domain = "google.com"
    
    # Test current "MCP" implementation (actually enhanced function calls)
    print(f"\n📋 Current Implementation (Enhanced Function Calls):")
    try:
        from app.mcps import get_domain_whois
        current_result = get_domain_whois(domain)
        
        print(f"✅ Domain WHOIS for {domain}:")
        print(f"  • Type: {current_result.get('mcp_type', 'Not specified')}")
        print(f"  • Has timestamp: {'verification_timestamp' in current_result}")
        print(f"  • Error handling: {'error' in current_result if 'error' in current_result else 'Success'}")
        print(f"  • Structured data: {len(current_result)} fields")
        
        # Show what's missing for true MCP compliance
        missing_mcp_features = []
        if 'chain_of_custody' not in current_result:
            missing_mcp_features.append("Chain of custody tracking")
        if 'confidence_score' not in current_result:
            missing_mcp_features.append("Confidence scoring")
        if 'verification_method' not in current_result:
            missing_mcp_features.append("Verification method metadata")
        if 'integrity_hash' not in current_result:
            missing_mcp_features.append("Data integrity hash")
        if 'provenance' not in current_result:
            missing_mcp_features.append("Data provenance tracking")
            
        print(f"\n⚠️  Missing MCP Features ({len(missing_mcp_features)}):")
        for feature in missing_mcp_features:
            print(f"  • {feature}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demonstrate what true MCP would look like
    print(f"\n🚀 True MCP Implementation (Conceptual):")
    mock_mcp_result = {
        "mcp_version": "1.0",
        "tool": "domain_whois", 
        "input": {"domain": domain},
        "execution": {
            "timestamp": "2024-06-26T10:30:00Z",
            "method": "authoritative_whois",
            "verification_level": "high",
            "execution_id": "mcp_exec_abc123"
        },
        "output": {
            "status": "success",
            "data": {"registrar": "MarkMonitor Inc.", "created": "1997-09-15"},
            "confidence": 0.95,
            "metadata": {
                "source_authority": "ICANN",
                "real_time": True,
                "verified": True
            }
        },
        "provenance": {
            "chain_of_custody": ["analyst_user", "mcp_verifier", "whois_authority"],
            "integrity_hash": "sha256:def456...",
            "reproducible": True,
            "audit_trail": ["whois_query_logged", "response_validated"]
        }
    }
    
    print(f"✨ Enhanced MCP Features:")
    print(f"  • MCP Version: {mock_mcp_result['mcp_version']}")
    print(f"  • Chain of Custody: {len(mock_mcp_result['provenance']['chain_of_custody'])} steps")
    print(f"  • Confidence Score: {mock_mcp_result['output']['confidence']}")
    print(f"  • Integrity Hash: {mock_mcp_result['provenance']['integrity_hash'][:20]}...")
    print(f"  • Audit Trail: {len(mock_mcp_result['provenance']['audit_trail'])} events")
    print(f"  • Reproducible: {mock_mcp_result['provenance']['reproducible']}")

async def test_enhanced_planner():
    """Test the enhanced planner node with various OSINT scenarios."""
    
    test_cases = [
        {
            "name": "Domain Investigation",
            "state": {
                "topic": "suspicious domain malware-example.com",
                "case_id": "TEST001",
                "model_id": "gpt-4o-mini",
                "temperature": 0.3,
                "long_term_memory": [],
                "search_queries": [],
                "search_results": [],
                "synthesized_findings": "",
                "num_steps": 0,
                "mcp_verification_list": [],
                "verified_data": {},
                "planner_reasoning": "",
                "synthesis_confidence": "",
                "information_gaps": []
            }
        },
        {
            "name": "Social Media Investigation", 
            "state": {
                "topic": "Twitter account @suspicious_user linked to misinformation campaign",
                "case_id": "TEST002",
                "model_id": "claude-3-sonnet-20240229",
                "temperature": 0.2,
                "long_term_memory": [
                    {
                        "finding": "Account created in 2023",
                        "confidence": "High",
                        "source": "Twitter API"
                    }
                ],
                "search_queries": [],
                "search_results": [],
                "synthesized_findings": "",
                "num_steps": 0,
                "mcp_verification_list": [],
                "verified_data": {},
                "planner_reasoning": "",
                "synthesis_confidence": "",
                "information_gaps": []
            }
        },
        {
            "name": "Corporate Investigation",
            "state": {
                "topic": "TechCorp Inc possible data breach incident",
                "case_id": "TEST003", 
                "model_id": "gpt-4-turbo",
                "temperature": 0.1,
                "long_term_memory": [
                    {
                        "finding": "Company filed SEC report mentioning security incident",
                        "confidence": "High", 
                        "source": "SEC EDGAR database"
                    },
                    {
                        "finding": "LinkedIn shows recent CISO departure",
                        "confidence": "Medium",
                        "source": "LinkedIn profiles"
                    }
                ],
                "search_queries": [],
                "search_results": [],
                "synthesized_findings": "",
                "num_steps": 0,
                "mcp_verification_list": [],
                "verified_data": {},
                "planner_reasoning": "",
                "synthesis_confidence": "",
                "information_gaps": []
            }
        }
    ]
    
    print("🔍 Testing Enhanced OSINT Agent Planner")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Topic: {test_case['state']['topic']}")
        print(f"Model: {test_case['state']['model_id']}")
        print(f"Memory entries: {len(test_case['state']['long_term_memory'])}")
        
        try:
            # Run just the planner node
            from app.agent import planner_node
            result = await planner_node(test_case['state'])
            
            print(f"\n✅ Results:")
            print(f"  • Queries generated: {len(result['search_queries'])}")
            print(f"  • Has reasoning: {'Yes' if result.get('planner_reasoning') else 'No'}")
            
            print(f"\n🎯 Generated Queries:")
            for j, query in enumerate(result['search_queries'], 1):
                words = len(query.split())
                print(f"  {j}. {query} ({words} words)")
            
            if result.get('planner_reasoning'):
                print(f"\n🧠 Strategic Reasoning:")
                print(f"  {result['planner_reasoning'][:200]}...")
            
            # Validate query quality
            quality_issues = []
            for query in result['search_queries']:
                words = len(query.split())
                if words < 5:
                    quality_issues.append(f"Query too short: '{query}' ({words} words)")
                elif words > 15:
                    quality_issues.append(f"Query too long: '{query}' ({words} words)")
                elif not any(char.isalnum() for char in query):
                    quality_issues.append(f"Query lacks content: '{query}'")
            
            if quality_issues:
                print(f"\n⚠️  Quality Issues:")
                for issue in quality_issues:
                    print(f"  • {issue}")
            else:
                print(f"\n✨ All queries meet quality standards!")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Testing completed!")

async def test_full_agent_workflow():
    """Test the complete enhanced agent workflow."""
    
    print("\n" + "=" * 60)
    print("🤖 Testing Complete Enhanced Agent Workflow")
    print("=" * 60)
    
    test_state = {
        "topic": "cybersecurity company FireEye acquisition by Symphony Technology Group",
        "case_id": "FULL_TEST_001",
        "model_id": "gpt-4o-mini",
        "temperature": 0.3,
        "long_term_memory": [
            {
                "finding": "FireEye was a prominent cybersecurity company",
                "confidence": "High",
                "source": "Previous research"
            }
        ],
        "search_queries": [],
        "search_results": [],
        "synthesized_findings": "",
        "num_steps": 0,
        "mcp_verification_list": [],
        "verified_data": {},
        "planner_reasoning": "",
        "synthesis_confidence": "",
        "information_gaps": []
    }
    
    try:
        print(f"🚀 Running complete agent workflow...")
        print(f"Topic: {test_state['topic']}")
        
        # This would require Langfuse tracking to be properly set up
        # For now, we'll just test the planner
        from app.agent import planner_node
        result = await planner_node(test_state)
        
        print(f"\n✅ Planner Results:")
        print(f"  • Generated {len(result['search_queries'])} queries")
        print(f"  • Strategic reasoning: {bool(result.get('planner_reasoning'))}")
        
        for i, query in enumerate(result['search_queries'], 1):
            print(f"  {i}. {query}")
            
    except Exception as e:
        print(f"❌ Error in full workflow test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_enhanced_planner())
    asyncio.run(test_full_agent_workflow())
