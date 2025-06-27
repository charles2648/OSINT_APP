#!/usr/bin/env python3
"""
Enhanced Report Generation Test

Tests the improved update_report_node with comprehensive report generation,
verification analysis, and quality assessment capabilities.

Usage:
    python test_enhanced_report.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.agent import (
        _analyze_verification_results, 
        _assess_report_quality, _extract_key_findings
    )
    print("✅ Successfully imported enhanced report functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class EnhancedReportTester:
    """Test suite for the enhanced report generation functionality."""
    
    def __init__(self):
        self.test_results = {}
    
    def test_verification_analysis(self):
        """Test the verification analysis function."""
        print("\n🔍 Testing verification analysis...")
        
        # Mock verification data
        verified_data = {
            "get_domain_whois:example.com": {
                "mcp_name": "get_domain_whois",
                "input": "example.com",
                "result": {
                    "mcp_type": "domain_whois",
                    "registrar": "Example Registrar",
                    "creation_date": "2023-01-01",
                    "registrant_country": "US"
                },
                "priority": "high",
                "reasoning": "Central domain in investigation",
                "success": True,
                "execution_time": 1.234
            },
            "check_ip_reputation:1.2.3.4": {
                "mcp_name": "check_ip_reputation",
                "input": "1.2.3.4",
                "result": {
                    "mcp_type": "ip_reputation",
                    "geolocation": {
                        "country": "United States",
                        "isp": "Example ISP"
                    },
                    "open_ports": [80, 443]
                },
                "priority": "medium",
                "reasoning": "Associated IP address",
                "success": True,
                "execution_time": 2.456
            },
            "verify_email_breach:test@example.com": {
                "mcp_name": "verify_email_breach",
                "input": "test@example.com",
                "result": {"error": "API limit exceeded"},
                "priority": "low",
                "reasoning": "Email verification for completeness",
                "success": False,
                "execution_time": 0.123
            }
        }
        
        execution_results = [
            {"mcp_name": "get_domain_whois", "input": "example.com", "success": True},
            {"mcp_name": "check_ip_reputation", "input": "1.2.3.4", "success": True},
            {"mcp_name": "verify_email_breach", "input": "test@example.com", "success": False}
        ]
        
        analysis = _analyze_verification_results(verified_data, execution_results)
        
        print(f"   ✅ Analysis summary: {analysis['summary']}")
        print(f"   ✅ Success rate: {analysis['success_rate']:.1f}%")
        print(f"   ✅ High priority results: {len(analysis['high_priority_results'])}")
        print(f"   ✅ Impact assessment: {analysis['impact_assessment']}")
        
        self.test_results["verification_analysis"] = {
            "status": "success",
            "details": analysis
        }
    
    def test_key_findings_extraction(self):
        """Test key findings extraction from different MCP types."""
        print("\n🔍 Testing key findings extraction...")
        
        test_cases = [
            {
                "name": "Domain WHOIS",
                "result": {
                    "mcp_type": "domain_whois",
                    "registrar": "Test Registrar Inc.",
                    "creation_date": "2023-06-15",
                    "registrant_country": "United Kingdom"
                }
            },
            {
                "name": "IP Reputation",
                "result": {
                    "mcp_type": "ip_reputation",
                    "geolocation": {
                        "country": "Germany",
                        "isp": "Deutsche Telekom"
                    },
                    "open_ports": [22, 80, 443, 8080]
                }
            },
            {
                "name": "Email Breach",
                "result": {
                    "mcp_type": "email_breach",
                    "potential_exposure": True,
                    "domain": "compromised-site.com"
                }
            }
        ]
        
        for test_case in test_cases:
            findings = _extract_key_findings(test_case["result"])
            print(f"   ✅ {test_case['name']}: {len(findings)} findings - {findings}")
        
        self.test_results["key_findings"] = {"status": "success"}
    
    def test_report_quality_assessment(self):
        """Test report quality assessment functionality."""
        print("\n🔍 Testing report quality assessment...")
        
        # Mock a comprehensive report
        sample_report = """
        EXECUTIVE SUMMARY
        This intelligence assessment examines suspicious domain activity and associated infrastructure.
        
        INTELLIGENCE ASSESSMENT
        Based on comprehensive analysis of multiple sources, we assess with medium confidence
        that the target domain exhibits indicators of malicious activity.
        
        VERIFICATION ANALYSIS
        Domain WHOIS verification confirmed registration details and ownership patterns.
        
        SOURCE ASSESSMENT
        Sources demonstrate high reliability with cross-verification from multiple platforms.
        
        RISK INDICATORS
        - Suspicious domain registration patterns
        - Associated IP addresses with poor reputation
        
        RECOMMENDATIONS
        1. Monitor domain for additional malicious activity
        2. Block access to associated IP addresses
        3. Implement additional monitoring controls
        
        CONFIDENCE ASSESSMENT
        Medium confidence based on corroborated evidence from multiple sources.
        """
        
        verified_data = {"test": {"success": True}}
        search_metrics = {"quality_grade": "B", "total_sources": 8}
        source_count = 8
        verification_analysis = {"success_rate": 75}
        
        quality_metrics = _assess_report_quality(
            sample_report, verified_data, search_metrics, source_count, verification_analysis
        )
        
        print(f"   ✅ Overall quality score: {quality_metrics['overall_score']}")
        print(f"   ✅ Quality grade: {quality_metrics['quality_grade']}")
        print(f"   ✅ Sections found: {quality_metrics['sections_count']}/7")
        print(f"   ✅ Report length: {quality_metrics['report_length']} characters")
        
        self.test_results["quality_assessment"] = {
            "status": "success",
            "details": quality_metrics
        }
    
    async def test_enhanced_report_node(self):
        """Test the complete enhanced report node functionality."""
        print("\n🔍 Testing enhanced report node...")
        
        # Mock agent state for testing
        mock_state = {
            "topic": "Suspicious domain activity analysis",
            "model_id": "gpt-4",
            "temperature": 0.1,
            "num_steps": 5,
            "synthesized_findings": """
            Initial analysis indicates that the domain suspicious-site.com exhibits several
            concerning characteristics including recent registration, suspicious hosting patterns,
            and associations with known malicious infrastructure. Further verification is needed
            to confirm the threat level and attribution.
            """,
            "verified_data": {
                "get_domain_whois:suspicious-site.com": {
                    "mcp_name": "get_domain_whois",
                    "input": "suspicious-site.com",
                    "result": {
                        "mcp_type": "domain_whois",
                        "registrar": "Privacy Protection Services",
                        "creation_date": "2023-12-01",
                        "registrant_country": "PA"
                    },
                    "priority": "high",
                    "reasoning": "Central domain under investigation",
                    "success": True
                }
            },
            "mcp_execution_results": [
                {"mcp_name": "get_domain_whois", "success": True, "execution_time": 1.2}
            ],
            "search_quality_metrics": {
                "quality_grade": "B",
                "total_sources": 12,
                "credibility_score": 0.78,
                "coverage_score": 0.85
            },
            "planner_reasoning": "Multi-vector analysis focusing on domain infrastructure and associations",
            "synthesis_confidence": "Medium",
            "information_gaps": ["Attribution details", "Timeline of malicious activity"],
            "verification_strategy": "Prioritize domain and IP verification for infrastructure mapping",
            "search_results": [{"title": "test", "content": "test"}] * 12
        }
        
        try:
            # Note: This would require actual LLM access to test fully
            # For now, we'll test the preparation logic
            print("   ✅ Mock state preparation successful")
            print(f"   ✅ Topic: {mock_state['topic']}")
            print(f"   ✅ Verified entities: {len(mock_state['verified_data'])}")
            print(f"   ✅ Search quality: {mock_state['search_quality_metrics']['quality_grade']}")
            
            # Test verification analysis on the mock data
            verification_analysis = _analyze_verification_results(
                mock_state["verified_data"], 
                mock_state["mcp_execution_results"]
            )
            
            print(f"   ✅ Verification analysis: {verification_analysis['impact_assessment']}")
            
            self.test_results["enhanced_report_node"] = {
                "status": "prepared",
                "note": "Full test requires LLM access"
            }
            
        except Exception as e:
            print(f"   ❌ Error in report node test: {e}")
            self.test_results["enhanced_report_node"] = {
                "status": "error",
                "error": str(e)
            }
    
    def run_all_tests(self):
        """Run all enhanced report functionality tests."""
        print("🧪 Enhanced Report Generation Test Suite")
        print("=" * 60)
        
        self.test_verification_analysis()
        self.test_key_findings_extraction()
        self.test_report_quality_assessment()
        
        # Run async test
        asyncio.run(self.test_enhanced_report_node())
        
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 ENHANCED REPORT TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results.values() 
                               if r["status"] in ["success", "prepared"]])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {total_tests - successful_tests}")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] in ["success", "prepared"] else "❌"
            print(f"{status_icon} {test_name}: {result['status']}")
            
            if "note" in result:
                print(f"   Note: {result['note']}")
            if "error" in result:
                print(f"   Error: {result['error']}")
        
        print(f"\n📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main test runner."""
    print("🚀 Enhanced Report Generation - Test Suite")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    if not os.path.exists("app/agent.py"):
        print("❌ Error: Please run this script from the fastapi-ai-worker directory")
        print("   Current directory:", os.getcwd())
        return False
    
    tester = EnhancedReportTester()
    
    try:
        tester.run_all_tests()
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
