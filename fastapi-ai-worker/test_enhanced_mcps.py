#!/usr/bin/env python3
"""
Enhanced MCP Testing Script

Tests all the enhanced MCP (Mission Critical Protocol) tools with comprehensive
test cases covering various OSINT scenarios and input validation.

Usage:
    python test_enhanced_mcps.py
"""

import sys
import os
import time
from datetime import datetime

# Add the app directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import all MCP functions (note: some require optional dependencies)
try:
    from app.mcps import (
        get_domain_whois, check_ip_reputation, verify_email_breach,
        analyze_url_safety, get_ssl_certificate, check_social_media_account,
        analyze_file_hash, analyze_phone_number, investigate_crypto_address,
        analyze_dns_records, extract_image_metadata, analyze_network_port,
        check_paste_site_exposure
    )
    print("✅ All MCP modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Note: Some MCPs require additional dependencies (dnspython, phonenumbers, Pillow)")
    sys.exit(1)

class MCPTestSuite:
    """Comprehensive test suite for all MCP tools."""
    
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_test(self, test_name: str, mcp_function, test_input, expected_type=None):
        """Run a single MCP test and record results."""
        print(f"\n🔍 Testing {test_name}")
        print(f"   Input: {test_input}")
        
        start_time = time.time()
        try:
            result = mcp_function(test_input)
            execution_time = time.time() - start_time
            
            # Basic validation
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict, got {type(result)}")
            
            if "error" in result:
                print(f"   ⚠️  Error result: {result['error']}")
                status = "error"
            else:
                print(f"   ✅ Success - execution time: {execution_time:.3f}s")
                if expected_type and result.get("mcp_type") != expected_type:
                    print(f"   ⚠️  Unexpected MCP type: {result.get('mcp_type')} (expected {expected_type})")
                status = "success"
            
            self.results[test_name] = {
                "status": status,
                "result": result,
                "execution_time": execution_time,
                "input": test_input
            }
            
            if status == "success":
                self.passed_tests += 1
            else:
                self.failed_tests += 1
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Exception: {str(e)}")
            self.results[test_name] = {
                "status": "exception",
                "error": str(e),
                "execution_time": execution_time,
                "input": test_input
            }
            self.failed_tests += 1
        
        self.total_tests += 1
    
    def run_network_port_test(self, test_name: str, ip: str, port: int):
        """Special test method for network port analysis that takes two parameters."""
        print(f"\n🔍 Testing {test_name}")
        print(f"   Input: {ip}:{port}")
        
        start_time = time.time()
        try:
            result = analyze_network_port(ip, port)
            execution_time = time.time() - start_time
            
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict, got {type(result)}")
            
            if "error" in result:
                print(f"   ⚠️  Error result: {result['error']}")
                status = "error"
            else:
                print(f"   ✅ Success - execution time: {execution_time:.3f}s")
                status = "success"
            
            self.results[test_name] = {
                "status": status,
                "result": result,
                "execution_time": execution_time,
                "input": f"{ip}:{port}"
            }
            
            if status == "success":
                self.passed_tests += 1
            else:
                self.failed_tests += 1
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Exception: {str(e)}")
            self.results[test_name] = {
                "status": "exception",
                "error": str(e),
                "execution_time": execution_time,
                "input": f"{ip}:{port}"
            }
            self.failed_tests += 1
        
        self.total_tests += 1
    
    def run_all_tests(self):
        """Run comprehensive tests for all MCP tools."""
        print("🧪 Starting Enhanced MCP Test Suite")
        print("=" * 60)
        
        # 1. Domain WHOIS Tests
        print("\n📋 DOMAIN WHOIS TESTS")
        self.run_test("WHOIS - Google", get_domain_whois, "google.com", "domain_whois")
        self.run_test("WHOIS - GitHub", get_domain_whois, "github.com", "domain_whois")
        self.run_test("WHOIS - Invalid Domain", get_domain_whois, "thisdomainshouldnotexist12345.com", "domain_whois")
        
        # 2. IP Reputation Tests
        print("\n🌐 IP REPUTATION TESTS")
        self.run_test("IP Rep - Google DNS", check_ip_reputation, "8.8.8.8", "ip_reputation")
        self.run_test("IP Rep - Cloudflare DNS", check_ip_reputation, "1.1.1.1", "ip_reputation")
        self.run_test("IP Rep - Private IP", check_ip_reputation, "192.168.1.1", "ip_reputation")
        self.run_test("IP Rep - Invalid IP", check_ip_reputation, "999.999.999.999", "ip_reputation")
        
        # 3. Email Breach Tests
        print("\n📧 EMAIL BREACH TESTS")
        self.run_test("Email - Yahoo (known breached)", verify_email_breach, "test@yahoo.com", "email_breach")
        self.run_test("Email - Gmail", verify_email_breach, "test@gmail.com", "email_breach")
        self.run_test("Email - Invalid format", verify_email_breach, "invalid-email", "email_breach")
        
        # 4. URL Safety Tests
        print("\n🔗 URL SAFETY TESTS")
        self.run_test("URL - Google HTTPS", analyze_url_safety, "https://www.google.com", "url_safety")
        self.run_test("URL - GitHub", analyze_url_safety, "https://github.com", "url_safety")
        self.run_test("URL - Suspicious patterns", analyze_url_safety, "http://192.168.1.1/phishing", "url_safety")
        
        # 5. SSL Certificate Tests
        print("\n🔒 SSL CERTIFICATE TESTS")
        self.run_test("SSL - Google", get_ssl_certificate, "google.com", "ssl_certificate")
        self.run_test("SSL - GitHub", get_ssl_certificate, "github.com", "ssl_certificate")
        self.run_test("SSL - Non-existent domain", get_ssl_certificate, "thisdomainshouldnotexist12345.com", "ssl_certificate")
        
        # 6. Social Media Account Tests
        print("\n👤 SOCIAL MEDIA TESTS")
        self.run_test("Social - @testuser", check_social_media_account, "@testuser", "social_media_account")
        self.run_test("Social - suspicious123", check_social_media_account, "suspicious123", "social_media_account")
        self.run_test("Social - bot_account_fake", check_social_media_account, "bot_account_fake", "social_media_account")
        
        # 7. File Hash Tests
        print("\n🔍 FILE HASH TESTS")
        self.run_test("Hash - MD5", analyze_file_hash, "5d41402abc4b2a76b9719d911017c592", "file_hash")
        self.run_test("Hash - SHA1", analyze_file_hash, "356a192b7913b04c54574d18c28d46e6395428ab", "file_hash")
        self.run_test("Hash - SHA256", analyze_file_hash, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "file_hash")
        self.run_test("Hash - Invalid", analyze_file_hash, "invalid-hash", "file_hash")
        
        # 8. Phone Number Tests (may require phonenumbers library)
        print("\n📞 PHONE NUMBER TESTS")
        try:
            self.run_test("Phone - US number", analyze_phone_number, "+1-555-123-4567", "phone_number")
            self.run_test("Phone - UK number", analyze_phone_number, "+44-20-7946-0958", "phone_number")
            self.run_test("Phone - Invalid format", analyze_phone_number, "invalid-phone", "phone_number")
        except Exception as e:
            print(f"⚠️  Phone number tests skipped: {e}")
        
        # 9. Cryptocurrency Address Tests
        print("\n💰 CRYPTOCURRENCY TESTS")
        self.run_test("Crypto - Bitcoin address", investigate_crypto_address, "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "crypto_address")
        self.run_test("Crypto - Ethereum address", investigate_crypto_address, "0x742d35Cc6634C0532925a3b8D9C9c2c8C2A5c3b9", "crypto_address")
        self.run_test("Crypto - Invalid address", investigate_crypto_address, "invalid-crypto-address", "crypto_address")
        
        # 10. DNS Records Tests (may require dnspython library)
        print("\n🌍 DNS RECORDS TESTS")
        try:
            self.run_test("DNS - Google.com", analyze_dns_records, "google.com", "dns_records")
            self.run_test("DNS - GitHub.com", analyze_dns_records, "github.com", "dns_records")
            self.run_test("DNS - Non-existent domain", analyze_dns_records, "thisdomainshouldnotexist12345.com", "dns_records")
        except Exception as e:
            print(f"⚠️  DNS tests skipped: {e}")
        
        # 11. Image Metadata Tests (requires Pillow)
        print("\n🖼️  IMAGE METADATA TESTS")
        try:
            # Simple 1x1 pixel PNG in base64
            test_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            self.run_test("Image - Valid PNG", extract_image_metadata, test_image, "image_metadata")
            self.run_test("Image - Invalid data", extract_image_metadata, "invalid-base64-data", "image_metadata")
        except Exception as e:
            print(f"⚠️  Image metadata tests skipped: {e}")
        
        # 12. Network Port Tests
        print("\n🔌 NETWORK PORT TESTS")
        self.run_network_port_test("Port - Google HTTP", "8.8.8.8", 80)
        self.run_network_port_test("Port - Google HTTPS", "8.8.8.8", 443)
        self.run_network_port_test("Port - Unlikely port", "8.8.8.8", 12345)
        
        # 13. Paste Site Exposure Tests
        print("\n📄 PASTE EXPOSURE TESTS")
        self.run_test("Paste - Email search", check_paste_site_exposure, "test@example.com", "paste_exposure")
        self.run_test("Paste - Phone search", check_paste_site_exposure, "555-123-4567", "paste_exposure")
        self.run_test("Paste - API key pattern", check_paste_site_exposure, "api_key_12345", "paste_exposure")
    
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        # Show failed tests
        failed_tests = [name for name, result in self.results.items() 
                       if result["status"] in ["error", "exception"]]
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test_name in failed_tests:
                result = self.results[test_name]
                print(f"   • {test_name}: {result.get('error', 'Unknown error')}")
        
        # Show performance metrics
        successful_results = [r for r in self.results.values() if r["status"] == "success"]
        if successful_results:
            avg_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
            max_time = max(r["execution_time"] for r in successful_results)
            min_time = min(r["execution_time"] for r in successful_results)
            
            print("\n⏱️  Performance Metrics:")
            print(f"   Average execution time: {avg_time:.3f}s")
            print(f"   Fastest test: {min_time:.3f}s")
            print(f"   Slowest test: {max_time:.3f}s")
        
        print(f"\n📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main test runner."""
    print("🚀 Enhanced OSINT MCP Tools - Comprehensive Test Suite")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    if not os.path.exists("app/mcps.py"):
        print("❌ Error: Please run this script from the fastapi-ai-worker directory")
        print("   Current directory:", os.getcwd())
        return False
    
    test_suite = MCPTestSuite()
    
    try:
        test_suite.run_all_tests()
        test_suite.print_summary()
        
        # Return success status based on results
        return test_suite.failed_tests == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
