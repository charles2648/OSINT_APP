#!/usr/bin/env python3
"""
Comprehensive test suite for all MCP functions to verify they work without errors.
"""

from app.mcps import (
    get_domain_whois, check_ip_reputation, verify_email_breach, analyze_url_safety,
    get_ssl_certificate, check_social_media_account, analyze_file_hash,
    analyze_phone_number, investigate_crypto_address, analyze_dns_records,
    extract_image_metadata, analyze_network_port, check_paste_site_exposure
)
# import base64  # Not needed, PNG data is already base64 encoded

def test_all_mcps():
    """Test all MCP functions with sample inputs."""
    
    tests = [
        ("get_domain_whois", lambda: get_domain_whois("google.com")),
        ("check_ip_reputation", lambda: check_ip_reputation("8.8.8.8")),
        ("verify_email_breach", lambda: verify_email_breach("test@example.com")),
        ("analyze_url_safety", lambda: analyze_url_safety("https://www.google.com")),
        ("get_ssl_certificate", lambda: get_ssl_certificate("google.com")),
        ("check_social_media_account", lambda: check_social_media_account("@testuser")),
        ("analyze_file_hash", lambda: analyze_file_hash("5d41402abc4b2a76b9719d911017c592")),
        ("analyze_phone_number", lambda: analyze_phone_number("+1-555-123-4567")),
        ("investigate_crypto_address", lambda: investigate_crypto_address("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")),
        ("analyze_dns_records", lambda: analyze_dns_records("google.com")),
        ("analyze_network_port", lambda: analyze_network_port("8.8.8.8", 53)),
        ("check_paste_site_exposure", lambda: check_paste_site_exposure("test@example.com")),
    ]
    
    # Test image metadata with a simple base64 encoded 1x1 pixel PNG
    png_1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI/hRXzSAAAAABJRU5ErkJggg=="
    tests.append(("extract_image_metadata", lambda: extract_image_metadata(png_1x1)))
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            success = isinstance(result, dict) and "mcp_type" in result
            results[test_name] = {
                "success": success,
                "mcp_type": result.get("mcp_type") if success else None,
                "error": result.get("error") if "error" in result else None
            }
            print(f"✅ {test_name}: {'PASS' if success else 'FAIL'}")
            if not success:
                print(f"   Result: {result}")
        except Exception as e:
            results[test_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    # Summary
    total_tests = len(tests)
    passed_tests = sum(1 for r in results.values() if r["success"])
    print(f"\n📊 Test Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All MCP functions are working correctly!")
    else:
        print("⚠️  Some MCP functions need attention.")
        for test_name, result in results.items():
            if not result["success"]:
                print(f"   - {test_name}: {result['error']}")
    
    return results

if __name__ == "__main__":
    test_all_mcps()
