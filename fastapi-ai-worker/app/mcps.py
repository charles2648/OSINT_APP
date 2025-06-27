# Mission-Critical Components (MCPs) - Deterministic, verifiable OSINT tools.

import whois  # type: ignore
import ssl # type: ignore
import socket
import hashlib
import re
import time
import requests # type: ignore
import dns.resolver
import phonenumbers
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import urlparse
from PIL import Image
from PIL.ExifTags import TAGS
import base64
import io

def get_domain_whois(domain: str) -> Dict[str, Any]:
    """
    Performs a WHOIS lookup on a domain and returns structured, factual data.
    """
    print(f"MCP: Performing WHOIS lookup for domain: {domain}")
    try:
        w = whois.whois(domain)
        if not w.domain_name:
            return {"error": f"Could not retrieve WHOIS data for {domain}. It may not be registered."}
        
        def format_date(d):
            if isinstance(d, list): 
                return [dt.isoformat() if isinstance(dt, datetime) else str(dt) for dt in d]
            if isinstance(d, datetime): 
                return d.isoformat()
            return str(d) if d else None
        
        return {
            "mcp_type": "domain_whois",
            "domain_name": w.domain_name,
            "registrar": w.registrar,
            "creation_date": format_date(w.creation_date),
            "expiration_date": format_date(w.expiration_date),
            "last_updated": format_date(w.updated_date),
            "name_servers": w.name_servers,
            "status": w.status,
            "emails": w.emails,
            "registrant_country": getattr(w, 'country', None),
            "registrant_org": getattr(w, 'org', None),
            "dnssec": getattr(w, 'dnssec', None),
            "verification_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"MCP Error: WHOIS lookup for {domain} failed: {e}")
        return {"error": f"WHOIS lookup failed: {e}", "mcp_type": "domain_whois"}

def check_ip_reputation(ip_address: str) -> Dict[str, Any]:
    """
    Check IP address reputation and gather intelligence data.
    """
    print(f"MCP: Checking IP reputation for: {ip_address}")
    try:
        result: Dict[str, Any] = {
            "mcp_type": "ip_reputation",
            "ip_address": ip_address,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Basic IP validation
        import ipaddress
        ip_obj = ipaddress.ip_address(ip_address)
        result["ip_version"] = str(ip_obj.version)
        result["is_private"] = str(ip_obj.is_private)
        result["is_reserved"] = str(ip_obj.is_reserved)
        result["is_multicast"] = str(ip_obj.is_multicast)
        
        # Reverse DNS lookup
        try:
            hostname = socket.gethostbyaddr(ip_address)[0]
            result["reverse_dns"] = hostname
        except socket.herror:
            result["reverse_dns"] = ""
        
        # Geolocation (using public IP geolocation service)
        try:
            # Using a free service for demo - in production, use commercial services
            geo_response = requests.get(f"http://ip-api.com/json/{ip_address}", timeout=10)
            if geo_response.status_code == 200:
                geo_data = geo_response.json()
                if geo_data.get('status') == 'success':
                    geolocation_info: Dict[str, Any] = {
                        "country": geo_data.get('country'),
                        "country_code": geo_data.get('countryCode'),
                        "region": geo_data.get('regionName'),
                        "city": geo_data.get('city'),
                        "isp": geo_data.get('isp'),
                        "org": geo_data.get('org'),
                        "as": geo_data.get('as'),
                        "timezone": geo_data.get('timezone')
                    }
                    result["geolocation"] = geolocation_info
                else:
                    result["geolocation"] = None
            else:
                result["geolocation"] = None
        except Exception as e:
            result["geolocation_error"] = str(e)
            result["geolocation"] = None
        
        # Port scan common ports (basic reconnaissance)
        common_ports = [22, 25, 53, 80, 443, 993, 995]
        open_ports = []
        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            try:
                result_code = sock.connect_ex((ip_address, port))
                if result_code == 0:
                    open_ports.append(port)
            except Exception:
                pass
            finally:
                sock.close()
        
        result["open_ports"] = open_ports
        result["scan_timestamp"] = datetime.now().isoformat()
        
        return result
        
    except Exception as e:
        print(f"MCP Error: IP reputation check for {ip_address} failed: {e}")
        return {"error": f"IP reputation check failed: {e}", "mcp_type": "ip_reputation"}

def verify_email_breach(email: str) -> Dict[str, Any]:
    """
    Check if email address appears in known data breaches.
    Note: This is a mock implementation. In production, integrate with HIBP API or similar.
    """
    print(f"MCP: Checking email breach status for: {email}")
    try:
        result = {
            "mcp_type": "email_breach",
            "email": email,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Email format validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return {"error": "Invalid email format", "mcp_type": "email_breach"}
        
        # Extract domain for analysis
        domain = email.split('@')[1].lower()
        result["domain"] = domain
        
        # Mock breach data (in production, use HIBP API or commercial services)
        # This would be replaced with actual API calls
        known_breached_domains = [
            'yahoo.com', 'linkedin.com', 'adobe.com', 'dropbox.com', 
            'twitter.com', 'facebook.com', 'myspace.com'
        ]
        
        if domain in known_breached_domains:
            result["potential_exposure"] = str(True)
            result["breach_indicators"] = str([
                {
                    "service": domain,
                    "type": "domain_breach_history",
                    "confidence": "medium",
                    "note": "Domain has history of data breaches"
                }
            ])
        else:
            result["potential_exposure"] = str(False)
            result["breach_indicators"] = str([])
        
        # Email pattern analysis
        if '+' in email.split('@')[0]:
            result["email_aliasing"] = str(True)
        if '.' in email.split('@')[0]:
            result["contains_dots"] = str(True)
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Email breach check for {email} failed: {e}")
        return {"error": f"Email breach check failed: {e}", "mcp_type": "email_breach"}

def analyze_url_safety(url: str) -> Dict[str, Any]:
    """
    Analyze URL for safety and security indicators.
    """
    print(f"MCP: Analyzing URL safety for: {url}")
    try:
        result = {
            "mcp_type": "url_safety",
            "url": url,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Parse URL components
        parsed = urlparse(url)
        result["scheme"] = parsed.scheme
        result["domain"] = parsed.netloc
        result["path"] = parsed.path
        result["query"] = parsed.query
        
        # Security indicators
        result["uses_https"] = str(parsed.scheme == 'https')
        result["has_query_params"] = str(bool(parsed.query))
        
        # Suspicious patterns
        suspicious_patterns = [
            r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP address
            r'bit\.ly|tinyurl|goo\.gl|t\.co',  # URL shorteners
            r'[a-z0-9]{20,}',  # Long random strings
            r'phishing|malware|virus|trojan',  # Obvious threats
        ]
        
        suspicious_indicators = []
        for pattern in suspicious_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                suspicious_indicators.append(pattern)
        
        result["suspicious_patterns"] = str(suspicious_indicators)
        result["risk_score"] = str(len(suspicious_indicators) / len(suspicious_patterns))
        
        # Domain reputation check
        domain = parsed.netloc.lower()
        trusted_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
            'github.com', 'stackoverflow.com', 'wikipedia.org'
        ]
        
        result["is_trusted_domain"] = str(any(trusted in domain for trusted in trusted_domains))
        
        # Try to fetch headers (with caution)
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            result["http_status"] = response.status_code
            result["server"] = response.headers.get('Server', 'Unknown')
            result["content_type"] = response.headers.get('Content-Type', 'Unknown')
            result["redirects"] = str(len(response.history))
        except Exception as e:
            result["connection_error"] = str(e)
        
        return result
        
    except Exception as e:
        print(f"MCP Error: URL safety analysis for {url} failed: {e}")
        return {"error": f"URL safety analysis failed: {e}", "mcp_type": "url_safety"}

def get_ssl_certificate(domain: str) -> Dict[str, Any]:
    """
    Extract and analyze SSL certificate information for a domain.
    """
    print(f"MCP: Analyzing SSL certificate for: {domain}")
    try:
        result = {
            "mcp_type": "ssl_certificate",
            "domain": domain,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Get SSL certificate
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                cert_der = ssock.getpeercert(binary_form=True)
                
                if not cert or not cert_der:
                    return {"error": "Could not retrieve SSL certificate", "mcp_type": "ssl_certificate"}
                
                # Certificate details - properly handle cert structure
                try:
                    subject_dict = dict(x[0] for x in cert['subject'] if isinstance(x, (list, tuple)) and len(x) >= 1 and isinstance(x[0], (list, tuple)) and len(x[0]) >= 2)
                    issuer_dict = dict(x[0] for x in cert['issuer'] if isinstance(x, (list, tuple)) and len(x) >= 1 and isinstance(x[0], (list, tuple)) and len(x[0]) >= 2)
                    
                    result["subject"] = str(subject_dict)
                    result["issuer"] = str(issuer_dict)
                    result["version"] = str(cert.get('version', ''))
                    result["serial_number"] = str(cert.get('serialNumber', ''))
                    result["not_before"] = str(cert.get('notBefore', ''))
                    result["not_after"] = str(cert.get('notAfter', ''))
                    
                    # Subject Alternative Names
                    if 'subjectAltName' in cert:
                        result["subject_alt_names"] = str([str(name[1]) for name in cert['subjectAltName'] if len(name) >= 2])
                    else:
                        result["subject_alt_names"] = str([])
                    
                    # Certificate fingerprints
                    result["sha1_fingerprint"] = hashlib.sha1(cert_der).hexdigest()
                    result["sha256_fingerprint"] = hashlib.sha256(cert_der).hexdigest()
                    
                    # Validity check
                    not_after_str = cert.get('notAfter')
                    not_before_str = cert.get('notBefore')
                    
                    if not_after_str and not_before_str:
                        try:
                            not_after = datetime.strptime(str(not_after_str), '%b %d %H:%M:%S %Y %Z')
                            not_before = datetime.strptime(str(not_before_str), '%b %d %H:%M:%S %Y %Z')
                            now = datetime.now()
                            
                            result["is_valid"] = str(not_before <= now <= not_after)
                            result["days_until_expiry"] = str((not_after - now).days)
                            result["is_expired"] = str(now > not_after)
                            result["is_self_signed"] = str(subject_dict == issuer_dict)
                        except ValueError:
                            result["is_valid"] = str(False)
                            result["days_until_expiry"] = str(0)
                            result["is_expired"] = str(True)
                            result["is_self_signed"] = str(False)
                    else:
                        result["is_valid"] = str(False)
                        result["days_until_expiry"] = str(0)
                        result["is_expired"] = str(True)
                        result["is_self_signed"] = str(False)
                    
                    # Certificate authority info
                    result["ca_issuer"] = issuer_dict.get('organizationName', 'Unknown')
                    
                except Exception as cert_parse_error:
                    result["error"] = f"Certificate parsing error: {str(cert_parse_error)}"
                
        return result
        
    except Exception as e:
        print(f"MCP Error: SSL certificate analysis for {domain} failed: {e}")
        return {"error": f"SSL certificate analysis failed: {e}", "mcp_type": "ssl_certificate"}

def check_social_media_account(handle: str) -> Dict[str, Any]:
    """
    Check social media account information and patterns.
    Note: This is a mock implementation for common patterns.
    """
    print(f"MCP: Checking social media account: {handle}")
    try:
        result = {
            "mcp_type": "social_media_account",
            "handle": handle,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Clean handle
        clean_handle = handle.lstrip('@').lower()
        result["clean_handle"] = clean_handle
        
        # Pattern analysis
        patterns = {
            "has_numbers": bool(re.search(r'\d', clean_handle)),
            "has_underscores": '_' in clean_handle,
            "length": len(clean_handle),
            "all_lowercase": clean_handle.islower(),
            "random_pattern": bool(re.search(r'[0-9]{3,}|[a-z]{10,}', clean_handle)),
        }
        result["patterns"] = str(patterns)
        
        # Suspicious indicators
        suspicious_words = ['fake', 'bot', 'spam', 'temp', 'test', 'anonymous']
        result["suspicious_keywords"] = str([word for word in suspicious_words if word in clean_handle])
        
        # Common platforms check (mock data)
        platforms = ['twitter', 'instagram', 'facebook', 'linkedin', 'github']
        result["potential_platforms"] = str(platforms)  # In production, actually check availability
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Social media account check for {handle} failed: {e}")
        return {"error": f"Social media account check failed: {e}", "mcp_type": "social_media_account"}

def analyze_file_hash(file_hash: str) -> Dict[str, Any]:
    """
    Analyze file hash for malware and threat intelligence.
    Note: This is a mock implementation. In production, integrate with VirusTotal API.
    """
    print(f"MCP: Analyzing file hash: {file_hash}")
    try:
        result = {
            "mcp_type": "file_hash",
            "hash": file_hash,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Determine hash type
        hash_length = len(file_hash)
        if hash_length == 32:
            result["hash_type"] = "MD5"
        elif hash_length == 40:
            result["hash_type"] = "SHA1"
        elif hash_length == 64:
            result["hash_type"] = "SHA256"
        else:
            return {"error": "Invalid hash format", "mcp_type": "file_hash"}
        
        # Mock threat analysis (replace with VirusTotal or similar API)
        # This would be actual API calls in production
        known_malicious_hashes = [
            "5d41402abc4b2a76b9719d911017c592",  # Example MD5
            "356a192b7913b04c54574d18c28d46e6395428ab",  # Example SHA1
        ]
        
        if file_hash.lower() in known_malicious_hashes:
            result["threat_detected"] = str(True)
            result["threat_type"] = "known_malware"
            result["confidence"] = "high"
        else:
            result["threat_detected"] = str(False)
            result["threat_type"] = "none"
            result["confidence"] = "medium"
        
        # File type analysis based on hash patterns (mock)
        result["estimated_file_type"] = "executable" if result["threat_detected"] else "unknown"
        
        return result
        
    except Exception as e:
        print(f"MCP Error: File hash analysis for {file_hash} failed: {e}")
        return {"error": f"File hash analysis failed: {e}", "mcp_type": "file_hash"}

def analyze_phone_number(phone_number: str) -> Dict[str, Any]:
    """
    Analyze phone number for country, carrier, and validity information.
    """
    print(f"MCP: Analyzing phone number: {phone_number}")
    try:
        result = {
            "mcp_type": "phone_number",
            "phone_number": phone_number,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Parse the phone number
        try:
            parsed_number = phonenumbers.parse(phone_number, None)
        except phonenumbers.phonenumberutil.NumberParseException as e:
            return {"error": f"Invalid phone number format: {e}", "mcp_type": "phone_number"}
        
        # Basic validation
        result["is_valid"] = str(phonenumbers.is_valid_number(parsed_number))
        result["is_possible"] = str(phonenumbers.is_possible_number(parsed_number))
        
        # Geographic information
        result["country_code"] = str(parsed_number.country_code) if parsed_number.country_code else ""
        result["national_number"] = str(parsed_number.national_number) if parsed_number.national_number else ""
        
        # Get country information
        try:
            from phonenumbers import geocoder, carrier
            result["country"] = geocoder.country_name_for_number(parsed_number, "en") or ""
            result["region"] = geocoder.description_for_number(parsed_number, "en") or ""
            result["carrier"] = carrier.name_for_number(parsed_number, "en") or ""
        except Exception:
            result["country"] = ""
            result["region"] = ""
            result["carrier"] = ""
        
        # Number type analysis
        from phonenumbers import phonenumberutil
        number_type = phonenumberutil.number_type(parsed_number)
        type_mapping = {
            0: "FIXED_LINE",
            1: "MOBILE", 
            2: "FIXED_LINE_OR_MOBILE",
            3: "TOLL_FREE",
            4: "PREMIUM_RATE",
            5: "SHARED_COST",
            6: "VOIP",
            7: "PERSONAL_NUMBER",
            8: "PAGER",
            9: "UAN",
            10: "VOICEMAIL",
            99: "UNKNOWN"
        }
        result["number_type"] = type_mapping.get(number_type, "UNKNOWN")
        
        # Format in different styles
        result["international_format"] = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        result["national_format"] = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.NATIONAL)
        result["e164_format"] = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Phone number analysis for {phone_number} failed: {e}")
        return {"error": f"Phone number analysis failed: {e}", "mcp_type": "phone_number"}

def investigate_crypto_address(address: str) -> Dict[str, Any]:
    """
    Investigate cryptocurrency address for blockchain analysis.
    """
    print(f"MCP: Investigating crypto address: {address}")
    try:
        result = {
            "mcp_type": "crypto_address",
            "address": address,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Detect cryptocurrency type based on address format
        crypto_patterns = {
            "bitcoin": r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$|^bc1[a-z0-9]{39,59}$",
            "ethereum": r"^0x[a-fA-F0-9]{40}$",
            "litecoin": r"^[LM3][a-km-zA-HJ-NP-Z1-9]{26,33}$",
            "monero": r"^4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}$",
            "ripple": r"^r[0-9a-zA-Z]{24,34}$",
        }
        
        detected_type = None
        for crypto_type, pattern in crypto_patterns.items():
            if re.match(pattern, address):
                detected_type = crypto_type
                break
        
        result["cryptocurrency_type"] = detected_type or "unknown"
        result["address_format_valid"] = str(detected_type is not None)
        
        # Address characteristics
        result["address_length"] = str(len(address))
        result["has_mixed_case"] = str(address != address.lower() and address != address.upper())
        
        # Mock blockchain analysis (in production, use blockchain APIs)
        if detected_type:
            result["mock_analysis"] = str({
                "estimated_balance": "0.00000000",  # Would be real API call
                "transaction_count": 0,
                "first_seen": None,
                "last_activity": None,
                "risk_score": "low",
                "tags": ["new_address"],
                "note": "Mock data - integrate with blockchain API for real analysis"
            })
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Crypto address investigation for {address} failed: {e}")
        return {"error": f"Crypto address investigation failed: {e}", "mcp_type": "crypto_address"}

def analyze_dns_records(domain: str) -> Dict[str, Any]:
    """
    Comprehensive DNS record analysis for a domain.
    """
    print(f"MCP: Analyzing DNS records for: {domain}")
    try:
        result = {
            "mcp_type": "dns_records",
            "domain": domain,
            "verification_timestamp": datetime.now().isoformat(),
            "records": {}
        }
        
        # Initialize records as a mutable dictionary
        records_dict: Dict[str, List[str]] = {}
        result["records"] = records_dict
        
        # DNS record types to check
        record_types = ['A', 'AAAA', 'CNAME', 'MX', 'NS', 'TXT', 'SOA', 'PTR']
        for record_type in record_types:
            try:
                answers = dns.resolver.resolve(domain, record_type)
                record_list = []
                for answer in answers:
                    record_list.append(str(answer))
                records_dict[record_type] = record_list
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout):
                records_dict[record_type] = []
            except Exception as e:
                records_dict[record_type] = [f"Error: {str(e)}"]
        # DNS security analysis
        result["security_analysis"] = {
            "has_spf": any("v=spf1" in record for record in records_dict.get("TXT", [])),
            "has_dmarc": any("v=DMARC1" in record for record in records_dict.get("TXT", [])),
            "has_dkim": any("v=DKIM1" in record for record in records_dict.get("TXT", [])),
            "mx_count": len(records_dict.get("MX", [])),
            "ns_count": len(records_dict.get("NS", [])),
            "has_ipv6": len(records_dict.get("AAAA", [])) > 0
        }
        
        # Subdomain enumeration (common subdomains)
        common_subdomains = ['www', 'mail', 'ftp', 'admin', 'api', 'dev', 'test', 'staging']
        subdomain_dict: Dict[str, List[str]] = {}
        result["subdomain_check"] = subdomain_dict
        
        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"
            try:
                answers = dns.resolver.resolve(full_domain, 'A')
                subdomain_dict[subdomain] = [str(answer) for answer in answers]
            except Exception:
                subdomain_dict[subdomain] = []
        
        return result
        
    except Exception as e:
        print(f"MCP Error: DNS analysis for {domain} failed: {e}")
        return {"error": f"DNS analysis failed: {e}", "mcp_type": "dns_records"}

def extract_image_metadata(image_data: str) -> Dict[str, Any]:
    """
    Extract metadata from image data (base64 encoded).
    """
    print("MCP: Extracting image metadata")
    try:
        result = {
            "mcp_type": "image_metadata",
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Decode base64 image data
        try:
            # Remove data:image/...;base64, prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return {"error": f"Invalid image data: {e}", "mcp_type": "image_metadata"}
        
        # Basic image information
        result["image_info"] = str({
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height
        })
        
        # Extract EXIF data
        exif_data = {}
        try:
            exif = image.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8')
                        except Exception:
                            value = str(value)
                    exif_data[tag] = value
        except AttributeError:
            exif_data = {}
        
        result["exif_data"] = str(exif_data)
        
        # Privacy-sensitive metadata detection
        sensitive_tags = ['GPS', 'DateTime', 'Make', 'Model', 'Software', 'Artist', 'Copyright']
        privacy_concerns = {}
        
        for tag in sensitive_tags:
            found_tags = [key for key in exif_data.keys() if isinstance(key, str) and tag.lower() in key.lower()]
            if found_tags:
                privacy_concerns[tag] = found_tags
        
        result["privacy_concerns"] = str(privacy_concerns)
        
        # GPS coordinate extraction
        if 'GPSInfo' in exif_data:
            result["gps_warning"] = "GPS coordinates found in image metadata"
            result["has_location_data"] = str(True)
        else:
            result["has_location_data"] = str(False)
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Image metadata extraction failed: {e}")
        return {"error": f"Image metadata extraction failed: {e}", "mcp_type": "image_metadata"}

def analyze_network_port(ip_address: str, port: int) -> Dict[str, Any]:
    """
    Analyze specific network port for service detection and security assessment.
    """
    print(f"MCP: Analyzing port {port} on {ip_address}")
    try:
        result = {
            "mcp_type": "network_port",
            "ip_address": ip_address,
            "port": port,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Common port service mapping
        common_services = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 993: "IMAPS",
            995: "POP3S", 3389: "RDP", 5432: "PostgreSQL", 3306: "MySQL",
            1433: "MSSQL", 6379: "Redis", 27017: "MongoDB"
        }
        
        result["expected_service"] = common_services.get(port, "Unknown")
        
        # Port connectivity test
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        try:
            start_time = time.time()
            result_code = sock.connect_ex((ip_address, port))
            connection_time = time.time() - start_time
            
            if result_code == 0:
                result["status"] = "open"
                result["connection_time"] = connection_time
                
                # Try to grab banner
                try:
                    sock.send(b"HEAD / HTTP/1.0\r\n\r\n")
                    banner = sock.recv(1024).decode('utf-8', errors='ignore')
                    result["banner"] = banner[:500]  # Limit banner size
                except Exception:
                    result["banner"] = None
            else:
                result["status"] = "closed"
                result["connection_time"] = connection_time
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
        finally:
            sock.close()
        
        # Security assessment
        if result["status"] == "open":
            security_flags = []
            
            # Check for insecure services
            insecure_ports = [21, 23, 25, 53, 80, 110, 143]
            if port in insecure_ports:
                security_flags.append("unencrypted_service")
            
            # Check for common attack vectors
            if port == 22 and result.get("banner"):
                banner_obj = result.get("banner")
                if banner_obj and isinstance(banner_obj, str) and "OpenSSH" in banner_obj:
                    security_flags.append("ssh_service_detected")
            
            if port in [3389, 5900]:  # RDP, VNC
                security_flags.append("remote_desktop_service")
            
            result["security_flags"] = security_flags
            result["risk_level"] = "high" if len(security_flags) > 1 else "medium" if security_flags else "low"
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Port analysis for {ip_address}:{port} failed: {e}")
        return {"error": f"Port analysis failed: {e}", "mcp_type": "network_port"}

def check_paste_site_exposure(search_term: str) -> Dict[str, Any]:
    """
    Check for data exposure on paste sites (mock implementation).
    """
    print(f"MCP: Checking paste site exposure for: {search_term}")
    try:
        result = {
            "mcp_type": "paste_exposure",
            "search_term": search_term,
            "verification_timestamp": datetime.now().isoformat()
        }
        
        # Mock paste site analysis (in production, integrate with paste monitoring APIs)
        paste_sites = ['pastebin.com', 'paste.org', 'ghostbin.com', 'hastebin.com']
        
        result["sites_checked"] = str(paste_sites)
        result["potential_exposures"] = str([])
        
        # Pattern-based risk assessment
        risk_patterns = {
            "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "phone": r'[\+]?[1-9]?[0-9]{7,15}',
            "ip": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "password": r'password|pwd|pass',
            "api_key": r'api[_-]?key|token',
            "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b'
        }
        
        detected_patterns = []
        for pattern_name, pattern in risk_patterns.items():
            if re.search(pattern, search_term, re.IGNORECASE):
                detected_patterns.append(pattern_name)
        
        if detected_patterns:
            result["sensitive_data_detected"] = str(detected_patterns)
            result["risk_level"] = "high"
            result["recommendation"] = "Monitor for potential data exposure"
        else:
            result["sensitive_data_detected"] = str([])
            result["risk_level"] = "low"
        
        # Mock exposure results
        result["mock_note"] = "This is mock data. In production, integrate with paste monitoring services."
        
        return result
        
    except Exception as e:
        print(f"MCP Error: Paste site exposure check for {search_term} failed: {e}")
        return {"error": f"Paste site exposure check failed: {e}", "mcp_type": "paste_exposure"}