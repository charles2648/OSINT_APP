# Mission-Critical Components (MCPs) - Deterministic, verifiable tools.

import whois
from datetime import datetime

def get_domain_whois(domain: str) -> dict:
    """
    Performs a WHOIS lookup on a domain and returns structured, factual data.
    """
    print(f"MCP: Performing WHOIS lookup for domain: {domain}")
    try:
        w = whois.whois(domain)
        if not w.domain_name:
            return {"error": f"Could not retrieve WHOIS data for {domain}. It may not be registered."}
        def format_date(d):
            if isinstance(d, list): return [dt.isoformat() for dt in d]
            if isinstance(d, datetime): return d.isoformat()
            return d
        return {
            "domain_name": w.domain_name, "registrar": w.registrar,
            "creation_date": format_date(w.creation_date), "expiration_date": format_date(w.expiration_date),
            "last_updated": format_date(w.updated_date), "name_servers": w.name_servers,
            "status": w.status, "emails": w.emails,
        }
    except Exception as e:
        print(f"MCP Error: WHOIS lookup for {domain} failed: {e}")
        return {"error": f"An exception occurred during WHOIS lookup: {e}"}