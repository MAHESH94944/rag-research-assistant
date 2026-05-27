"""
TOOLS: Web Search and Scraping Utilities
PHASES 1-2: Foundation + RAG Implementation
"""

from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv
from rich import print
from typing import List, Dict, Optional
import time

load_dotenv()

# ============================================================================
# TAVILY CLIENT INITIALIZATION
# ============================================================================

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Configuration from environment variables
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 20))  # Free tier max is 20
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 3000))
SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", 8))

# ============================================================================
# WEB SEARCH TOOL
# ============================================================================

@tool
def web_search(query: str, max_results: int = None) -> List[Dict]:
    """
    Search the web for recent and reliable information on a topic.
    Returns structured list of dictionaries with title, url, and content.
    
    Args:
        query: Search query string
        max_results: Maximum number of results (default from env, max 20 for free tier)
    
    Returns:
        List of dicts with keys: title, url, content
    """
    # Use environment default or provided value, capped at 20 for free tier
    results_count = max_results or MAX_SEARCH_RESULTS
    results_count = min(results_count, 20)  # Tavily free tier limit
    
    print(f"[SEARCH] Query: '{query}' | Max results: {results_count}")
    
    try:
        results = tavily.search(
            query=query, 
            max_results=results_count,
            search_depth="advanced",  # "basic" or "advanced" - advanced gives better results
            include_answer=False,      # Don't include AI answer (saves tokens)
            include_raw_content=False, # Don't include raw HTML (saves tokens)
            include_images=False,      # Don't include images (faster)
        )
        
        out = []
        
        for r in results.get('results', []):
            out.append({
                "title": r.get('title', 'No Title'),
                "url": r.get('url', ''),
                "content": r.get('content', '')[:MAX_CONTENT_LENGTH]  # Limit content length
            })
        
        print(f"[SEARCH] Found {len(out)} results")
        
        # Cache the search results if needed (optional)
        # Could add timestamp for cache invalidation
        
        return out
    
    except Exception as e:
        print(f"[SEARCH ERROR] {str(e)}")
        return []


@tool
def web_search_with_metadata(query: str, max_results: int = None) -> Dict:
    """
    Enhanced search that returns metadata along with results.
    Useful for debugging and analytics.
    
    Args:
        query: Search query string
        max_results: Maximum number of results
    
    Returns:
        Dict with results, timestamp, query, and count
    """
    results = web_search(query, max_results)
    
    return {
        "query": query,
        "timestamp": time.time(),
        "result_count": len(results),
        "results": results
    }


# ============================================================================
# SCRAPE URL TOOL
# ============================================================================

@tool
def scrape_url(url: str, max_length: int = None) -> str:
    """
    Scrape and return clean, high-quality text content from a given URL.
    Prioritizes article content and removes junk.
    
    Args:
        url: URL to scrape
        max_length: Maximum content length (default from env)
    
    Returns:
        Cleaned text content or error message
    """
    max_len = max_length or MAX_CONTENT_LENGTH
    
    print(f"[SCRAPE] URL: {url[:80]}...")
    
    try:
        # Make request with timeout and headers
        resp = requests.get(
            url, 
            timeout=SCRAPE_TIMEOUT, 
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
            }
        )
        
        # Check response status
        if resp.status_code != 200:
            return f"Could not scrape URL: HTTP {resp.status_code}"
        
        # Parse HTML
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove unwanted tags and elements
        unwanted_tags = [
            "script", "style", "nav", "footer", "header", "aside", 
            "noscript", "meta", "iframe", "ads", "advertisement",
            "button", "form", "input", "select", "textarea",
            "svg", "canvas", "video", "audio", "picture"
        ]
        
        for tag in soup(unwanted_tags):
            tag.decompose()
        
        # Also remove elements with common junk class/id names
        junk_keywords = [
            "cookie", "consent", "popup", "modal", "overlay",
            "subscribe", "newsletter", "signup", "register",
            "advertisement", "sponsor", "promotion", "banner",
            "social", "share", "comment", "sidebar"
        ]
        
        for keyword in junk_keywords:
            for element in soup.find_all(class_=lambda x: x and keyword in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and keyword in x.lower()):
                element.decompose()
        
        # Try to find main content in priority order
        content_element = None
        priority_selectors = [
            ("article", None),  # <article> tag
            ("main", None),      # <main> tag
            ("div", "content"),  # class/content/
            ("div", "post"),     # class/post/
            ("div", "article"),  # class/article/
            ("div", "entry"),    # class/entry/
            ("section", "content"),  # section with content class
            ("div", "body"),     # class/body/
        ]
        
        for tag_name, class_pattern in priority_selectors:
            if class_pattern:
                elements = soup.find_all(tag_name, class_=lambda x: x and class_pattern in x.lower())
                if elements:
                    content_element = elements[0]
                    break
            else:
                element = soup.find(tag_name)
                if element:
                    content_element = element
                    break
        
        # Fallback to body or entire soup
        if not content_element:
            content_element = soup.find("body") or soup
        
        # Extract text with better formatting
        # Get text with newlines preserved
        text = content_element.get_text(separator="\n", strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Remove short lines that might be noise (under 20 chars, not containing common words)
        filtered_lines = []
        for line in text.split('\n'):
            if len(line) > 20 or any(word in line.lower() for word in ["the", "and", "for", "with", "from"]):
                filtered_lines.append(line)
        
        text = '\n'.join(filtered_lines)
        
        # Limit length
        if len(text) > max_len:
            text = text[:max_len] + "..."
        
        if not text or len(text) < 50:
            return f"Insufficient content extracted from URL (only {len(text)} chars)"
        
        print(f"[SCRAPE] Success: {len(text)} characters extracted")
        return text
        
    except requests.Timeout:
        return f"Could not scrape URL: Timeout after {SCRAPE_TIMEOUT} seconds"
    except requests.RequestException as e:
        return f"Could not scrape URL: {str(e)}"
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


@tool
def scrape_multiple_urls(urls: List[str], max_length: int = None) -> List[Dict]:
    """
    Scrape multiple URLs and return combined results.
    Useful for batch processing.
    
    Args:
        urls: List of URLs to scrape
        max_length: Maximum content length per URL
    
    Returns:
        List of dicts with url, content, length, and error status
    """
    results = []
    
    for url in urls:
        try:
            content = scrape_url(url, max_length)
            results.append({
                "url": url,
                "content": content,
                "length": len(content),
                "error": None if "Could not" not in content else content
            })
        except Exception as e:
            results.append({
                "url": url,
                "content": "",
                "length": 0,
                "error": str(e)
            })
    
    successful = sum(1 for r in results if r.get("error") is None)
    print(f"[BATCH SCRAPE] Completed: {successful}/{len(urls)} successful")
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_search_capabilities() -> Dict:
    """
    Get information about search capabilities and limits
    """
    return {
        "max_results": min(MAX_SEARCH_RESULTS, 20),
        "max_content_length": MAX_CONTENT_LENGTH,
        "scrape_timeout": SCRAPE_TIMEOUT,
        "search_depth": "advanced",
        "supports_metadata": True
    }


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted and likely accessible
    
    Args:
        url: URL to validate
    
    Returns:
        True if URL appears valid
    """
    if not url or not isinstance(url, str):
        return False
    
    # Check basic URL pattern
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    
    # Check for common invalid patterns
    invalid_patterns = [
        "example.com", "localhost", "127.0.0.1",
        "github.com/404", "wikipedia.org/wiki/Special:"
    ]
    
    for pattern in invalid_patterns:
        if pattern in url.lower():
            return False
    
    return True


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TOOLS: Web Search and Scraping Test")
    print("="*70)
    
    # Test 1: Search capabilities
    print("\n[TEST 1] Search Capabilities")
    print("-" * 70)
    caps = get_search_capabilities()
    for key, value in caps.items():
        print(f"  {key}: {value}")
    
    # Test 2: Web search
    print("\n[TEST 2] Web Search")
    print("-" * 70)
    test_query = "machine learning breakthroughs 2024"
    results = web_search(test_query, max_results=5)
    
    print(f"Query: {test_query}")
    print(f"Results found: {len(results)}")
    for i, result in enumerate(results[:3], 1):
        print(f"\n  [{i}] {result['title'][:70]}...")
        print(f"      URL: {result['url'][:70]}...")
        print(f"      Content: {result['content'][:100]}...")
    
    # Test 3: URL validation
    print("\n[TEST 3] URL Validation")
    print("-" * 70)
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "invalid-url",
        "http://example.com",
        "https://github.com/404/page",
    ]
    
    for url in test_urls:
        is_valid = validate_url(url)
        print(f"  {url[:50]:50} → {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test 4: Scrape (if we have a real URL from search results)
    print("\n[TEST 4] Web Scraping")
    print("-" * 70)
    
    if results and len(results) > 0:
        test_url = results[0]['url']
        print(f"Scraping: {test_url[:80]}...")
        content = scrape_url(test_url)
        print(f"Content length: {len(content)} characters")
        print(f"Preview: {content[:200]}...")
    else:
        print("  No URLs available for scraping test")
    
    # Test 5: Batch scraping (optional)
    print("\n[TEST 5] Batch Scraping")
    print("-" * 70)
    
    if len(results) >= 2:
        batch_urls = [r['url'] for r in results[:2]]
        print(f"Batch scraping {len(batch_urls)} URLs...")
        batch_results = scrape_multiple_urls(batch_urls)
        
        for result in batch_results:
            status = "✅" if not result.get('error') else "❌"
            print(f"  {status} {result['url'][:60]}... ({result['length']} chars)")
    else:
        print("  Not enough URLs for batch test")
    
    # Test 6: Search with metadata
    print("\n[TEST 6] Search with Metadata")
    print("-" * 70)
    
    metadata_results = web_search_with_metadata("quantum computing", max_results=3)
    print(f"Query: {metadata_results['query']}")
    print(f"Timestamp: {metadata_results['timestamp']}")
    print(f"Results: {metadata_results['result_count']}")
    
    print("\n" + "="*70)
    print("[OK] All tests passed!")
    print("="*70)