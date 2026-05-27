"""
STEP 13: Async Scraping
Concurrent scraping using aiohttp and asyncio
Scrapes multiple URLs simultaneously for better performance
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict
import time

# ============================================================================
# ASYNC SCRAPER
# ============================================================================

class AsyncScraper:
    """Concurrent web scraper using aiohttp"""
    
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        )
    }
    
    MAX_CONTENT_LENGTH = 3000
    TIMEOUT = aiohttp.ClientTimeout(total=10)
    
    @staticmethod
    async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict:
        """
        Fetch and scrape a single URL
        
        Args:
            session: aiohttp session
            url: URL to fetch
        
        Returns:
            Dict with url, title, content
        """
        try:
            async with session.get(url, timeout=AsyncScraper.TIMEOUT, 
                                   headers=AsyncScraper.HEADERS) as response:
                
                if response.status != 200:
                    return {"url": url, "title": "Error", "content": "", "error": f"Status {response.status}"}
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "Unknown"
                
                # Extract content (prioritize semantic tags)
                content = ""
                for tag in soup.find_all(['article', 'main', 'section']):
                    if tag:
                        content += tag.get_text(separator=' ', strip=True)
                        if len(content) > AsyncScraper.MAX_CONTENT_LENGTH:
                            break
                
                # Fallback to body if no semantic tags
                if not content and soup.body:
                    content = soup.body.get_text(separator=' ', strip=True)
                
                # Remove junk tags from content
                junk_phrases = [
                    "advertisement", "cookie", "subscribe", "sign up",
                    "privacy policy", "terms of service", "copyright"
                ]
                
                lines = content.split('\n')
                clean_lines = [
                    line for line in lines
                    if not any(junk in line.lower() for junk in junk_phrases)
                ]
                
                content = '\n'.join(clean_lines)[:AsyncScraper.MAX_CONTENT_LENGTH]
                
                return {
                    "url": url,
                    "title": title,
                    "content": content,
                    "error": None
                }
        
        except asyncio.TimeoutError:
            return {"url": url, "title": "Timeout", "content": "", "error": "Timeout"}
        except Exception as e:
            return {"url": url, "title": "Error", "content": "", "error": str(e)}
    
    @staticmethod
    async def scrape_urls(urls: List[str], max_concurrent: int = 5) -> List[Dict]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            max_concurrent: Max concurrent requests
        
        Returns:
            List of scraped content
        """
        # Create connector with connection limit
        connector = aiohttp.TCPConnector(limit=max_concurrent)
        
        async with aiohttp.ClientSession(connector=connector, 
                                         timeout=AsyncScraper.TIMEOUT) as session:
            
            # Create tasks for all URLs
            tasks = [
                AsyncScraper.fetch_url(session, url)
                for url in urls
            ]
            
            # Run tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            cleaned_results = []
            for result in results:
                if isinstance(result, Exception):
                    cleaned_results.append({
                        "url": "unknown",
                        "title": "Error",
                        "content": "",
                        "error": str(result)
                    })
                else:
                    cleaned_results.append(result)
            
            return cleaned_results

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def scrape_urls_async(urls: List[str], max_concurrent: int = 5) -> List[Dict]:
    """
    Synchronous wrapper for async scraping
    
    Args:
        urls: List of URLs to scrape
        max_concurrent: Max concurrent requests
    
    Returns:
        List of scraped content
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            AsyncScraper.scrape_urls(urls, max_concurrent)
        )
        return results
    finally:
        loop.close()

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 13: ASYNC SCRAPING TEST")
    print("="*70)
    
    # Test URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Data_science"
    ]
    
    print(f"\n[TEST] Scraping {len(test_urls)} URLs concurrently...")
    print("="*70)
    
    # Time the scraping
    start_time = time.time()
    
    results = scrape_urls_async(test_urls, max_concurrent=5)
    
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.2f} seconds\n")
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result['title'][:40]}")
        print(f"    URL: {result['url']}")
        if result['error']:
            print(f"    Error: {result['error']}")
        else:
            print(f"    Content: {len(result['content'])} chars")
            print(f"    Preview: {result['content'][:100]}...")
        print()
    
    # Summary
    successful = sum(1 for r in results if not r['error'])
    failed = len(results) - successful
    
    print("="*70)
    print(f"Summary: {successful} successful, {failed} failed")
    print(f"Avg time per URL: {elapsed/len(test_urls):.2f}s")
    print("[OK] Async scraping test passed!")
