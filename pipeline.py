"""
PIPELINE: Main Research Pipeline with RAG Integration
PHASES 1-2: Foundation + RAG Implementation
"""

from agents import writer_chain, critic_chain
from tools import scrape_url, web_search
from rag import RAGSystem
import sys

def run_research_pipeline(topic: str) -> dict:
    """
    Main research pipeline with RAG integration
    
    Args:
        topic: Research topic/question
    
    Returns:
        Dictionary with search_results, urls, scraped_content, 
        report, sources, feedback
    """
    
    state = {}

    # =========================================================================
    # STEP 1: Web Search
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 1 - Searching for information ...")
    print("="*50)

    # Direct tool call (no agent wrapper for better control)
    search_results_raw = web_search.invoke(topic)
    
    # Handle both structured list and agent message formats
    if isinstance(search_results_raw, list):
        search_results_list = search_results_raw
    elif isinstance(search_results_raw, dict) and 'type' in search_results_raw:
        # Agent message format - extract text
        import json
        try:
            search_results_list = json.loads(search_results_raw.get('text', '[]'))
        except:
            search_results_list = []
    else:
        search_results_list = search_results_raw if isinstance(search_results_raw, list) else []
    
    state["search_results"] = search_results_list
    
    print(f"\n[OK] Search results:")
    if search_results_list and isinstance(search_results_list, list):
        print(f"  Found {len(search_results_list)} results")
        for i, r in enumerate(search_results_list[:3]):
            if isinstance(r, dict) and 'title' in r:
                print(f"  [{i+1}] {r['title'][:70]}...")

    # =========================================================================
    # STEP 2: Extract URLs directly (removed reader agent)
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 2 - Extracting URLs directly from search results ...")
    print("="*50)

    # Extract URLs manually from structured search results
    urls = [item["url"] for item in (state['search_results'] if isinstance(state['search_results'], list) else [])]
    state['urls'] = urls
    
    print(f"\n[OK] Extracted {len(urls)} URLs")
    for i, url in enumerate(urls[:5], 1):
        print(f"  [{i}] {url[:80]}...")

    # =========================================================================
    # STEP 3: Multi-URL Scraping
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 3 - Scraping top URLs for detailed content ...")
    print("="*50)
    
    all_content = []
    num_urls_to_scrape = min(5, len(urls))  # Scrape up to 5 URLs
    
    # Create URL to title mapping for sources
    url_to_title = {item["url"]: item["title"] for item in state['search_results'] if isinstance(item, dict)}
    
    for i, url in enumerate(urls[:num_urls_to_scrape], 1):
        print(f"\n  [{i}/{num_urls_to_scrape}] Scraping: {url[:60]}...")
        try:
            content = scrape_url.invoke(url)
            if content and len(content) > 100:  # Only keep if substantial content
                all_content.append({
                    "url": url,
                    "title": url_to_title.get(url, "Unknown Source"),
                    "content": content,
                    "length": len(content)
                })
                print(f"       [OK] Scraped {len(content)} characters")
            else:
                print(f"       [SKIP] Insufficient content ({len(content)} chars)")
        except Exception as e:
            print(f"       [FAIL] Failed: {str(e)[:50]}")
    
    state['scraped_content'] = all_content
    print(f"\n[OK] Successfully scraped {len(all_content)} URLs")
    if all_content:
        total_length = sum(item['length'] for item in all_content)
        print(f"  Total content length: {total_length} characters")

    # =========================================================================
    # STEP 4: RAG System - Build, Retrieve, and Generate Report
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 4 - Building RAG system and retrieving context ...")
    print("="*50)

    if state['scraped_content']:
        print(f"\n  Building RAG system from {len(state['scraped_content'])} scraped documents...")
        
        # Extract just the content from scraped items
        documents = [item['content'] for item in state['scraped_content']]
        
        # Build RAG system once
        rag = RAGSystem(chunk_size=1000, chunk_overlap=200, retrieval_k=3)
        rag.build(documents)
        
        # Retrieve relevant context based on the topic
        print(f"  Retrieving relevant context for topic: '{topic}'...")
        
        # SINGLE RETRIEVAL - Get chunks and context in one operation
        retrieved_chunks = rag.query(topic, k=3)
        
        # Build context with sources
        context_with_sources = "RETRIEVED CONTEXT FROM RAG:\n\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            # Extract source index from metadata
            source_meta = chunk.metadata.get('source', 'doc_0')
            try:
                source_idx = int(source_meta.split('_')[-1])
            except (ValueError, IndexError):
                source_idx = 0
            
            if source_idx < len(state['scraped_content']):
                source_url = state['scraped_content'][source_idx]['url']
                source_title = state['scraped_content'][source_idx]['title']
                context_with_sources += f"[{i}] Source: {source_title}\n    URL: {source_url}\n    Content: {chunk.page_content}\n\n"
            else:
                context_with_sources += f"[{i}] {chunk.page_content}\n\n"
        
        print(f"  [OK] Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Prepare research context for writer
        research_combined = (
            f"SEARCH RESULTS SUMMARY:\n"
            f"Found {len(state['search_results'])} results for '{topic}'\n\n"
            f"{context_with_sources}"
        )
        
        # Also store retrieved context for potential future use
        state['retrieved_context'] = context_with_sources
        state['rag_chunks_retrieved'] = len(retrieved_chunks)
        
    else:
        # Fallback if no content was scraped
        print("\n  [WARN] No scraped content available, using search results only...")
        search_text = "\n".join([
            f"- {item.get('title', 'Unknown')}: {item.get('content', '')[:200]}"
            for item in (state['search_results'] if isinstance(state['search_results'], list) else [])[:5]
        ])
        research_combined = f"SEARCH RESULTS:\n{search_text}"
        state['retrieved_context'] = research_combined

    # =========================================================================
    # STEP 4b: Writer Chain - Generate Report
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 4b - Writer is drafting the report ...")
    print("="*50)

    try:
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined
        })
        print(f"  [OK] Report generated: {len(state['report'])} characters")
    except Exception as e:
        print(f"  [ERROR] Writer failed: {e}")
        state["report"] = f"Error generating report: {e}"

    # =========================================================================
    # STEP 4c: Add Source Citations
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 4c - Adding source citations ...")
    print("="*50)
    
    # Collect unique sources from search results and scraped content
    sources = []
    sources_added = set()
    
    # Add sources from scraped content (most authoritative)
    for item in state['scraped_content']:
        if item['url'] not in sources_added:
            sources.append({
                "title": item.get("title", "Unknown Source"),
                "url": item['url']
            })
            sources_added.add(item['url'])
    
    # Add remaining sources from search results
    for item in (state['search_results'] if isinstance(state['search_results'], list) else [])[:5]:
        if isinstance(item, dict) and item.get('url') not in sources_added:
            sources.append({
                "title": item.get('title', 'Unknown Source'),
                "url": item.get('url', '')
            })
            sources_added.add(item.get('url'))
    
    # Build sources section
    sources_section = "\n\n## Sources\n\n"
    for i, source in enumerate(sources, 1):
        sources_section += f"{i}. **{source['title']}**\n   {source['url']}\n\n"
    
    # Append sources to report (if not already present)
    if "## Sources" not in state["report"]:
        state["report"] = state["report"] + sources_section
    else:
        print("  [NOTE] Sources section already in report")
    
    state["sources"] = sources
    
    print(f"[OK] Added {len(sources)} sources to report")

    # Display report preview
    print("\n" + "="*50)
    print("REPORT PREVIEW (first 500 chars):")
    print("="*50)
    print(state['report'][:500])
    if len(state['report']) > 500:
        print("...")

    # =========================================================================
    # STEP 5: Critic Review
    # =========================================================================
    print("\n" + "="*50)
    print("STEP 5 - Critic is reviewing the report ...")
    print("="*50)

    try:
        state["feedback"] = critic_chain.invoke({
            "report": state['report']
        })
        print(f"  [OK] Feedback generated: {len(state['feedback'])} characters")
    except Exception as e:
        print(f"  [ERROR] Critic failed: {e}")
        state["feedback"] = f"Error generating feedback: {e}"

    print("\n" + "="*50)
    print("CRITIC FEEDBACK (first 500 chars):")
    print("="*50)
    print(state['feedback'][:500])
    if len(state['feedback']) > 500:
        print("...")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"\nSummary:")
    print(f"  - Search Results: {len(state.get('search_results', []))}")
    print(f"  - URLs Extracted: {len(state.get('urls', []))}")
    print(f"  - Scraped Sources: {len(state.get('scraped_content', []))}")
    print(f"  - RAG Chunks: {state.get('rag_chunks_retrieved', 0)}")
    print(f"  - Report Length: {len(state.get('report', ''))} characters")
    print(f"  - Sources Cited: {len(state.get('sources', []))}")
    print(f"  - Feedback Length: {len(state.get('feedback', ''))} characters")

    return state


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def run_pipeline_simple(topic: str) -> dict:
    """
    Run pipeline with minimal output (for programmatic use)
    
    Args:
        topic: Research topic
    
    Returns:
        State dictionary with results
    """
    # Temporarily suppress detailed output
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        result = run_research_pipeline(topic)
    finally:
        sys.stdout = old_stdout
    
    return result


def get_report_only(topic: str) -> str:
    """
    Get only the report text (no feedback or metadata)
    
    Args:
        topic: Research topic
    
    Returns:
        Report text with sources
    """
    result = run_research_pipeline(topic)
    return result.get("report", "")


def get_feedback_only(topic: str) -> str:
    """
    Get only the feedback text
    
    Args:
        topic: Research topic
    
    Returns:
        Feedback text
    """
    result = run_research_pipeline(topic)
    return result.get("feedback", "")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Get topic from user
    topic = input("\n🔍 Enter a research topic: ").strip()
    
    if not topic:
        print("❌ No topic entered. Using default: 'artificial intelligence'")
        topic = "artificial intelligence"
    
    print(f"\n📚 Researching: {topic}")
    print("⏳ This may take 30-60 seconds...\n")
    
    # Run pipeline
    result = run_research_pipeline(topic)
    
    # Save results to file
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_{topic.replace(' ', '_')[:30]}_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"RESEARCH REPORT: {topic}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*70 + "\n\n")
        f.write(result.get("report", "No report generated"))
        f.write("\n\n" + "="*70 + "\n")
        f.write("CRITIC FEEDBACK\n")
        f.write("="*70 + "\n\n")
        f.write(result.get("feedback", "No feedback generated"))
    
    print(f"\n💾 Report saved to: {filename}")