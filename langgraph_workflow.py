"""
STEP 10: LangGraph Workflow
Graph-based orchestration for RAG pipeline
Replaces manual sequential pipeline with state graph
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from typing_extensions import TypedDict
from typing import Annotated, List, Optional
from operator import add
import json

from agents import writer_chain, critic_chain
from tools import web_search, scrape_url
from rag import RAGSystem

# ============================================================================
# STATE DEFINITION
# ============================================================================

class PipelineState(TypedDict):
    """State passed through the workflow graph"""
    topic: str
    search_results: List[dict]
    urls: List[str]
    scraped_content: List[dict]
    chunks: List[Document]
    rag_system: Optional[RAGSystem]
    retrieved_context: str
    report: str
    feedback: str
    sources: List[dict]
    metadata: dict  # For tracking execution info

# ============================================================================
# WORKFLOW NODES
# ============================================================================

def search_node(state: PipelineState) -> PipelineState:
    """
    Node 1: Web Search
    Performs web search for the given topic
    """
    print("\n[GRAPH] Entering SEARCH Node")
    print(f"  Topic: {state['topic']}")
    
    try:
        search_results = web_search.invoke(state['topic'])
        
        # Handle both list and agent message formats
        if isinstance(search_results, list):
            search_results_list = search_results
        elif isinstance(search_results, dict) and 'text' in search_results:
            try:
                search_results_list = json.loads(search_results['text'])
            except:
                search_results_list = []
        else:
            search_results_list = search_results if isinstance(search_results, list) else []
        
        state['search_results'] = search_results_list
        state['metadata']['search_count'] = len(search_results_list)
        
        print(f"  [OK] Found {len(search_results_list)} results")
        return state
    except Exception as e:
        print(f"  [ERROR] Search failed: {e}")
        state['search_results'] = []
        return state

def scrape_node(state: PipelineState) -> PipelineState:
    """
    Node 2: Content Scraping
    Extracts URLs and scrapes top 3 sources
    """
    print("\n[GRAPH] Entering SCRAPE Node")
    
    if not state['search_results']:
        print("  [WARN] No search results to scrape")
        state['scraped_content'] = []
        return state
    
    # Extract URLs
    urls = [item["url"] for item in state['search_results']]
    state['urls'] = urls[:10]  # Keep top 10
    
    # Create URL to title mapping
    url_to_title = {item["url"]: item["title"] for item in state['search_results']}
    
    # Scrape top 3 URLs
    all_content = []
    num_to_scrape = min(3, len(urls))
    
    print(f"  Scraping {num_to_scrape} URLs...")
    
    for i, url in enumerate(urls[:num_to_scrape], 1):
        try:
            content = scrape_url.invoke(url)
            all_content.append({
                "url": url,
                "title": url_to_title.get(url, "Unknown"),
                "content": content,
                "length": len(content)
            })
            print(f"    [{i}] {url[:50]}... ({len(content)} chars)")
        except Exception as e:
            print(f"    [{i}] [SKIP] {url[:50]}... ({str(e)[:30]})")
    
    state['scraped_content'] = all_content
    state['metadata']['scraped_count'] = len(all_content)
    print(f"  [OK] Scraped {len(all_content)} sources")
    
    return state

def chunk_node(state: PipelineState) -> PipelineState:
    """
    Node 3: Text Chunking
    Converts scraped content into RAG chunks
    """
    print("\n[GRAPH] Entering CHUNK Node")
    
    if not state['scraped_content']:
        print("  [WARN] No content to chunk")
        state['chunks'] = []
        return state
    
    # Extract content from scraped items
    documents = [item['content'] for item in state['scraped_content']]
    
    # Create RAG system and chunk
    rag = RAGSystem(chunk_size=1000, chunk_overlap=200, retrieval_k=3)
    
    print(f"  Building RAG from {len(documents)} documents...")
    rag.build(documents)
    
    state['chunks'] = rag.chunks
    state['rag_system'] = rag
    state['metadata']['chunk_count'] = len(rag.chunks)
    
    print(f"  [OK] Created {len(rag.chunks)} chunks")
    return state

def retriever_node(state: PipelineState) -> PipelineState:
    """
    Node 4: Semantic Retrieval
    Retrieves most relevant chunks for the topic
    """
    print("\n[GRAPH] Entering RETRIEVER Node")
    
    if not state['rag_system']:
        print("  [WARN] No RAG system available")
        state['retrieved_context'] = ""
        return state
    
    print(f"  Retrieving context for: {state['topic']}")
    
    # Retrieve relevant chunks
    retrieved_chunks = state['rag_system'].query(state['topic'], k=3)
    
    # Format with sources
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source_idx = int(chunk.metadata.get('source', 'doc_0').split('_')[-1])
        if source_idx < len(state['scraped_content']):
            source_url = state['scraped_content'][source_idx]['url']
            context_lines.append(f"[Source {i}] {source_url}\n{chunk.page_content}")
    
    retrieved_context = "\n\n".join(context_lines)
    state['retrieved_context'] = retrieved_context
    state['metadata']['retrieved_chunks'] = len(retrieved_chunks)
    
    print(f"  [OK] Retrieved {len(retrieved_chunks)} relevant chunks")
    return state

def writer_node(state: PipelineState) -> PipelineState:
    """
    Node 5: Report Writing
    Generates report using RAG context
    """
    print("\n[GRAPH] Entering WRITER Node")
    
    print(f"  Generating report for: {state['topic']}")
    
    # Generate report
    report = writer_chain.invoke({
        "topic": state['topic'],
        "research": state['retrieved_context']
    })
    
    # Add sources
    sources_section = "\n\n## Sources\n\n"
    sources = []
    for i, item in enumerate(state['scraped_content'], 1):
        sources_section += f"{i}. **{item['title']}**\n   {item['url']}\n\n"
        sources.append({
            "title": item['title'],
            "url": item['url']
        })
    
    report_with_sources = report + sources_section
    state['report'] = report_with_sources
    state['sources'] = sources
    
    print(f"  [OK] Report generated ({len(report_with_sources)} chars)")
    return state

def critic_node(state: PipelineState) -> PipelineState:
    """
    Node 6: Quality Review
    Provides feedback and scoring
    """
    print("\n[GRAPH] Entering CRITIC Node")
    
    print("  Reviewing report quality...")
    
    # Get feedback
    feedback = critic_chain.invoke({
        "report": state['report']
    })
    
    state['feedback'] = feedback
    print(f"  [OK] Feedback provided ({len(feedback)} chars)")
    
    return state

# ============================================================================
# WORKFLOW GRAPH
# ============================================================================

def create_workflow_graph():
    """
    Creates and compiles the LangGraph workflow
    
    Graph structure:
    START
      ↓
    SEARCH → SCRAPE → CHUNK → RETRIEVER → WRITER → CRITIC
      ↓
     END
    """
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    
    # Add edges (linear flow)
    workflow.add_edge(START, "search")
    workflow.add_edge("search", "scrape")
    workflow.add_edge("scrape", "chunk")
    workflow.add_edge("chunk", "retriever")
    workflow.add_edge("retriever", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", END)
    
    # Compile graph
    return workflow.compile()

# ============================================================================
# EXECUTION
# ============================================================================

def run_graph_pipeline(topic: str) -> dict:
    """
    Executes the workflow graph for a given topic
    
    Args:
        topic: Research topic
    
    Returns:
        Final state with all results
    """
    print("\n" + "="*70)
    print("LANGGRAPH WORKFLOW EXECUTION")
    print("="*70)
    
    # Create graph
    graph = create_workflow_graph()
    
    # Initialize state
    initial_state = PipelineState(
        topic=topic,
        search_results=[],
        urls=[],
        scraped_content=[],
        chunks=[],
        rag_system=None,
        retrieved_context="",
        report="",
        feedback="",
        sources=[],
        metadata={
            "topic": topic,
            "start_time": None,
            "search_count": 0,
            "scraped_count": 0,
            "chunk_count": 0,
            "retrieved_chunks": 0
        }
    )
    
    # Execute graph
    final_state = graph.invoke(initial_state)
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nExecution Summary:")
    print(f"  - Topic: {final_state['metadata']['topic']}")
    print(f"  - Search Results: {final_state['metadata']['search_count']}")
    print(f"  - Scraped Sources: {final_state['metadata']['scraped_count']}")
    print(f"  - Chunks Created: {final_state['metadata']['chunk_count']}")
    print(f"  - Chunks Retrieved: {final_state['metadata']['retrieved_chunks']}")
    print(f"  - Report Length: {len(final_state['report'])} characters")
    
    return final_state

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test workflow
    topic = "machine learning applications"
    result = run_graph_pipeline(topic)
    
    print(f"\n[REPORT PREVIEW]\n{result['report'][:300]}...\n")
    print(f"[FEEDBACK PREVIEW]\n{result['feedback'][:300]}...\n")
