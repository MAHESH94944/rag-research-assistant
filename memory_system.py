"""
STEP 12: Memory System
Stores and retrieves previous research, queries, and reports
Uses ChromaDB for vector-based memory
"""

import chromadb
from chromadb.config import Settings
import json
from datetime import datetime
from typing import List, Dict, Optional
from rag import CustomHuggingFaceEmbeddings

# ============================================================================
# MEMORY MANAGER
# ============================================================================

class ResearchMemory:
    """
    Vector-based memory system for research history
    Stores: previous queries, research summaries, reports, sources
    """
    
    def __init__(self, db_path: str = "./research_memory"):
        """
        Initialize memory system with ChromaDB
        
        Args:
            db_path: Path to ChromaDB storage
        """
        self.db_path = db_path
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create/get collections
        self.queries_collection = self.client.get_or_create_collection(
            name="research_queries",
            metadata={"description": "Previous research queries"}
        )
        
        self.reports_collection = self.client.get_or_create_collection(
            name="research_reports",
            metadata={"description": "Generated research reports"}
        )
        
        self.sources_collection = self.client.get_or_create_collection(
            name="research_sources",
            metadata={"description": "Sourced and cited materials"}
        )
        
        print(f"[MEMORY] Initialized ChromaDB at {db_path}")
    
    def store_query(self, query: str, metadata: Dict = None) -> str:
        """
        Store a research query
        
        Args:
            query: The research query
            metadata: Optional metadata (timestamp, user, etc.)
        
        Returns:
            Query ID
        """
        query_id = f"query_{datetime.now().isoformat()}"
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
        }
        if metadata:
            meta.update(metadata)
        
        self.queries_collection.add(
            ids=[query_id],
            documents=[query],
            metadatas=[meta]
        )
        
        print(f"[MEMORY] Stored query: {query[:50]}...")
        return query_id
    
    def store_report(self, query: str, report: str, sources: List[Dict],
                    metadata: Dict = None) -> str:
        """
        Store a generated report with metadata
        
        Args:
            query: Original query
            report: Generated report
            sources: List of sources used
            metadata: Optional additional metadata
        
        Returns:
            Report ID
        """
        report_id = f"report_{datetime.now().isoformat()}"
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "report_length": len(report),
            "source_count": len(sources),
            "sources": json.dumps(sources),
        }
        if metadata:
            meta.update(metadata)
        
        self.reports_collection.add(
            ids=[report_id],
            documents=[report],
            metadatas=[meta]
        )
        
        print(f"[MEMORY] Stored report: {len(report)} chars, {len(sources)} sources")
        return report_id
    
    def store_source(self, url: str, title: str, content: str,
                    metadata: Dict = None) -> str:
        """
        Store a source document
        
        Args:
            url: Source URL
            title: Source title
            content: Source content
            metadata: Optional metadata
        
        Returns:
            Source ID
        """
        source_id = f"source_{datetime.now().isoformat()}"
        
        meta = {
            "timestamp": datetime.now().isoformat(),
            "url": url,
            "title": title,
            "content_length": len(content),
        }
        if metadata:
            meta.update(metadata)
        
        self.sources_collection.add(
            ids=[source_id],
            documents=[content],
            metadatas=[meta]
        )
        
        print(f"[MEMORY] Stored source: {title[:40]}...")
        return source_id
    
    def retrieve_similar_queries(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar past queries
        
        Args:
            query: Current query
            k: Number of results
        
        Returns:
            List of similar queries
        """
        results = self.queries_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        similar = []
        for i, doc in enumerate(results['documents'][0]):
            similar.append({
                "query": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                "distance": results['distances'][0][i] if results['distances'] else 0
            })
        
        return similar
    
    def retrieve_similar_reports(self, query: str, k: int = 3) -> List[Dict]:
        """
        Retrieve similar past reports
        
        Args:
            query: Current query
            k: Number of results
        
        Returns:
            List of similar reports
        """
        results = self.reports_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        similar = []
        for i, doc in enumerate(results['documents'][0]):
            similar.append({
                "report": doc[:500],  # Preview
                "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                "distance": results['distances'][0][i] if results['distances'] else 0
            })
        
        return similar
    
    def retrieve_relevant_sources(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant past sources
        
        Args:
            query: Current query
            k: Number of results
        
        Returns:
            List of relevant sources
        """
        results = self.sources_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i] if results['metadatas'][0] else {}
            sources.append({
                "title": meta.get("title", "Unknown"),
                "url": meta.get("url", ""),
                "content_preview": doc[:300],
                "distance": results['distances'][0][i] if results['distances'] else 0
            })
        
        return sources
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "queries_stored": self.queries_collection.count(),
            "reports_stored": self.reports_collection.count(),
            "sources_stored": self.sources_collection.count(),
            "total_items": (
                self.queries_collection.count() +
                self.reports_collection.count() +
                self.sources_collection.count()
            ),
            "db_path": self.db_path
        }
    
    def clear_all(self):
        """Clear all memory [USE WITH CAUTION]"""
        self.client.delete_collection(name="research_queries")
        self.client.delete_collection(name="research_reports")
        self.client.delete_collection(name="research_sources")
        print("[MEMORY] All memory cleared")

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 12: MEMORY SYSTEM TEST")
    print("="*70)
    
    # Initialize memory
    memory = ResearchMemory()
    
    # Test storing data
    print("\n[TEST 1] Storing research data...")
    
    query1 = "What are the applications of machine learning?"
    report1 = "Machine learning has applications in healthcare, finance, autonomous vehicles..."
    sources1 = [
        {"title": "ML Overview", "url": "https://example.com/ml"},
        {"title": "ML Applications", "url": "https://example.com/apps"}
    ]
    
    query_id = memory.store_query(query1)
    report_id = memory.store_report(query1, report1, sources1)
    memory.store_source("https://example.com/ml", "ML Overview", 
                       "Machine learning is a subset of AI...")
    
    # Test retrieving similar queries
    print("\n[TEST 2] Retrieving similar queries...")
    similar_queries = memory.retrieve_similar_queries("What is deep learning used for?", k=2)
    for sq in similar_queries:
        print(f"  Similar: {sq['query'][:60]}...")
    
    # Test retrieving similar reports
    print("\n[TEST 3] Retrieving similar reports...")
    similar_reports = memory.retrieve_similar_reports("ML uses", k=2)
    for sr in similar_reports:
        print(f"  Report: {sr['report'][:80]}...")
    
    # Test retrieving sources
    print("\n[TEST 4] Retrieving relevant sources...")
    sources = memory.retrieve_relevant_sources("learning algorithms", k=2)
    for s in sources:
        print(f"  Source: {s['title']} ({s['url']})")
    
    # Get stats
    print("\n[TEST 5] Memory statistics...")
    stats = memory.get_memory_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    print("\n[OK] Memory system test passed!")
