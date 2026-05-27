"""
PHASE 2: Real RAG Implementation
Implements text chunking, embeddings, FAISS vector DB, and retriever
with save/load capabilities and async support
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os
import json
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# CUSTOM EMBEDDINGS CLASS (using sentence-transformers directly)
# ============================================================================

class CustomHuggingFaceEmbeddings(Embeddings):
    """Custom embeddings using sentence-transformers with caching"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir=None):
        self.model_name = model_name
        cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
    
    def embed_documents(self, texts):
        """Embed search documents."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text):
        """Embed query text."""
        embedding = self.model.encode(text)
        return embedding.tolist()


# ============================================================================
# PHASE 2.1: TEXT CHUNKING
# ============================================================================

def create_text_splitter(chunk_size=1000, chunk_overlap=200):
    """
    Create a recursive character text splitter for breaking large texts into chunks.
    
    Args:
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
    
    Returns:
        RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks.
    
    Args:
        documents: List of Document objects or strings
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document chunks with metadata
    """
    splitter = create_text_splitter(chunk_size, chunk_overlap)
    
    # Convert strings to Documents if needed
    docs = []
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            docs.append(Document(page_content=doc, metadata={"source": f"doc_{i}", "index": i}))
        elif isinstance(doc, Document):
            docs.append(doc)
        elif isinstance(doc, dict) and "page_content" in doc:
            docs.append(Document(page_content=doc["page_content"], metadata=doc.get("metadata", {"source": f"doc_{i}"})))
        else:
            docs.append(Document(page_content=str(doc), metadata={"source": f"doc_{i}"}))
    
    # Split documents
    chunks = splitter.split_documents(docs)
    
    # Add chunk indices for better tracking
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    return chunks

# ============================================================================
# PHASE 2.2: EMBEDDINGS
# ============================================================================

def create_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Create HuggingFace embeddings for semantic search.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        CustomHuggingFaceEmbeddings instance
    """
    embeddings = CustomHuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# ============================================================================
# PHASE 2.3: FAISS VECTOR DATABASE
# ============================================================================

def create_vector_store(chunks, embeddings=None):
    """
    Create FAISS vector database from document chunks.
    
    Args:
        chunks: List of Document chunks
        embeddings: HuggingFaceEmbeddings instance
    
    Returns:
        FAISS vector store
    """
    if embeddings is None:
        embeddings = create_embeddings()
    
    # Create FAISS index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vector_store

# ============================================================================
# PHASE 2.4: RETRIEVER
# ============================================================================

def create_retriever(vector_store, k=3, search_type="similarity"):
    """
    Create a retriever for semantic search.
    
    Args:
        vector_store: FAISS vector store
        k: Number of top chunks to retrieve
        search_type: Type of search ("similarity" or "mmr")
    
    Returns:
        Retriever instance
    """
    if search_type == "similarity":
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
    elif search_type == "mmr":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 2}
        )
    else:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
    return retriever

def retrieve_relevant_chunks(query, retriever, k=3):
    """
    Retrieve the top k most relevant chunks for a query.
    
    Args:
        query: Search query
        retriever: Retriever instance
        k: Number of results (overridden by retriever config)
    
    Returns:
        List of relevant Document chunks
    """
    results = retriever.invoke(query)
    return results

# ============================================================================
# UNIFIED RAG SYSTEM WITH SAVE/LOAD
# ============================================================================

class RAGSystem:
    """
    Complete RAG system: Chunking → Embeddings → Vector DB → Retriever
    Supports save/load, async operations, and source tracking
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=200, embedding_model="all-MiniLM-L6-v2", retrieval_k=3):
        """
        Initialize RAG system.
        
        Args:
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
            embedding_model: HuggingFace model for embeddings
            retrieval_k: Number of chunks to retrieve
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.retrieval_k = retrieval_k
        
        self.splitter = None
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.chunks = None
        self.source_map = {}  # Map chunk indices to source URLs/titles
        self.build_timestamp = None
        
    def build(self, documents, source_metadata=None):
        """
        Build complete RAG system from documents.
        
        Args:
            documents: List of document strings or Document objects
            source_metadata: Optional list of metadata for each document
        
        Returns:
            Self for chaining
        """
        import time
        self.build_timestamp = time.time()
        
        print("\n" + "="*70)
        print("Building RAG System")
        print("="*70)
        
        # Step 1: Create text splitter
        print("\n[1/4] Creating text splitter...")
        self.splitter = create_text_splitter(self.chunk_size, self.chunk_overlap)
        print("      [OK] Splitter created")
        
        # Step 2: Split documents into chunks
        print("\n[2/4] Splitting documents into chunks...")
        self.chunks = split_documents(documents, self.chunk_size, self.chunk_overlap)
        print(f"      [OK] Created {len(self.chunks)} chunks")
        if self.chunks:
            avg_size = sum(len(c.page_content) for c in self.chunks) // len(self.chunks)
            print(f"      - Average chunk size: {avg_size} characters")
        
        # Build source mapping
        if source_metadata:
            for idx, chunk in enumerate(self.chunks):
                source_idx = chunk.metadata.get('source', 'doc_0').split('_')[-1]
                try:
                    source_idx_int = int(source_idx)
                    if source_idx_int < len(source_metadata):
                        self.source_map[idx] = source_metadata[source_idx_int]
                except (ValueError, IndexError):
                    pass
        
        # Step 3: Create embeddings
        print("\n[3/4] Initializing embeddings...")
        self.embeddings = create_embeddings(self.embedding_model)
        print(f"      [OK] Embeddings initialized")
        print(f"      - Model: {self.embedding_model}")
        
        # Step 4: Create vector store and retriever
        print("\n[4/4] Creating vector database and retriever...")
        self.vector_store = create_vector_store(self.chunks, self.embeddings)
        self.retriever = create_retriever(self.vector_store, self.retrieval_k)
        print(f"      [OK] FAISS vector store created")
        print(f"      [OK] Retriever initialized (k={self.retrieval_k})")
        
        print("\n" + "="*70)
        print("[OK] RAG System Ready!")
        print("="*70)
        
        return self
    
    def query(self, query_text, k=None):
        """
        Query the RAG system for relevant chunks.
        
        Args:
            query_text: Search query
            k: Override number of results (uses self.retrieval_k if None)
        
        Returns:
            List of relevant Document chunks
        """
        if self.retriever is None:
            raise ValueError("RAG system not built yet. Call build() first.")
        
        k = k or self.retrieval_k
        results = self.retriever.invoke(query_text)
        return results[:k]
    
    def query_with_scores(self, query_text, k=None):
        """
        Query the RAG system and return chunks with similarity scores.
        
        Args:
            query_text: Search query
            k: Number of results
        
        Returns:
            List of tuples (Document, score)
        """
        if self.vector_store is None:
            raise ValueError("RAG system not built yet. Call build() first.")
        
        k = k or self.retrieval_k
        results_with_scores = self.vector_store.similarity_search_with_score(query_text, k=k)
        return results_with_scores
    
    def get_context(self, query_text, k=None, include_sources=True):
        """
        Get context string from top k relevant chunks.
        
        Args:
            query_text: Search query
            k: Number of chunks
            include_sources: Whether to include source information
        
        Returns:
            Combined context string
        """
        results = self.query(query_text, k)
        
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = result.metadata.get('source', 'unknown')
            if include_sources and source_info in self.source_map:
                source = self.source_map[source_info]
                context_parts.append(f"[{i}] Source: {source.get('title', 'Unknown')}\n    URL: {source.get('url', '')}\n    Content: {result.page_content}")
            else:
                context_parts.append(f"[{i}] {result.page_content}")
        
        return "\n\n".join(context_parts)
    
    # ========================================================================
    # SAVE/LOAD FUNCTIONALITY
    # ========================================================================
    
    def save(self, path: str):
        """
        Save RAG system to disk.
        
        Args:
            path: Directory path to save to
        """
        if self.vector_store is None:
            raise ValueError("Cannot save: RAG system not built yet")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Saving RAG system to {path}]")
        
        # Save FAISS index
        index_path = save_path / "faiss_index"
        self.vector_store.save_local(str(index_path))
        print(f"  [OK] FAISS index saved")
        
        # Save chunks as pickle
        chunks_path = save_path / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'source_map': self.source_map,
                'build_timestamp': self.build_timestamp
            }, f)
        print(f"  [OK] Chunks saved ({len(self.chunks)} items)")
        
        # Save configuration
        config = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'retrieval_k': self.retrieval_k,
            'build_timestamp': self.build_timestamp,
            'total_chunks': len(self.chunks) if self.chunks else 0
        }
        
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  [OK] Configuration saved")
        
        print(f"[OK] RAG system saved to {path}")
    
    def load(self, path: str):
        """
        Load RAG system from disk.
        
        Args:
            path: Directory path to load from
        
        Returns:
            Self for chaining
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        
        print(f"\n[Loading RAG system from {path}]")
        
        # Load configuration
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.chunk_size = config.get('chunk_size', self.chunk_size)
            self.chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
            self.embedding_model = config.get('embedding_model', self.embedding_model)
            self.retrieval_k = config.get('retrieval_k', self.retrieval_k)
            self.build_timestamp = config.get('build_timestamp')
            print(f"  [OK] Configuration loaded")
        
        # Initialize embeddings
        self.embeddings = create_embeddings(self.embedding_model)
        
        # Load FAISS index
        index_path = load_path / "faiss_index"
        if index_path.exists():
            self.vector_store = FAISS.load_local(
                str(index_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = create_retriever(self.vector_store, self.retrieval_k)
            print(f"  [OK] FAISS index loaded")
        else:
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        # Load chunks
        chunks_path = load_path / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, 'rb') as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.source_map = data.get('source_map', {})
            print(f"  [OK] Chunks loaded ({len(self.chunks)} items)")
        
        print(f"[OK] RAG system loaded from {path}")
        return self
    
    # ========================================================================
    # ASYNC OPERATIONS
    # ========================================================================
    
    async def aquery(self, query_text: str, k: int = None) -> List[Document]:
        """
        Async query the RAG system.
        
        Args:
            query_text: Search query
            k: Number of results
        
        Returns:
            List of relevant Document chunks
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                self.query, 
                query_text, 
                k
            )
        return result
    
    async def aget_context(self, query_text: str, k: int = None, include_sources: bool = True) -> str:
        """
        Async get context string.
        
        Args:
            query_text: Search query
            k: Number of chunks
            include_sources: Whether to include source information
        
        Returns:
            Combined context string
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.get_context,
                query_text,
                k,
                include_sources
            )
        return result
    
    async def abuild(self, documents: List, source_metadata: List = None):
        """
        Async build RAG system.
        
        Args:
            documents: List of documents
            source_metadata: Optional source metadata
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                self.build,
                documents,
                source_metadata
            )
        return self
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_chunks": len(self.chunks) if self.chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "retrieval_k": self.retrieval_k,
            "vector_store_type": type(self.vector_store).__name__ if self.vector_store else None,
            "build_timestamp": self.build_timestamp,
            "source_count": len(self.source_map)
        }
    
    def add_sources(self, source_metadata: List[Dict]):
        """
        Add source metadata mapping.
        
        Args:
            source_metadata: List of source dicts with url, title, etc.
        """
        for idx, source in enumerate(source_metadata):
            self.source_map[str(idx)] = source
    
    def get_source_for_chunk(self, chunk_index: int) -> Optional[Dict]:
        """
        Get source metadata for a specific chunk.
        
        Args:
            chunk_index: Index of the chunk
        
        Returns:
            Source metadata dict or None
        """
        return self.source_map.get(str(chunk_index))
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Alias for query method"""
        return self.query(query, k)
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """Get chunks with similarity scores"""
        return self.query_with_scores(query, k)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_rag_system(path: str) -> RAGSystem:
    """Convenience function to load a saved RAG system"""
    rag = RAGSystem()
    return rag.load(path)


def create_rag_from_documents(documents: List, source_metadata: List = None, save_path: str = None) -> RAGSystem:
    """Create and optionally save a RAG system from documents"""
    rag = RAGSystem()
    rag.build(documents, source_metadata)
    
    if save_path:
        rag.save(save_path)
    
    return rag


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("PHASE 2: RAG System with Save/Load + Async Test")
    print("="*70)
    
    # Sample documents with source metadata
    sample_docs = [
        "Artificial intelligence is transforming the world. Machine learning algorithms learn from data without being explicitly programmed. Deep learning uses neural networks with multiple layers.",
        "Climate change is a global challenge. Renewable energy sources like solar and wind are becoming increasingly important. Carbon emissions must be reduced to meet climate goals.",
        "Quantum computing represents the future of computation. Qubits can exist in multiple states simultaneously. Quantum algorithms can solve certain problems exponentially faster.",
        "Blockchain technology enables secure and transparent transactions. Smart contracts automate agreements on the blockchain. Cryptocurrencies like Bitcoin use blockchain technology.",
    ]
    
    source_metadata = [
        {"title": "AI Overview", "url": "https://example.com/ai", "type": "article"},
        {"title": "Climate Change Report", "url": "https://example.com/climate", "type": "report"},
        {"title": "Quantum Computing Guide", "url": "https://example.com/quantum", "type": "guide"},
        {"title": "Blockchain Explained", "url": "https://example.com/blockchain", "type": "article"},
    ]
    
    # Test 1: Create and build RAG system
    print("\n[TEST 1] Creating RAG system...")
    rag = RAGSystem(chunk_size=200, chunk_overlap=50, retrieval_k=2)
    rag.build(sample_docs, source_metadata)
    
    # Test 2: Query
    print("\n[TEST 2] Querying...")
    results = rag.query("What is machine learning?")
    print(f"Retrieved {len(results)} chunks")
    for i, result in enumerate(results, 1):
        print(f"  [{i}] {result.page_content[:100]}...")
    
    # Test 3: Save
    print("\n[TEST 3] Saving RAG system...")
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    rag.save(temp_dir)
    
    # Test 4: Load
    print("\n[TEST 4] Loading RAG system...")
    rag2 = RAGSystem()
    rag2.load(temp_dir)
    
    # Test 5: Verify loaded system works
    print("\n[TEST 5] Testing loaded system...")
    results2 = rag2.query("renewable energy")
    print(f"Loaded system retrieved {len(results2)} chunks")
    
    # Test 6: Async query
    print("\n[TEST 6] Async query test...")
    async def test_async():
        results = await rag.aquery("quantum computing")
        print(f"Async retrieved {len(results)} chunks")
        return results

    import asyncio
    asyncio.run(test_async())
    
    # Test 7: Query with scores
    print("\n[TEST 7] Query with scores...")
    results_with_scores = rag.query_with_scores("blockchain", k=2)
    for doc, score in results_with_scores:
        print(f"  Score: {score:.4f} - {doc.page_content[:80]}...")
    
    # Test 8: Get stats
    print("\n[TEST 8] System statistics...")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    print("\n" + "="*70)
    print("[OK] All tests passed! Save/load, async, and source storage working!")
    print("="*70)