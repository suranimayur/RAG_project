"""Retrieval Module for PRODX RAG App

This module orchestrates retrieval: embeds query, searches vector store, and optionally reranks results.
Integrates embeddings and storage modules for similarity-based document retrieval.

Key Features:
- Embeds user query using the same model as documents.
- Searches FAISS index for top-k similar chunks.
- Optional reranking: Uses a cross-encoder for better relevance (if enabled).
- Business Context: Retrieves relevant PRODX context (e.g., logs, docs) for queries on job failures or update impacts, e.g., "cryptography update effect".

Usage:
  from src.retrieval.retrieval import Retriever
  retriever = Retriever(embedding_model='all-MiniLM-L6-v2')
  context = retriever.retrieve("Impact of updating cryptography module?", k=5)
  # context: List of dicts {'content': str, 'metadata': dict, 'score': float}

Requires: sentence-transformers, faiss, numpy. For reranking: sentence-transformers (cross-encoder model).
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.storage.storage import VectorStore
from src.embeddings.embeddings import EmbeddingGenerator


class Retriever:
    """Handles query embedding, search, and reranking for RAG retrieval."""

    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = 'cpu',
                 use_reranker: bool = False,
                 reranker_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 index_path: str = 'index.faiss',
                 metadata_path: str = 'metadata.json'):
        """
        Initialize retriever with embedding model and vector store.
        
        Args:
            model_name: Embedding model for query.
            device: 'cpu' or 'cuda'.
            use_reranker: Enable cross-encoder reranking (slower but more accurate).
            reranker_name: Cross-encoder model if enabled.
            index_path, metadata_path: Paths for vector store.
        """
        self.embedding_generator = EmbeddingGenerator(model_name, device)
        self.vector_store = VectorStore(dimension=self.embedding_generator.embedding_dim, 
                                       index_path=index_path, metadata_path=metadata_path)
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = CrossEncoder(reranker_name, device=device)
            print(f"Loaded reranker {reranker_name} on {device}")

    def load_index(self) -> None:
        """Load the vector store index if not already loaded."""
        self.vector_store.load()

    def retrieve(self, query: str, k: int = 5, rerank_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query.
        
        Args:
            query: Natural language query string.
            k: Number of final results.
            rerank_k: Number to rerank before selecting top-k (if None, k=20).
        
        Returns:
            List of dicts: {'content': str, 'metadata': dict, 'score': float}.
        """
        self.load_index()
        
        # Embed query
        query_embedding = self.embedding_generator.embed_single(query)
        
        # Initial search
        search_k = rerank_k or k * 3  # Fetch more for reranking
        initial_results = self.vector_store.get_top_docs(query_embedding, search_k)
        
        if not self.use_reranker or len(initial_results) <= k:
            # No reranking or small set: return top-k
            return sorted(initial_results[:k], key=lambda x: x['score'])  # Sort by score (asc for L2)
        
        # Reranking with cross-encoder
        query_docs = [(query, doc['content']) for doc in initial_results]
        rerank_scores = self.reranker.predict(query_docs)
        
        # Combine and sort
        reranked = []
        for i, doc in enumerate(initial_results):
            doc['score'] = float(rerank_scores[i])  # Higher is better for cross-encoder
            reranked.append(doc)
        reranked = sorted(reranked[:rerank_k], key=lambda x: x['score'], reverse=True)[:k]
        
        return reranked

    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(q, k) for q in queries]

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval stats."""
        return {
            'total_vectors': self.vector_store.index.ntotal,
            'use_reranker': self.use_reranker,
            'embedding_dim': self.embedding_generator.embedding_dim
        }


# Example usage (for testing)
if __name__ == "__main__":
    retriever = Retriever(use_reranker=False)  # Reranker optional for speed
    # Assume index is loaded with sample data
    results = retriever.retrieve("What is the impact of PRODX update on Airflow jobs?", k=3)
    print(f"Retrieved {len(results)} docs.")
    for doc in results:
        print(f"Score: {doc['score']:.4f}, Source: {doc['metadata'].get('source', 'Unknown')}")
    stats = retriever.get_stats()
    print(f"Stats: {stats}")