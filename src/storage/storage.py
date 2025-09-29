"""Storage Module for PRODX RAG App

This module manages the vector database using FAISS for efficient similarity search.
Supports adding embedded documents, persistence (save/load index to disk), and basic retrieval.

Key Features:
- FAISS IndexFlatL2 for cosine similarity (simple, fast for <1M vectors).
- Persistence to 'index.faiss' and metadata to JSON.
- Business Context: Stores embeddings of PRODX artifacts for quick retrieval on queries about framework, updates, impacts.

Usage:
  from src.storage.storage import VectorStore
  store = VectorStore()
  store.add_documents(embedded_docs)  # from embeddings
  store.save('index.faiss')
  results = store.search(query_embedding, k=5)

Requires: faiss-cpu (or gpu), numpy.
"""

import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path


class VectorStore:
    """FAISS-based vector storage for RAG knowledge base."""

    def __init__(self, dimension: int = 384, index_path: str = 'index.faiss', metadata_path: str = 'metadata.json'):
        """
        Initialize the vector store.
        
        Args:
            dimension: Embedding dimension (default 384 for MiniLM).
            index_path: Path to save/load FAISS index.
            metadata_path: Path to save/load metadata JSON.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.metadata = []  # List of metadata dicts corresponding to index vectors
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.is_loaded = False

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """
        Add embedded documents to the index.
        
        Args:
            docs: List of dicts with 'embedding' (list/float), 'content', 'metadata'.
        """
        embeddings = np.array([doc['embedding'] for doc in docs], dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend([doc['metadata'] for doc in docs])
        self.is_loaded = True
        print(f"Added {len(docs)} documents to index. Total vectors: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, int]]:
        """
        Perform similarity search.
        
        Args:
            query_embedding: np.array of shape (dimension,).
            k: Number of nearest neighbors.
        
        Returns:
            List of (distance, index) tuples; lower distance = more similar.
        """
        if not self.is_loaded:
            raise ValueError("Index not loaded. Call add_documents or load first.")
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return list(zip(distances[0], indices[0]))

    def get_top_docs(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents with scores.
        
        Returns:
            List of dicts: {'content': str, 'metadata': dict, 'score': float}.
        """
        results = self.search(query_embedding, k)
        top_docs = []
        for score, idx in results:
            doc = {
                'content': self.metadata[idx].get('content', ''),  # Note: metadata doesn't have content; adjust if needed
                'metadata': self.metadata[idx],
                'score': float(score)  # Lower is better for L2
            }
            top_docs.append(doc)
        return top_docs

    def save(self) -> None:
        """Save index and metadata to disk."""
        if self.index_path.exists():
            print(f"Overwriting {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        print(f"Saved index to {self.index_path} (vectors: {self.index.ntotal}) and metadata to {self.metadata_path}")

    def load(self) -> None:
        """Load index and metadata from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file {self.index_path} not found.")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.is_loaded = True
        print(f"Loaded index with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")

    def clear(self) -> None:
        """Clear the index."""
        self.index.reset()
        self.metadata = []
        self.is_loaded = False


# Example usage (for testing)
if __name__ == "__main__":
    store = VectorStore()
    # Simulate embedded docs
    sample_embeddings = np.random.random((3, 384)).astype('float32')
    sample_docs = [
        {'embedding': sample_embeddings[0].tolist(), 'metadata': {'source': 'sample1.txt'}},
        {'embedding': sample_embeddings[1].tolist(), 'metadata': {'source': 'sample2.txt'}},
        {'embedding': sample_embeddings[2].tolist(), 'metadata': {'source': 'sample3.txt'}}
    ]
    store.add_documents(sample_docs)
    query_emb = np.random.random(384).astype('float32')
    results = store.search(query_emb, k=2)
    print(f"Search results: {results}")
    store.save()