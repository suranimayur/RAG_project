"""Embeddings Module for PRODX RAG App

This module generates vector embeddings from text chunks using HuggingFace's sentence-transformers.
Supports batch processing for efficiency and model caching.

Key Features:
- Uses 'all-MiniLM-L6-v2' model: Lightweight (22M params), fast, good for semantic similarity in PRODX context.
- Integrates with ingestion chunks: Converts to 384-dim vectors.
- Business Context: Embeds PRODX docs/logs for retrieval on queries like framework updates or dependency impacts.

Usage:
  from src.embeddings.embeddings import EmbeddingGenerator
  generator = EmbeddingGenerator()
  vectors = generator.embed_docs([{'content': 'text1', 'metadata': {}}, ...])
  # vectors: List of np.array (shape: (n_docs, embedding_dim))

Requires: sentence-transformers, torch (GPU optional for faster inference).
"""

import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Handles embedding generation using HuggingFace models."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier.
            device: 'cpu' or 'cuda' for GPU acceleration.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Loaded {model_name} model (dim: {self.embedding_dim}) on {device}")

    def _prepare_texts(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Extract content from docs list."""
        return [doc['content'] for doc in docs]

    def embed_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents (chunks from ingestion).
        
        Args:
            docs: List of dicts with 'content' and 'metadata'.
        
        Returns:
            List of dicts with added 'embedding' key (np.array).
        """
        texts = self._prepare_texts(docs)
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Attach embeddings to original docs
        for i, doc in enumerate(docs):
            doc['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization if needed
        return docs

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string for query embedding."""
        return self.model.encode([text])[0]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model details for logging."""
        return {
            'model_name': self.model[0].auto_model.config.name_or_path,
            'embedding_dim': self.embedding_dim,
            'max_seq_length': self.model[0].auto_model.config.max_position_embeddings
        }


# Example usage (for testing)
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    sample_docs = [
        {'content': 'PRODX framework uses AWS EMR for compute.', 'metadata': {}},
        {'content': 'Dependency conflict in cryptography module.', 'metadata': {}}
    ]
    embedded_docs = generator.embed_docs(sample_docs)
    print(f"Embedded {len(embedded_docs)} docs. Sample embedding length: {len(embedded_docs[0]['embedding'])}")
    model_info = generator.get_model_info()
    print(f"Model: {model_info['model_name']}, Dim: {model_info['embedding_dim']}")