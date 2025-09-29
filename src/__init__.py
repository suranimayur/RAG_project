# src package initializer for RAG app modules
"""
This package contains all core modules for the PRODX RAG application:
- ingestion: Data loading and chunking
- embeddings: Vector generation using HuggingFace
- storage: FAISS vector database management
- retrieval: Similarity search and reranking
- generation: LLM response generation with LangChain
- analyzer: Impact analysis for updates and dependencies
- api: FastAPI endpoints for queries

The architecture is modular to allow independent development and testing.
"""