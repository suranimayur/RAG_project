"""Main Entry Point for PRODX RAG App

This script serves as the central entry point:
- Builds the knowledge base: Ingests sample data, embeds, stores in FAISS.
- Runs the FastAPI server.
- Option to run Streamlit UI.

Usage:
  python main.py --mode build  # Build index from data/
  python main.py --mode api    # Start FastAPI server (default)
  python main.py --mode ui     # Start Streamlit UI

Business Context: Initializes the RAG system for PRODX analysis. Run 'build' once to index samples, then query via API/UI.

Requires all modules implemented.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ingestion.ingestion import DataIngester
from embeddings.embeddings import EmbeddingGenerator
from storage.storage import VectorStore
from api.api import app as fastapi_app
from analyzer.analyzer import ImpactAnalyzer  # For validation

def build_index():
    """Ingest, embed, and store sample data."""
    print("Building knowledge base from data/...")
    
    # Initialize components
    ingester = DataIngester()
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Load and chunk
    print("1. Ingesting data...")
    chunks = ingester.load_and_chunk_all()
    print(f"Loaded {len(chunks)} chunks from {len(list(ingester.data_dir.glob('*')))} files.")
    
    if not chunks:
        print("No data found. Ensure data/ has samples.")
        return False
    
    # Embed
    print("2. Generating embeddings...")
    embedded_docs = embedding_gen.embed_docs(chunks)
    print(f"Embedded {len(embedded_docs)} docs (dim: {len(embedded_docs[0]['embedding'])}).")
    
    # Store
    print("3. Building vector store...")
    vector_store.add_documents(embedded_docs)
    vector_store.save()
    print(f"Index saved: {vector_store.index.ntotal} vectors.")
    
    # Validate analyzer
    analyzer = ImpactAnalyzer()
    test_report = analyzer.analyze_dependency_update('cryptography', '3.2.0', '3.4.8')
    print("4. Analyzer test passed. Sample risk:", test_report['risk_level'])
    
    print("Knowledge base built successfully!")
    return True

def run_api():
    """Run FastAPI server."""
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, reload=True)
    return True

def run_ui():
    """Run Streamlit UI."""
    import subprocess
    subprocess.run(["streamlit", "run", "streamlit_ui.py"])
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRODX RAG App Entry Point")
    parser.add_argument("--mode", choices=["build", "api", "ui"], default="api",
                        help="Mode: build index, run API, or UI")
    
    args = parser.parse_args()
    
    success = False
    if args.mode == "build":
        success = build_index()
    elif args.mode == "api":
        success = run_api()
    elif args.mode == "ui":
        success = run_ui()
    
    sys.exit(0 if success else 1)