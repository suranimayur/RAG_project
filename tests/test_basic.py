"""Basic Tests for PRODX RAG App

This file contains pytest unit tests for core modules.
Run with: pytest tests/test_basic.py -v

Tests cover:
- Ingestion: Load and chunk sample data.
- Embeddings: Generate vectors.
- Storage: Add and search.
- Retrieval: Query similarity.
- Generation: Mock LLM response.
- Analyzer: Dependency conflict detection.

Business Context: Ensures modular components work together for reliable PRODX analysis.

Requires: pytest.
"""

import pytest
import numpy as np
from pathlib import Path

# Internal imports (add src to path for testing)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ingestion.ingestion import DataIngester, Chunk
from embeddings.embeddings import EmbeddingGenerator
from storage.storage import VectorStore
from retrieval.retrieval import Retriever
from generation.generation import RAGGenerator
from analyzer.analyzer import ImpactAnalyzer


@pytest.fixture
def sample_data_dir():
    """Fixture for data directory."""
    return Path(__file__).parent.parent / 'data'

@pytest.fixture
def ingester(sample_data_dir):
    """Fixture for ingester."""
    return DataIngester(data_dir=str(sample_data_dir))

def test_ingestion_load_yaml(ingester):
    """Test loading a YAML file."""
    chunks = ingester.load_file('sample_yaml_wrapper.yaml')
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert 'ingest_config' in chunks[0].content  # Check for YAML content
    assert 'business_context' in chunks[0].metadata

def test_embeddings_generate(ingester):
    """Test embedding generation."""
    chunks = ingester.load_and_chunk_all()
    if not chunks:
        pytest.skip("No sample data available")
    
    generator = EmbeddingGenerator()
    embedded = generator.embed_docs(chunks)
    assert len(embedded) == len(chunks)
    assert 'embedding' in embedded[0]
    assert len(embedded[0]['embedding']) == 384  # MiniLM dim

def test_storage_add_search():
    """Test vector store add and search."""
    store = VectorStore()
    # Sample embedded docs
    sample_embeds = [np.random.random(384).tolist() for _ in range(3)]
    sample_docs = [{'embedding': emb, 'metadata': {'source': f'test{i}'}} for i, emb in enumerate(sample_embeds)]
    store.add_documents(sample_docs)
    assert store.index.ntotal == 3
    
    query_emb = np.random.random(384)
    results = store.search(query_emb, k=2)
    assert len(results) == 2
    assert all(isinstance(r[0], np.float32) for r in results)

def test_retrieval_retrieve():
    """Test retriever (requires loaded index)."""
    retriever = Retriever(use_reranker=False)
    # For test, assume index built; mock or skip if not
    try:
        retriever.load_index()
        results = retriever.retrieve("test query", k=1)
        assert len(results) >= 1
        assert 'score' in results[0]
    except FileNotFoundError:
        pytest.skip("Index not built; run python main.py --mode build first")

def test_generation_generate():
    """Test generator with mock docs."""
    generator = RAGGenerator(llm_model='distilgpt2')
    sample_docs = [
        {'content': 'Test context about PRODX.', 'metadata': {'source': 'test'}, 'score': 0.1}
    ]
    response = generator.generate("Test query", sample_docs)
    assert isinstance(response, str)
    assert len(response) > 10  # Basic check

def test_analyzer_dependency():
    """Test dependency analysis."""
    analyzer = ImpactAnalyzer()
    report = analyzer.analyze_dependency_update('cryptography', '3.2.0', '3.4.8', lob='009')
    assert 'conflicts' in report
    assert report['risk_level'] == 'HIGH'
    assert len(report['suggestions']) > 0

def test_analyzer_log():
    """Test log analysis."""
    analyzer = ImpactAnalyzer()
    report = analyzer.analyze_log_for_impact('sample_airflow_log.txt', 'schema_evolution')
    assert 'impacts' in report
    assert report['severity'] == 'ERROR'

def test_end_to_end_build(ingester):
    """Basic E2E: Ingest to embed."""
    chunks = ingester.load_and_chunk_all()
    if chunks:
        generator = EmbeddingGenerator()
        embedded = generator.embed_docs(chunks)
        store = VectorStore()
        store.add_documents(embedded)
        assert store.index.ntotal == len(embedded)
    else:
        pytest.skip("No data for E2E test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])