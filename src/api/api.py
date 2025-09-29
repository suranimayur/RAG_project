"""API Module for PRODX RAG App

This module provides FastAPI endpoints for the RAG application.
Integrates all modules: ingestion, embeddings, storage, retrieval, generation, analyzer.

Endpoints:
- POST /ingest: Load and index new data (e.g., upload files).
- POST /query: Natural language query for RAG response.
- POST /analyze_dependency: Analyze package update impact.
- POST /analyze_log: Analyze log for issues.
- GET /health: Health check.

Business Context: Exposes RAG as API for tenants/capability teams to query PRODX issues via natural language or targeted analysis, reducing ticketing delays.

Run with: uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000

Requires: fastapi, uvicorn.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path

# Internal imports
from src.ingestion.ingestion import DataIngester
from src.embeddings.embeddings import EmbeddingGenerator
from src.storage.storage import VectorStore
from src.retrieval.retrieval import Retriever
from src.generation.generation import RAGGenerator
from src.analyzer.analyzer import ImpactAnalyzer


app = FastAPI(title="PRODX RAG API", description="API for analyzing PRODX framework with RAG and impact tools.", version="1.0.0")

# CORS for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In prod, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components (singleton-like for demo)
ingester = DataIngester(data_dir='data/')
embedding_gen = EmbeddingGenerator()
vector_store = VectorStore(index_path='index.faiss', metadata_path='metadata.json')
retriever = Retriever(index_path='index.faiss', metadata_path='metadata.json')
generator = RAGGenerator(retriever=retriever)  # End-to-end capable
analyzer = ImpactAnalyzer(data_dir='data/')

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    k: int = 3

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict]
    confidence: Optional[float] = None

class DependencyRequest(BaseModel):
    package: str
    current_version: str
    new_version: str
    lob: Optional[str] = None

class DependencyResponse(BaseModel):
    conflicts: List[str]
    affected_lobs: List[str]
    suggestions: List[str]
    risk_level: str

class LogRequest(BaseModel):
    log_file: str
    update_type: str = "general"

class LogResponse(BaseModel):
    matches: List[str]
    impacts: List[str]
    fixes: List[str]
    severity: str

@app.post("/ingest", summary="Ingest and index data")
async def ingest_data(file: UploadFile = File(...)):
    """Upload and ingest a file (YAML, JSON, TXT, MD) to update the knowledge base."""
    if not file.filename.endswith(('.yaml', '.yml', '.json', '.txt', '.md', '.log')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Save uploaded file to data/
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Load and chunk
        chunks = ingester.load_file(file.filename)
        
        # Embed
        embedded_docs = embedding_gen.embed_docs([chunk.to_dict() for chunk in chunks])
        
        # Add to store (load if needed)
        if vector_store.index.ntotal == 0:
            vector_store.load()
        vector_store.add_documents(embedded_docs)
        vector_store.save()
        
        return {"status": "success", "indexed": len(embedded_docs), "file": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """RAG query: Get response for natural language question about PRODX."""
    try:
        response = generator.generate_with_retrieval(request.query, k=request.k)
        docs = retriever.retrieve(request.query, k=request.k)
        sources = [{"source": d['metadata'].get('source'), "score": d['score']} for d in docs]
        confidence = sum(1 / (1 + s) for s in [d['score'] for d in docs]) / len(docs) if docs else 0.5  # Simple avg
        
        return QueryResponse(response=response, sources=sources, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_dependency", response_model=DependencyResponse)
async def analyze_dependency(request: DependencyRequest):
    """Analyze impact of dependency update."""
    report = analyzer.analyze_dependency_update(
        request.package, request.current_version, request.new_version, request.lob
    )
    return DependencyResponse(
        conflicts=report['conflicts'],
        affected_lobs=report['affected_lobs'],
        suggestions=report['suggestions'],
        risk_level=report['risk_level']
    )

@app.post("/analyze_log", response_model=LogResponse)
async def analyze_log(request: LogRequest):
    """Analyze log file for impacts."""
    report = analyzer.analyze_log_for_impact(request.log_file, request.update_type)
    return LogResponse(
        matches=report['matches'],
        impacts=report['impacts'],
        fixes=report['fixes'],
        severity=report['severity']
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vectors_count": vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else 0,
        "modules": ["ingestion", "embeddings", "storage", "retrieval", "generation", "analyzer"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)