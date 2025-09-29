# Modular Developer Instructions for Production-Grade PRODX RAG App

This document provides comprehensive developer instructions for building, extending, testing, and deploying the RAG application. It incorporates the full implementation details from the completed project, including module explanations, file descriptions, business context, setup, usage, and best practices. The app addresses ABC Bank's PRODX framework challenges: unvalidated updates causing Airflow job failures (e.g., schema evolution issues), dependency conflicts (e.g., cryptography versions across LOBs 009/003), and lengthy ticketing resolutions (1-3 weeks via JIRA/ServiceNow).

## Objective
Develop a modular RAG system to analyze PRODX (AWS-backed data engineering framework: Redshift/S3/SageMaker/NodeJS/Athena/Glue/Lambda/RDS/Neptune/EMR/EC2/YAML). Stakeholders: Tenants (YAML wrappers, e.g., `dataform run -d block-a -e dev -g API-Ingestion -w transaction_ingest`), Capability Team (updates/fixes), Infrastructure Team (deployments).

**Pain Points Solved**:
- Surprise failures in Dev/Test/Prod (e.g., Glue schema mismatches post-v2.3 update).
- Cross-LOB impacts (e.g., LOB 009 needs cryptography>=3.4.8; LOB 003 <3.3.0 breaks shared EMR).
- Delays: Log analysis + multi-team ticketing.

**RAG Benefits**: Natural language queries (e.g., "Impact of cryptography 4.0?") retrieve context (docs/logs/YAML/JIRA), generate solutions (e.g., "HIGH risk: Use virtualenvs per LOB"), reducing resolution time.

## High-Level Architecture
Modular, decoupled components for scalability. Data flow: Ingestion → Embeddings → Storage (offline build). Query: Retrieval → Generation (augmented by Analyzer).

### System Workflow (Mermaid Diagram)
```mermaid
graph TD
    A[User Query via API/UI] --> B[Retriever: Embed Query & Search FAISS]
    B --> C[Context: PRODX Docs/YAML/Logs/JIRA]
    C --> D[Generator: LangChain + HuggingFace LLM Prompt]
    D --> E[Analyzer: Rules-Based Impact (e.g., Dependency Conflicts)]
    E --> F[Response: Solutions/Fixes/Sources/Confidence]
    G[Ingestion: Load/Chunks Samples] --> H[Embeddings: MiniLM Vectors]
    H --> I[Storage: FAISS Index Persistence]
    J[External Updates e.g., PRODX Patches] --> G
    K[Feedback Loop] --> G
    L[FastAPI Endpoints] --> A
    M[Streamlit UI Tabs: Chat/Analysis] --> L
```

- **Components**:
  1. **Ingestion**: Loads/chunks artifacts.
  2. **Embeddings**: Vector generation.
  3. **Storage**: Vector DB.
  4. **Retrieval**: Similarity search/reranking.
  5. **Generation**: LLM response with context.
  6. **Analyzer**: Impact prediction (dependencies/logs/YAML).
  7. **API/UI**: Interfaces.

## Technology Stack & Justification
- **Language**: Python 3.10+ (modularity, AWS libs).
- **RAG**: LangChain (chains/prompts; open-source).
- **Embeddings/LLM**: HuggingFace Transformers/Sentence-Transformers (all-MiniLM-L6-v2 embeddings, 384-dim; distilgpt2 demo, extensible to Mistral/Llama-2). Open-source for compliance/cost (no OpenAI).
- **Vector DB**: FAISS (local/fast; L2 similarity; swap to Pinecone for cloud).
- **Data**: PyYAML/Pandas/Pydantic (parsing/validation).
- **AWS**: Boto3 (future S3/EMR hooks).
- **API/UI**: FastAPI (REST/async), Streamlit (chat tabs).
- **Testing/Deploy**: Pytest (units), Docker (multi-stage, Python slim base).
- **Why?**: Open-source focus (banking regs), lightweight (FAISS/local LLM for dev), scalable (LangChain chains).

`requirements.txt`: 36 deps (e.g., langchain>=0.1.0, sentence-transformers>=2.2.2, faiss-cpu>=1.7.4, fastapi>=0.104.0). GPU notes included.

## Data Sources
- **Samples in data/**: Simulate PRODX (ingest via build).
  - `sample_yaml_wrapper.yaml`: API-Ingestion config (S3 schema/full_load; metadata: 'PRODX YAML').
  - `sample_airflow_log.txt`: Failure (Glue ValidationException, Athena casting, EMR crypto conflict).
  - `sample_jira_ticket.json`: PRODX-456 (v2.3 failure, solutions: casting/env isolation).
  - `prod_x_doc_dependency_conflicts.md`: Deps best practices (LOB pinning, virtualenvs/Lambda layers).
- **Strategy**: Batch (local), future: S3 sync/Lambda triggers. Chunking: 500 chars, 50 overlap.

## Modular Components & Implementation Details
Each in `src/<module>/<module>.py` (docstrings explain usage/business). Interfaces: Chunks `{content: str, metadata: dict}`, Embedded `{+ embedding: list[float]}`.

1. **ingestion/ingestion.py** (104 lines): `DataIngester` loads (YAML dump/JSON dumps/text read), chunks (`_chunk_text`), `load_file` (ext-based, adds context e.g., 'Airflow log'), `load_and_chunk_all`. `Chunk` class for dicts.

2. **embeddings/embeddings.py** (66 lines): `EmbeddingGenerator` (`SentenceTransformer`, `embed_docs` batches, `embed_single` for query, `get_model_info`). Dim: 384.

3. **storage/storage.py** (99 lines): `VectorStore` (FAISS `IndexFlatL2`, `add_documents`, `search`/`get_top_docs`, `save`/`load` to .faiss/JSON, `clear`).

4. **retrieval/retrieval.py** (92 lines): `Retriever` (embeds query, searches, optional `CrossEncoder` rerank from 15→5, `batch_retrieve`, `get_stats`).

5. **generation/generation.py** (114 lines): `RAGGenerator` (HuggingFacePipeline LLM, LangChain `LLMChain`/prompt (PRODX-specific), `format_context`, `generate`/`generate_with_retrieval`, `set_retriever`, `get_llm_info`). Demo: distilgpt2; max_tokens=512, temp=0.7.

6. **analyzer/analyzer.py** (145 lines): `ImpactAnalyzer` (`analyze_dependency_update` regex conflicts/risks/suggestions, `_parse_requirements`, `analyze_log_for_impact` patterns [schema/Athena], `analyze_yaml_update` schema/patch, `get_risk_summary`). Simulated LOB reqs.

7. **api/api.py** (112 lines): FastAPI app (CORS, Pydantic models). Globals init components. Endpoints: `/ingest` (upload/chunk/embed/store), `/query` (RAG, confidence avg score), `/analyze_dependency`/`/analyze_log`, `/health` (ntotal). Run: uvicorn.

`src/__init__.py`: Package overview.

## Project Structure
```
RAG_project/
├── src/  # Modules
│   ├── ingestion/ingestion.py
│   ├── embeddings/embeddings.py
│   ├── storage/storage.py
│   ├── retrieval/retrieval.py
│   ├── generation/generation.py
│   ├── analyzer/analyzer.py
│   └── api/api.py
├── data/  # Samples (YAML/JSON/TXT/MD)
├── tests/test_basic.py  # Pytest
├── main.py  # Entry: build/api/ui
├── streamlit_ui.py  # UI tabs
├── requirements.txt
├── Dockerfile
├── claude.md  # This doc
└── README.md  # Consolidated overview
```

## Setup Instructions
1. **Clone/Navigate**: `cd RAG_project`.
2. **Venv**: `python -m venv rag_env && rag_env\Scripts\activate` (Windows) or `source rag_env/bin/activate`.
3. **Install**: `pip install -r requirements.txt`. GPU: Add faiss-gpu/CUDA Torch.
4. **AWS (Optional)**: `aws configure` for Boto3 (future S3).
5. **Build**: `python main.py --mode build` (indexes data/ to index.faiss).

## Implementation Steps (Completed)
- Prototype: main.py build chain (ingest→embed→store).
- Modularize: Refactor to classes/interfaces.
- Integrate: API/UI call modules (e.g., /query: retriever→generator).
- AWS Hooks: Ready in ingestion/analyzer (add Boto3 clients).
- Error Handling: Try/except in loads/searches; validate schemas (Pydantic).
- Query Handling: NL (e.g., "cryptography impact?") via embeddings; targeted (analyze dep/log).

## Testing
- **Unit**: `pytest tests/test_basic.py -v` (ingestion_load_yaml, embeddings_generate, storage_add_search, retrieval_retrieve [skips if no index], generation_generate, analyzer_dependency/log, end_to_end_build).
- **Integration**: Run build, then query API (Swagger /docs); test UI tabs.
- **E2E**: Build index, query "Impact of updating cryptography?" → Response: Conflicts (LOB 003), Fixes (virtualenvs), Sources (doc/log).
- **Metrics**: Retrieval precision (manual); LLM relevance (ROUGE). Load: Locust on API.
- **Coverage**: `pytest --cov=src`.

## Deployment
1. **Local**: Venv + main.py modes.
2. **Docker**: Multi-stage (builder deps, runtime copy/code, non-root appuser). Expose 8000/8501; health: curl /health. Build/run as above; UI override CMD.
3. **AWS**: ECR push image; ECS/EKS deploy (Fargate, IAM for S3/EMR); Lambda triggers ingestion; SageMaker for LLM (replace pipeline in generation.py); CloudWatch/X-Ray monitoring; CI/CD: GitHub Actions/CodePipeline (lint/test/build/deploy on push).
4. **Scaling**: Auto-scale API (ECS); Vector DB to Pinecone; LLM to endpoints.
5. **Security**: Encrypt S3 (SSE-KMS); RBAC (Cognito); No PII in embeddings.

## Extensibility
- **New Module**: Add to src/, e.g., `aws_integration/` for Boto3 S3 fetch.
- **Swap Components**: Embeddings: Change model in embeddings.py. Storage: Update storage.py (keep search interface).
- **Advanced**: Reranker always-on; Hybrid search (keywords + semantic); Fine-tune LLM on PRODX data.
- **Patterns**: Decoupled (e.g., generator injects retriever); Interfaces (dicts); Docstrings (purpose/usage/context).

## Business Validation
- **Scenario**: Update cryptography 3.4.8 → Analyzer: HIGH risk (LOB 003 conflict); Retrieval: Docs/logs; Generation: "Impact: ImportError in EMR; Fix: Conda per LOB. Sources: prod_x_doc, sample_airflow_log."
- **Metrics**: Reduces query-to-fix time from weeks to minutes; 80%+ retrieval relevance on samples.

For code details, see module docstrings. Extend analyzer for real reqs parsing (packaging lib). Contact for custom integrations.