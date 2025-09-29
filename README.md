# PRODX RAG Application - Production-Grade Modular Documentation

This README provides a consolidated overview of the entire project, including file descriptions, module explanations, setup, usage, and testing. It builds on the developer instructions in `claude.md`. The app is a Retrieval-Augmented Generation (RAG) system for analyzing the ABC Bank PRODX framework, addressing business challenges like unvalidated updates causing Airflow job failures and dependency conflicts (e.g., cryptography versions across LOBs 009/003).

## Project Overview
- **Objective**: Enable natural language queries to understand PRODX workings, predict update impacts, and suggest fixes, reducing 1-3 week resolution times via ticketing/JIRA.
- **Key Features**:
  - Modular Python components for ingestion, embedding, storage, retrieval, generation, and analysis.
  - FastAPI API endpoints for queries/analysis.
  - Streamlit UI for chat/dependency/log analysis.
  - Sample datasets simulating PRODX artifacts (YAML, logs, JIRA, docs).
  - Docker support for deployment.
- **Tech Stack**: Python 3.10+, LangChain, HuggingFace (embeddings/LLM), FAISS vector DB, FastAPI, Streamlit, Pytest.
- **Business Simulation**: Samples cover API-Ingestion YAML, Airflow failure logs (schema issue), JIRA ticket (PRODX-456), PRODX doc (dependency conflicts).

## File Structure & Descriptions

### Root Files
- **[requirements.txt](requirements.txt)**: Dependencies list (36 lines). Includes LangChain, HuggingFace Transformers/Sentence-Transformers, FAISS-CPU, Boto3 (AWS), FastAPI/Uvicorn, Streamlit, Pytest, PyYAML/Pandas. Install: `pip install -r requirements.txt`. Notes for GPU: Add `faiss-gpu` and CUDA-indexed Torch.
- **[main.py](main.py)** (62 lines): Entry point script. Modes:
  - `--mode build`: Ingests/embeds/stores samples (creates `index.faiss`/`metadata.json`).
  - `--mode api`: Runs FastAPI server (`uvicorn` on port 8000).
  - `--mode ui`: Runs Streamlit (`streamlit run streamlit_ui.py` on 8501). Initializes components; uses `sys.path` for src/.
- **[streamlit_ui.py](streamlit_ui.py)** (124 lines): Streamlit web UI. Tabs:
  - **RAG Chat**: Chat input for queries (calls `/query` API; shows response/sources/confidence via Pandas).
  - **Dependency Analysis**: Inputs for package/version/LOB (calls `/analyze_dependency`; displays risks/conflicts/suggestions).
  - **Log Analysis**: Select log/update_type (calls `/analyze_log`; shows impacts/fixes/severity).
  - Sidebar: API URL config. Uses `requests` for backend calls; session state for chat history.
- **[Dockerfile](Dockerfile)** (26 lines): Multi-stage Docker build. Base: Python 3.10-slim. Stages builder (deps install) and runtime (copy code, non-root user). Exposes 8000 (API)/8501 (UI); health check on `/health`. CMD: `python main.py --mode api`. Build: `docker build -t prodx-rag .`; Run: `docker run -p 8000:8000 prodx-rag`.
- **[claude.md](claude.md)** (148 lines): Core developer instructions (from initial plan). Covers objective, architecture (Mermaid diagram), stack justification (open-source HuggingFace/FAISS/LangChain), data sources, modular tasks, setup (venv/pip), project structure tree, implementation steps, testing (Pytest/unit/E2E), deployment (Docker/ECS/SageMaker/CI-CD).

### data/ Directory (Sample Datasets - Simulating Business Problems)
These files provide PRODX context for ingestion/testing. Ingest via `main.py --mode build` or API `/ingest`.
- **[sample_yaml_wrapper.yaml](data/sample_yaml_wrapper.yaml)** (35 lines): Tenant YAML for API-Ingestion group. Configures S3 inbound/curated buckets, JSON schema (transaction_id/amount), load pattern (full_load), Glue/Redshift/Athena integration, validation (no_nulls/range_check). Simulates ingestion pipeline; command: `dataform run -d block-a -e dev -g API-Ingestion -w transaction_ingest`. Metadata: business_context='PRODX YAML for ingestion pipeline'.
- **[sample_airflow_log.txt](data/sample_airflow_log.txt)** (25 lines): Simulated Airflow DAG failure log after PRODX v2.3 update. Shows Glue ValidationException (schema mismatch: amount string vs number), Athena parser issue, EMR ComputeHub cryptography conflict (3.4.8 vs 3.2.0). Impacts: Delayed curation for block-a/LOB 009. Analysis note: Rollback or update YAML casting. References JIRA PRODX-456.
- **[sample_jira_ticket.json](data/sample_jira_ticket.json)** (29 lines): JIRA export (PRODX-456). Details Airflow failure post-v2.3 (schema evolution/dependency conflict), root cause (unvalidated Glue/Athena), steps to reproduce, proposed solutions (casting/isolated envs), stakeholders (tenants/infra), resolution (v2.3.1 patch). Labels: PRODX-Update/Airflow-Failure/Dependency-Conflict.
- **[prod_x_doc_dependency_conflicts.md](data/prod_x_doc_dependency_conflicts.md)** (52 lines): PRODX doc on Python deps in EMR/EC2/Lambda. Covers LOBs (block-a/009, block-b/003), requirements.txt pinning (cryptography examples), conflicts (e.g., 3.4.8 vs <3.3.0 causing ImportError), impacts (DAG delays/regulatory risks), best practices (virtualenvs/Lambda layers/pip check/schema evolution). Resources: AWS EMR docs/Airflow integration.

### src/ Directory (Modular Components)
Each module in `src/<module>/` has an `__init__.py` (optional, but src/__init__.py exists for package). All files include docstrings explaining purpose, usage, business context, params/returns.

- **[src/__init__.py](src/__init__.py)** (11 lines): Package initializer. Lists modules and architecture overview (decoupled for scalability).

- **src/ingestion/ingestion.py** (104 lines): Loads/chunks data (YAML/JSON/TXT/MD/logs). Classes: `Chunk` (content/metadata), `DataIngester` (_load_yaml/json/text, _chunk_text with overlap, load_file with business_context metadata, load_and_chunk_all). Supported extensions: .yaml/.json/.txt/.md/.log. Example: Loads samples, adds 500-char chunks.

- **src/embeddings/embeddings.py** (66 lines): Generates vectors using SentenceTransformer ('all-MiniLM-L6-v2', 384-dim). Class: `EmbeddingGenerator` (embed_docs batches texts, embed_single for queries, get_model_info). Prepares texts from docs; converts embeddings to lists for serialization.

- **src/storage/storage.py** (99 lines): FAISS vector DB (IndexFlatL2 for L2 similarity). Class: `VectorStore` (add_documents, search/get_top_docs, save/load/clear). Persistence: index.faiss (faiss.write_index), metadata.json (json.dump). Stats: ntotal vectors.

- **src/retrieval/retrieval.py** (92 lines): Orchestrates query embedding/search/reranking. Class: `Retriever` (init with EmbeddingGenerator/VectorStore, retrieve [search initial + optional CrossEncoder rerank], batch_retrieve, get_stats). Loads index; fetches top-k (default 5, rerank from 15).

- **src/generation/generation.py** (114 lines): LLM response with LangChain RAG chain. Class: `RAGGenerator` (HuggingFacePipeline 'distilgpt2' demo/extensible to Mistral, prompt_template for PRODX context/query, format_context, generate/generate_with_retrieval, set_retriever, get_llm_info). Formats docs into context string.

- **src/analyzer/analyzer.py** (145 lines): Impact prediction (rules-based). Class: `ImpactAnalyzer` (analyze_dependency_update [conflict check via regex, report with risks/suggestions], _parse_requirements, analyze_log_for_impact [regex patterns for schema/Athena], analyze_yaml_update [schema/patch checks], get_risk_summary). Simulated LOB reqs/update patterns from data.

- **src/api/api.py** (112 lines): FastAPI app (title "PRODX RAG API"). Endpoints: POST /ingest (upload file, ingest/embed/store), POST /query (RAG via generator, returns response/sources/confidence), POST /analyze_dependency (analyzer report), POST /analyze_log (log impacts), GET /health (status/vectors). CORS enabled; Pydantic models; initializes components as globals. Run: uvicorn src.api.api:app --reload.

### tests/ Directory
- **[test_basic.py](tests/test_basic.py)** (82 lines): Pytest suite. Fixtures: ingester/data_dir. Tests: ingestion_load_yaml, embeddings_generate, storage_add_search, retrieval_retrieve (skips if no index), generation_generate, analyzer_dependency/log, end_to_end_build. Run: `pytest tests/ -v`. Covers units/integration; requires built index for retrieval.

## Setup & Usage
1. **Environment**: `python -m venv rag_env && source rag_env/bin/activate` (Windows: Scripts\activate). `pip install -r requirements.txt`.
2. **Build Knowledge Base**: `python main.py --mode build` (ingests data/, creates index.faiss).
3. **Run API**: `python main.py --mode api` (http://localhost:8000/docs for Swagger).
4. **Run UI**: `python main.py --mode ui` (http://localhost:8501).
5. **Test**: `pytest tests/ -v` (build index first if needed).
6. **Deploy**: `docker build -t prodx-rag . && docker run -p 8000:8000 prodx-rag` (API); override CMD for UI: `docker run -p 8501:8501 prodx-rag streamlit run streamlit_ui.py`. For AWS: Push to ECR, deploy ECS/EKS with IAM for S3/EMR.

## End-to-End Example
- Build: Indexes samples (e.g., detects cryptography conflict).
- Query via UI/API: "Impact of updating cryptography to 3.4.8?" → Retrieves doc/log/JIRA/YAML, generates: "HIGH risk for LOB 003 (version <3.3.0); suggest virtualenvs in EMR. Sources: prod_x_doc..., sample_airflow_log...".
- Analyzer: Dependency check confirms conflicts/suggestions (isolate per LOB).
- Log: Analyzes sample_airflow_log → ERROR severity, fix: explicit casting.

## Extensibility & Best Practices
- **Modularity**: Components decoupled (e.g., swap FAISS for Pinecone in storage.py).
- **Production**: Add AWS (Boto3 in ingestion for S3 sync), quantization for LLM, monitoring (CloudWatch), auth (JWT in API).
- **Limitations**: Local LLM (distilgpt2) for demo; use SageMaker for scale. Tests basic; add coverage.
- **Compliance**: No PII in samples; encrypt S3 in prod.

For detailed code explanations, see docstrings in each file. Questions? Refer to claude.md or extend via modules.