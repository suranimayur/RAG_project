# Multi-stage Dockerfile for PRODX RAG App
# Stage 1: Builder for dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir 'torch<2.1.0' --index-url https://download.pytorch.org/whl/cpu  # CPU-optimized for embeddings/LLM

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run API; override for UI with docker run -p 8501:8501 ... sh -c "streamlit run streamlit_ui.py"
CMD ["python", "main.py", "--mode", "api"]