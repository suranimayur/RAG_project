"""Ingestion Module for PRODX RAG App

This module handles loading and chunking of data sources:
- YAML wrappers (PyYAML)
- JSON files (json module)
- Log files (plain text)
- Markdown docs (plain text)

Chunks text into smaller pieces for embedding (e.g., 500 chars with 50 char overlap).
Supports metadata tracking (file path, type, business context).

Usage:
  from src.ingestion.ingestion import DataIngester
  ingester = DataIngester(data_dir='data/')
  docs = ingester.load_and_chunk_all()
  # docs: List of dicts {'content': str, 'metadata': dict}

Business Context: Ingests PRODX-related artifacts (YAML schemas, Airflow logs, JIRA exports, framework docs)
to build the knowledge base for RAG queries on framework analysis and impact prediction.
"""

import os
import json
import yaml
from typing import List, Dict, Any
from pathlib import Path


class Chunk:
    """Represents a text chunk with metadata."""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content.strip()
        self.metadata = metadata.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {'content': self.content, 'metadata': self.metadata}


class DataIngester:
    """Main class for data ingestion and chunking."""

    def __init__(self, data_dir: str = 'data/', chunk_size: int = 500, overlap: int = 50):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist.")

    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(Chunk(chunk_text, metadata))
            start = end - self.overlap
            if start >= len(text):
                break
        return chunks

    def _load_yaml(self, file_path: Path) -> str:
        """Load YAML file content as formatted string."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def _load_json(self, file_path: Path) -> str:
        """Load JSON file content as formatted string."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)

    def _load_text(self, file_path: Path) -> str:
        """Load plain text or MD file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_file(self, file_path: str) -> List[Chunk]:
        """Load and chunk a single file based on extension."""
        path = self.data_dir / file_path
        if not path.exists():
            raise FileNotFoundError(f"File {file_path} not found in {self.data_dir}")

        metadata = {
            'source': file_path,
            'type': path.suffix,
            'size': os.path.getsize(path)
        }

        if path.suffix == '.yaml' or path.suffix == '.yml':
            text = self._load_yaml(path)
        elif path.suffix == '.json':
            text = self._load_json(path)
        elif path.suffix in ['.txt', '.md', '.log']:
            text = self._load_text(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        # Add business context if known (e.g., for PRODX files)
        if 'yaml_wrapper' in file_path:
            metadata['business_context'] = 'PRODX YAML for ingestion pipeline'
        elif 'airflow_log' in file_path:
            metadata['business_context'] = 'Airflow job failure log'
        elif 'jira' in file_path:
            metadata['business_context'] = 'JIRA ticket for issue resolution'
        elif 'dependency' in file_path:
            metadata['business_context'] = 'PRODX doc on infrastructure dependencies'

        return self._chunk_text(text, metadata)

    def load_and_chunk_all(self) -> List[Dict[str, Any]]:
        """Load and chunk all files in data/ directory."""
        all_docs = []
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                try:
                    chunks = self.load_file(file_path.name)
                    for chunk in chunks:
                        all_docs.append(chunk.to_dict())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return all_docs

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['.yaml', '.yml', '.json', '.txt', '.md', '.log']


# Example usage (for testing)
if __name__ == "__main__":
    ingester = DataIngester()
    docs = ingester.load_and_chunk_all()
    print(f"Loaded {len(docs)} chunks from {ingester.data_dir}")
    if docs:
        print(f"Sample chunk: {docs[0]['content'][:100]}...")