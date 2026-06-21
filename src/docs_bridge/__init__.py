"""docs-bridge RAG core.

Shared by two images: the ingest-worker (two-pass, hash-delta ingestion) and the
docs-bridge server (hybrid search -> rerank -> MCP/REST). The ONNX/INT8 embedder
in `embed_onnx` is the single source of truth for query/doc vector parity.
"""

__version__ = "0.1.0"
