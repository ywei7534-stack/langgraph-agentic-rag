"""Central configuration for the agentic RAG package."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_MODEL = os.environ.get("AGENTIC_RAG_LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.environ.get("AGENTIC_RAG_EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
INDEX_DIR = os.environ.get(
    "AGENTIC_RAG_INDEX_DIR", str(PROJECT_ROOT / "data" / "faiss_index")
)
MAX_REWRITES = int(os.environ.get("AGENTIC_RAG_MAX_REWRITES", "2"))
RETRIEVAL_K = int(os.environ.get("AGENTIC_RAG_RETRIEVAL_K", "4"))

DEFAULT_SOURCES = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def require_env(key: str) -> str:
    """Return an environment variable or raise a clear error if it is missing."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {key}. "
            "Set it before running (for example $env:{key} = '...' in PowerShell)."
        )
    return value
