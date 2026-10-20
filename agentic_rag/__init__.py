"""Agentic RAG with persistent retrieval and a multi-tool, multi-turn agent.

Improvements over the original LangChain tutorial:
  1. Persistent FAISS index (Ollama bge-m3 embeddings) and multi-format data
     sources (URLs, local files, PDFs, directories).
  2. A multi-tool, multi-turn conversational agent with cited answers.
"""

__version__ = "0.1.0"
