"""LLM and embedding model factories."""

from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek
from langchain_ollama import OllamaEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_llm() -> ChatDeepSeek:
    config.require_env("DEEPSEEK_API_KEY")
    return ChatDeepSeek(model=config.LLM_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
