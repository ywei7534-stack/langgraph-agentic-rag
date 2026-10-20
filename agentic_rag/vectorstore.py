"""Persistent FAISS vector store."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config, loaders, models


_text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,
    chunk_overlap=100,
)


def build_index(sources: list[str]) -> FAISS:
    docs = loaders.load_sources(sources)
    if not docs:
        raise ValueError("No documents loaded from the provided sources.")

    splits = _text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, models.get_embeddings())
    Path(config.INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(config.INDEX_DIR)
    get_retriever.cache_clear()
    print(f"[index] saved {len(splits)} chunks from {len(docs)} docs to {config.INDEX_DIR}")
    return vectorstore


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    index_files = list(Path(config.INDEX_DIR).glob("*")) if Path(config.INDEX_DIR).exists() else []
    if index_files:
        return FAISS.load_local(
            config.INDEX_DIR,
            models.get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    print("[index] no index found; building from default sources...")
    return build_index(config.DEFAULT_SOURCES)


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    return get_vectorstore().as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
