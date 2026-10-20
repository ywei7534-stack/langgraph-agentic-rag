"""Document loaders for URLs, local files, PDFs, and directories."""

from __future__ import annotations

import os
from pathlib import Path

import bs4
import requests
from langchain_core.documents import Document


SUPPORTED_EXTS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".pdf",
    ".py",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".html",
}


def load_web_page(url: str) -> list[Document]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


def load_local_file(path: str | Path) -> list[Document]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.resolve().as_uri()})]


def _iter_local_files(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if Path(filepath).suffix.lower() in SUPPORTED_EXTS:
                yield filepath


def load_sources(sources: list[str]) -> list[Document]:
    docs: list[Document] = []
    for source in sources:
        if source.startswith(("http://", "https://")):
            docs.extend(load_web_page(source))
        elif os.path.isdir(source):
            for filepath in _iter_local_files(source):
                docs.extend(load_local_file(filepath))
        elif os.path.isfile(source):
            docs.extend(load_local_file(source))
        else:
            print(f"[warn] skipped, not a URL/file/dir: {source}")
    return docs
