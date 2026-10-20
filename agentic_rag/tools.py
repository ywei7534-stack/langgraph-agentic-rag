"""Tools available to the agent."""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime

from langchain.tools import tool

from . import config, vectorstore


@tool
def retrieve_documents(query: str) -> str:
    """Search the local knowledge base and return relevant chunks with source URLs."""
    docs = vectorstore.get_retriever().invoke(query)
    if not docs:
        return "No relevant documents found."

    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{index}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


@tool
def search_web(query: str) -> str:
    """Search the live web and return current information."""
    from tavily import TavilyClient

    config.require_env("TAVILY_API_KEY")
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=3, search_depth="basic")

    results = response.get("results", [])
    if not results:
        return "No web results found."

    parts = []
    for result in results:
        parts.append(
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Content: {result.get('content', '')}"
        )
    return "\n\n".join(parts)


@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression, for example '3 * (4 + 2)'."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate expression: {exc}"


def get_tools() -> list:
    tools = [retrieve_documents, get_current_time, calculate]
    if os.environ.get("TAVILY_API_KEY"):
        tools.append(search_web)
    return tools
