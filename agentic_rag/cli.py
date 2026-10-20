"""Command-line interface for ingestion and interactive chat."""

from __future__ import annotations

import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from . import config, graph, vectorstore


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _print_update(node: str, update: dict) -> None:
    messages = update.get("messages")
    if not messages:
        return

    message = messages[-1]
    if isinstance(message, AIMessage):
        if message.tool_calls:
            names = ", ".join(call["name"] for call in message.tool_calls)
            print(f"[{node}] calling: {names}")
        elif message.content:
            print(f"\n{message.content}\n")
    elif isinstance(message, ToolMessage):
        preview = str(message.content).strip().replace("\n", " ")[:120]
        print(f"[{node}] retrieved ({len(str(message.content))} chars): {preview}...")
    elif isinstance(message, HumanMessage) and node == "rewrite_question":
        print(f"[{node}] rewritten question: {message.content}")


def ask(graph, question: str, config: dict) -> None:
    print(f"\nYou: {question}")
    for update in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="updates",
    ):
        for node, node_update in update.items():
            _print_update(node, node_update)


def cmd_ingest(sources: list[str]) -> None:
    vectorstore.build_index(sources or config.DEFAULT_SOURCES)


def cmd_chat(question: str) -> None:
    graph_ = graph.build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "default"}}
    vectorstore.get_vectorstore()  # ensure the index exists before the first query
    ask(graph_, question, cfg)


def interactive() -> None:
    graph_ = graph.build_graph(MemorySaver())
    thread_id = uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    vectorstore.get_vectorstore()  # ensure the index exists before the first query

    print("Agentic RAG chat (multi-turn).")
    print("Commands: 'exit' to quit, 'reset' to start a new conversation.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            thread_id = uuid4().hex
            cfg = {"configurable": {"thread_id": thread_id}}
            print("(conversation reset)")
            continue

        ask(graph_, question, cfg)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        interactive()
    elif args[0] == "ingest":
        cmd_ingest(args[1:])
    elif args[0] == "chat":
        if len(args) < 2:
            print('usage: python -m agentic_rag chat "your question here"')
            sys.exit(1)
        cmd_chat(" ".join(args[1:]))
    else:
        print('usage: python -m agentic_rag [ingest [sources...] | chat "question"]')
        sys.exit(1)
