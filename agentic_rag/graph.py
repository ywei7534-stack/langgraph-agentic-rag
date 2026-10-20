"""LangGraph nodes, routing, and graph assembly."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from . import config, models, prompts, tools


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rewrite_count: int


def _latest_user_question(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def prepare(state: AgentState) -> dict:
    """Reset the per-turn retry counter."""
    return {"rewrite_count": 0}


def agent(state: AgentState) -> dict:
    """Call the model with tools; it either answers or requests a tool call."""
    messages = [SystemMessage(content=prompts.SYSTEM_PROMPT)] + list(state["messages"])
    response = models.get_llm().bind_tools(tools.get_tools()).invoke(messages)
    return {"messages": [response]}


def route_agent(state: AgentState):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    if state.get("rewrite_count", 0) >= config.MAX_REWRITES:
        return "generate_answer"

    question = _latest_user_question(state["messages"])
    context = state["messages"][-1].content
    prompt = prompts.GRADE_PROMPT.format(question=question, context=context)
    result = models.get_llm().with_structured_output(prompts.GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    if result.binary_score == "yes":
        return "generate_answer"
    return "rewrite_question"


def route_after_tools(state: AgentState):
    """Grade retrieved docs; otherwise loop back to the agent for the next step."""
    last_message = state["messages"][-1]
    if getattr(last_message, "name", None) == "retrieve_documents":
        return grade_documents(state)
    return "agent"


def rewrite_question(state: AgentState) -> dict:
    question = _latest_user_question(state["messages"])
    prompt = prompts.REWRITE_PROMPT.format(question=question)
    response = models.get_llm().invoke([{"role": "user", "content": prompt}])
    return {
        "messages": [HumanMessage(content=response.content)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def generate_answer(state: AgentState) -> dict:
    question = _latest_user_question(state["messages"])
    context = state["messages"][-1].content
    prompt = prompts.GENERATE_PROMPT.format(question=question, context=context)
    response = models.get_llm().invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


def build_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("prepare", prepare)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools.get_tools()))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "prepare")
    workflow.add_edge("prepare", "agent")

    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {"tools": "tools", END: END},
    )
    workflow.add_conditional_edges("tools", route_after_tools)
    workflow.add_edge("rewrite_question", "agent")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(checkpointer=checkpointer)
