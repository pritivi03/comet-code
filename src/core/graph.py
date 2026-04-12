"""Builds and compiles the Comet LangGraph agent graph."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from core.graph_state import AgentState
from core.nodes import (
    make_execute_tools_node,
    make_call_llm_node,
    route_after_tools,
    route_on_response_type,
)


def build_agent_graph(
    llm,
    on_event=None,
):
    """Compile and return the agent StateGraph.

    The graph is rebuilt each run_task() call so the on_event callback is
    captured fresh in the call_llm node closure.

    Graph shape:
        START → call_llm → route_on_response_type
                    ├── "execute_tools" → execute_tools → call_llm  (loop)
                    └── "end"           → END  (type: "final" lands here directly)
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("call_llm", make_call_llm_node(llm, on_event))
    builder.add_node("execute_tools", make_execute_tools_node(on_event))

    # Entry point
    builder.set_entry_point("call_llm")

    # After call_llm: branch on response type
    # "final" → END, "tool_calls" → execute_tools, "retry" → call_llm
    builder.add_conditional_edges(
        "call_llm",
        route_on_response_type,
        {
            "execute_tools": "execute_tools",
            "retry": "call_llm",
            "end": END,
        },
    )

    # After tools: either continue or end on budget/attempt failure.
    builder.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "call_llm": "call_llm",
            "end": END,
        },
    )

    return builder.compile()
