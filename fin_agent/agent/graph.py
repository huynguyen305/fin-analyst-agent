"""LangGraph definition (High-Level Design, section 3)."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from fin_agent.agent.nodes.analysis import (
    critic, make_lens, make_qa, make_recommender, route_after_critic, route_after_skills,
)
from fin_agent.agent.nodes.data import gather, load_skills, quantify, route_after_gather
from fin_agent.agent.nodes.intake import make_intake, route_after_intake
from fin_agent.agent.nodes.review import review, route_to_review, sync
from fin_agent.agent.state import AgentState
from fin_agent.llm.base import LLMProvider


def build_graph(llm: LLMProvider, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("intake", make_intake(llm))
    g.add_node("gather", gather)
    g.add_node("quantify", quantify)
    g.add_node("load_skills", load_skills)
    g.add_node("lens", make_lens(llm))
    g.add_node("critic", critic)
    g.add_node("recommend", make_recommender(llm))
    g.add_node("qa", make_qa(llm))
    g.add_node("review", review)
    g.add_node("sync", sync)

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_after_intake, ["gather", "intake"])
    g.add_conditional_edges("gather", route_after_gather, {"quantify": "quantify", "end": END})
    g.add_edge("quantify", "load_skills")
    g.add_conditional_edges("load_skills", route_after_skills, ["qa", "lens"])
    g.add_edge("lens", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["lens", "recommend"])
    g.add_conditional_edges("recommend", route_to_review, {"review": "review", "end": END})
    g.add_conditional_edges("qa", route_to_review, {"review": "review", "end": END})
    g.add_edge("review", "sync")
    g.add_edge("sync", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
