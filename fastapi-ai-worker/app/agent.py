# Contains the advanced LangGraph agent with memory and MCP integration.

import os
import json
from dotenv import load_dotenv
from typing import List, Dict, TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from tavily import TavilyClient  # type: ignore[import-untyped]

from .mcps import get_domain_whois
from .llm_selector import get_llm_instance

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

class AgentState(TypedDict):
    topic: str
    case_id: str
    model_id: str
    temperature: float
    long_term_memory: List[Dict]
    search_queries: List[str]
    search_results: List[Dict]
    synthesized_findings: str
    num_steps: int
    mcp_verification_list: List[Dict]
    verified_data: Dict

class PlannerSchema(BaseModel):
    search_queries: List[str] = Field(
        description="A list of 3-5 specific search queries for OSINT research",
        min_items=3,
        max_items=5
    )

async def planner_node(state: AgentState):
    llm = get_llm_instance(state["model_id"], state["temperature"])
    memory_str = json.dumps(state.get("long_term_memory", []), indent=2)
    prompt = f"""You are an expert OSINT researcher. Based on the topic '{state['topic']}' and your long-term memory of past approved findings, create a research plan.

Past Findings:
{memory_str}

REQUIREMENTS:
- Generate exactly 3-5 specific search queries
- Each query should be focused and actionable
- Queries should build upon existing knowledge
- Use specific terms, not generic phrases
- Format as a JSON list of strings

Generate 3-5 queries to build upon this knowledge.
"""
    structured_llm = llm.with_structured_output(PlannerSchema)
    response = await structured_llm.ainvoke(prompt)
    
    # Extract search_queries from response (handle both dict and BaseModel)
    if hasattr(response, 'search_queries'):
        search_queries = response.search_queries
    elif isinstance(response, dict):
        search_queries = response.get('search_queries', [])
    else:
        search_queries = []
    
    # Validate the response
    if not search_queries or len(search_queries) < 3:
        search_queries = [f"OSINT research on {state['topic']}", 
                         f"{state['topic']} background information",
                         f"{state['topic']} related entities"]
    
    return {"search_queries": search_queries, "num_steps": 1}

async def search_node(state: AgentState):
    search_queries = state["search_queries"]
    all_results = []
    for query in search_queries:
        try:
            response = tavily_client.search(query=query, search_depth="basic", max_results=3)
            all_results.extend(response.get("results", []))
        except Exception as e:
            all_results.append({"query": query, "error": str(e)})
    return {"search_results": all_results, "num_steps": state["num_steps"] + 1}

class SynthesisSchema(BaseModel):
    synthesized_findings: str = Field(
        description="A detailed summary of the research findings",
        min_length=50
    )

async def synthesis_node(state: AgentState):
    llm = get_llm_instance(state["model_id"], state["temperature"])
    results_str = json.dumps(state["search_results"])
    prompt = f"Synthesize these search results for the topic '{state['topic']}' into a coherent report:\n\n{results_str}"
    structured_llm = llm.with_structured_output(SynthesisSchema)
    response = await structured_llm.ainvoke(prompt)
    
    # Extract synthesized_findings from response (handle both dict and BaseModel)
    if hasattr(response, 'synthesized_findings'):
        synthesized_findings = response.synthesized_findings
    elif isinstance(response, dict):
        synthesized_findings = response.get('synthesized_findings', '')
    else:
        synthesized_findings = ''
    
    return {"synthesized_findings": synthesized_findings, "num_steps": state["num_steps"] + 1}

class MCPIdentificationSchema(BaseModel):
    mcp_tasks: List[Dict] = Field(
        description="A list of tasks for MCPs. E.g., [{'mcp_name': 'get_domain_whois', 'input': 'example.com'}]",
        default=[]
    )

async def mcp_identification_node(state: AgentState):
    llm = get_llm_instance(state["model_id"], state["temperature"])
    prompt = f"""You are a verification engine. From the report below, identify all domain names that can be verified with the 'get_domain_whois' tool.
Report:
'{state['synthesized_findings']}'
Create a list of tasks. If none, return an empty list.
"""
    structured_llm = llm.with_structured_output(MCPIdentificationSchema)
    response = await structured_llm.ainvoke(prompt)
    
    # Extract mcp_tasks from response (handle both dict and BaseModel)
    if hasattr(response, 'mcp_tasks'):
        mcp_tasks = response.mcp_tasks
    elif isinstance(response, dict):
        mcp_tasks = response.get('mcp_tasks', [])
    else:
        mcp_tasks = []
    
    return {"mcp_verification_list": mcp_tasks, "num_steps": state["num_steps"] + 1}

async def mcp_execution_node(state: AgentState):
    mcp_tasks = state.get("mcp_verification_list", [])
    verified_data = {}
    for task in mcp_tasks:
        if task.get('mcp_name') == 'get_domain_whois':
            domain = task.get('input')
            if domain:
                verified_data[domain] = get_domain_whois(domain)
    return {"verified_data": verified_data, "num_steps": state["num_steps"] + 1}

class FinalReportSchema(BaseModel):
    final_report: str = Field(
        description="The final intelligence report, updated with verified data",
        min_length=100
    )

async def update_report_node(state: AgentState):
    llm = get_llm_instance(state["model_id"], state["temperature"])
    verified_data_str = json.dumps(state.get("verified_data"), indent=2)
    prompt = f"""You are an editor. Update the draft report with the verified data from MCPs.
Draft Report:
'{state['synthesized_findings']}'
Verified Data:
{verified_data_str}
Integrate the verified facts, replacing uncertain statements.
"""
    structured_llm = llm.with_structured_output(FinalReportSchema)
    response = await structured_llm.ainvoke(prompt)
    
    # Extract final_report from response (handle both dict and BaseModel)
    if hasattr(response, 'final_report'):
        final_report = response.final_report
    elif isinstance(response, dict):
        final_report = response.get('final_report', '')
    else:
        final_report = ''
    
    return {"synthesized_findings": final_report}

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("search", search_node)
workflow.add_node("synthesis", synthesis_node)
workflow.add_node("mcp_identifier", mcp_identification_node)
workflow.add_node("mcp_executor", mcp_execution_node)
workflow.add_node("final_updater", update_report_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "search")
workflow.add_edge("search", "synthesis")
workflow.add_edge("synthesis", "mcp_identifier")
workflow.add_edge("mcp_identifier", "mcp_executor")
workflow.add_edge("mcp_executor", "final_updater")
workflow.add_edge("final_updater", END)

agent_executor = workflow.compile()