from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.generator import generate_strategy
from agents.validator import validate_strategy
from agents.coder import write_code
from agents.executor import execute_backtest

def route_validation(state: AgentState) -> str:
    if state.get("approved", False):
        print("-> Routing to: CODER (Strategy Approved by Validator)")
        return "coder"
    elif state.get("revision_count", 0) >= 3:
        print("-> Routing to: END (Max Concept Revisions Reached)")
        return END
    else:
        print("-> Routing to: GENERATOR (Strategy Rejected, revising...)")
        return "generator"

def route_execution(state: AgentState) -> str:
    error = state.get("execution_error", "")
    if not error:
        print("-> Routing to: END (Execution Successful!)")
        return END
    elif state.get("code_revisions", 0) >= 5:
        print("-> Routing to: END (Max Code Revisions Reached)")
        return END
    else:
        print("-> Routing to: CODER (Execution Failed. Sending traceback...)")
        return "coder"

workflow = StateGraph(AgentState)

workflow.add_node("generator", generate_strategy)
workflow.add_node("validator", validate_strategy)
workflow.add_node("coder", write_code)
workflow.add_node("executor", execute_backtest)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "validator")
workflow.add_conditional_edges("validator", route_validation, {"coder": "coder", "generator": "generator", END: END})
workflow.add_edge("coder", "executor")
workflow.add_conditional_edges("executor", route_execution, {"coder": "coder", END: END})

# We do NOT compile the checkpointer here. We just export the workflow blueprint.