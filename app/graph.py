from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.generator import generate_strategy
from agents.validator import validate_strategy
from agents.coder import write_code
from agents.executor import execute_backtest # <-- NEW IMPORT

def route_validation(state: AgentState) -> str:
    """Routes based on the Risk Officer's decision."""
    if state.get("approved", False):
        print("-> Routing to: CODER (Strategy Approved)")
        return "coder"
    elif state.get("revision_count", 0) >= 3:
        print("-> Routing to: END (Max Concept Revisions Reached)")
        return END
    else:
        print("-> Routing to: GENERATOR (Strategy Rejected, revising...)")
        return "generator"

def route_execution(state: AgentState) -> str:
    """Routes based on whether the Python code crashed."""
    error = state.get("execution_error", "")
    
    if not error:
        print("-> Routing to: END (Execution Successful!)")
        return END
    elif state.get("code_revisions", 0) >= 5:
        # Hardware/Cost Safety: Prevent the AI from trying to fix impossible bugs forever
        print("-> Routing to: END (Max Code Revisions Reached. Code is irreparably broken.)")
        return END
    else:
        print("-> Routing to: CODER (Execution Failed. Sending traceback for fix...)")
        return "coder"

# Initialize Graph
workflow = StateGraph(AgentState)

# Add all 4 Nodes
workflow.add_node("generator", generate_strategy)
workflow.add_node("validator", validate_strategy)
workflow.add_node("coder", write_code)
workflow.add_node("executor", execute_backtest) # <-- NEW NODE

# Define the standard Flow
workflow.add_edge(START, "generator")
workflow.add_edge("generator", "validator")

# Conditional: Validator -> Coder | Generator | END
workflow.add_conditional_edges(
    "validator",
    route_validation,
    {
        "coder": "coder",
        "generator": "generator",
        END: END
    }
)

# Standard: The Coder ALWAYS sends its code to the Executor for testing
workflow.add_edge("coder", "executor")

# Conditional: Executor -> END | Coder
workflow.add_conditional_edges(
    "executor",
    route_execution,
    {
        "coder": "coder",
        END: END
    }
)

app_graph = workflow.compile()