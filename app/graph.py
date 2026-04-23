from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.generator import generate_strategy
from agents.validator import validate_strategy

# 1. Define the Router Function
def route_validation(state: AgentState) -> str:
    """Routes the graph based on the Validator's decision."""
    if state.get("approved", False):
        print("-> Routing to: END (Strategy Approved)")
        return END
    elif state.get("revision_count", 0) >= 3:
        # Hardware/Cost Safety: Prevent infinite loops
        print("-> Routing to: END (Max Revisions Reached)")
        return END
    else:
        print("-> Routing to: GENERATOR (Strategy Rejected, revising...)")
        return "generator"

# 2. Initialize the Graph
workflow = StateGraph(AgentState)

# 3. Add Nodes (The Agents)
workflow.add_node("generator", generate_strategy)
workflow.add_node("validator", validate_strategy)

# 4. Define the Edges (The Flow)
# Start -> Generator
workflow.add_edge(START, "generator")

# Generator -> Validator
workflow.add_edge("generator", "validator")

# Validator -> Conditional Router (Generator or END)
workflow.add_conditional_edges(
    "validator",
    route_validation,
    {
        "generator": "generator",
        END: END
    }
)

# 5. Compile the Graph
app_graph = workflow.compile()