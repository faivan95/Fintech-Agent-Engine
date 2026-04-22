from langgraph.graph import StateGraph, START, END
from state import AgentState
from agents.generator import generate_strategy

# Initialize the state graph with our defined shared memory
workflow = StateGraph(AgentState)

# Add our Generator agent as a node in the graph
workflow.add_node("generator", generate_strategy)

# Define the edges
workflow.add_edge(START, "generator")
workflow.add_edge("generator", END)

# Compile the graph into an executable application
app_graph = workflow.compile()