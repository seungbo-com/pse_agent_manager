# workflows/graph.py
from langgraph.graph import StateGraph, END
from agents.optimizer import optimizer_agent_node
from state.schemas import MoleculeState

# Initialize the Graph
workflow = StateGraph(MoleculeState)

# Add Nodes
workflow.add_node("agent", optimizer_agent_node)
# Note: In a full build, you'd add a "ToolNode" here to execute the tool calls

# Define Logic (Edges)
def should_continue(state):
    if state['max_force'] < 0.05:
        return "end"
    return "continue"

workflow.set_entry_point("agent")

# The Control Loop
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent", # Loop back if not converged
        "end": END           # Stop if converged
    }
)

app = workflow.compile()