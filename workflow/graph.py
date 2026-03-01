# workflows/graph.py
from langgraph.graph import StateGraph, END
from agents.optimizer import optimizer_agent_node
from state.schemas import MoleculeState, ExplorerState

from agents.controller_nodes import (
    sensor_node,
    comparator_node,
    actuator_node,
    analyzer_node,
    perturbation_node
)
# Initialize the Graph
workflow = StateGraph(MoleculeState)

# Add Nodes
workflow.add_node("agent", optimizer_agent_node)
# Node A: Calculates current 'r' and 'theta' from the XYZ string
workflow.add_node("sensor", sensor_node)
# Node B: Reads 'true_energy.csv', finds the target, and calculates error
workflow.add_node("comparator", comparator_node)
# Node C: Applies mathematical kinematic formula to update the XYZ
workflow.add_node("actuator", actuator_node)

# --- DEFINE LOGIC (The Conditional Edge) ---
def check_convergence(state: MoleculeState):
    """
    Evaluates if the molecule has reached the map's minimum.
    """
    # Convergence thresholds
    r_tol = 0.001    # Angstroms
    theta_tol = 0.1  # Degrees

    # Check if errors are within tolerance
    if abs(state['error_r']) < r_tol and abs(state['error_theta']) < theta_tol:
        print(f"Converged in {state['step_count']} steps!")
        return "end"

    # Safety cutoff to prevent infinite loops
    if state['step_count'] >= 4000:
        print("Reached maximum steps without converging.")
        return "end"

    return "continue"

workflow.set_entry_point("sensor")
workflow.add_edge("sensor", "comparator")
workflow.add_edge("comparator", "actuator")
workflow.add_conditional_edges(
    "actuator",check_convergence,
    {
        "continue": "sensor",
        "end": END
    }
)
workflow_app = workflow.compile()


def run_inner_optimizer(state: ExplorerState) -> dict:
    """
    Takes the Explorer's current XYZ, runs the full local optimization loop,
    and returns the finalized, minimized XYZ back to the Explorer.
    """

    print("\n[EXPLORER] Handing structure to Local Optimizer...")
    # print("State Keys: ", state.keys())

    # state['global_step_count'] = 1 # defining the first count

    # Setup the starting state for the inner loop
    inner_initial_state = MoleculeState(
        xyz_string=state['current_xyz'],
        current_r=0.0, current_theta=0.0,
        current_energy=0.0, expected_energy=0.0,
        error_r=0.0, error_theta=0.0, step_count=0
    )

    output_file = "data/optimization_movie.xyz"
    log_file = "data/optimization_log.csv"
    tracked_state = dict(inner_initial_state)

    for event in workflow_app.stream(inner_initial_state):

        # LangGraph yields a dictionary like: {'sensor': {'current_r': 1.0, ...}}
        node_name = list(event.keys())[0]
        node_data = event[node_name]

        # Update our local tracker with whatever this specific node just changed
        tracked_state.update(node_data)

        if node_name == 'comparator':
            step_count = tracked_state.get('step_count', 0)
            # Extract the metrics
            xyz = tracked_state['xyz_string']
            r = tracked_state['current_r']
            theta = tracked_state['current_theta']
            e_real = tracked_state['current_energy']
            e_map = tracked_state['expected_energy']
            err_r = tracked_state['error_r']
            err_theta = tracked_state['error_theta']

            # Save the Geometry (The Structure)
            with open(output_file, "a") as f:
                # Ensure there's a newline between XYZ blocks
                f.write(xyz.strip() + "\n")

            # Save the Numbers (The Data)
            with open(log_file, "a") as f:
                f.write(f"{state['global_step_count']},{step_count},{r:.4f},{theta:.2f},{e_real:.6f},{e_map:.6f},{err_r:.4f},{err_theta:.2f}\n")

            print(f"Logged Step {step_count} to disk.")

    # Run the inner graph until it hits END
    # final_inner_state = workflow_app.invoke(inner_initial_state)

    print("[EXPLORER] Local Optimization complete.")

    # Return the newly optimized geometry to the Explorer's state
    return {
        "current_xyz": tracked_state['xyz_string'],
        # Assuming your inner graph logs the final energy, you can pull it up here
        "current_energy": tracked_state['current_energy']
    }

def check_frustration(state: ExplorerState):
    """
    :return:
    Evaluate if the sufficient coverage for PES has been reached.
    """
    if state['frustration'] >= 3:
        print("PES kick failed - 3 Attempt")
        return 'end'
    else:
        print(f"PES kick failed - {state['frustration']} Attempt")
        return 'kick'


# --- 3. BUILD THE OUTER GRAPH (The Basin Hopper) ---
explorer_graph = StateGraph(ExplorerState)
# Node 1: Run the optimizer sub-graph
explorer_graph.add_node("local_optimization", run_inner_optimizer)
# Node 2: Check if this minimum is new or old (Updates 'frustration')
explorer_graph.add_node("check_memory", analyzer_node)
# Node 3: Apply Monte Carlo Rattle (and multiply kick_strength by 1.05 here!)
explorer_graph.add_node("apply_kick", perturbation_node)

# --- Outer Graph Wiring ---
explorer_graph.set_entry_point("local_optimization")

# After optimization, always catalog the result
explorer_graph.add_edge("local_optimization", "check_memory")

# After cataloging, the router decides: Kick or Stop?
explorer_graph.add_conditional_edges(
    "check_memory",
    check_frustration,
    {
        "kick": "apply_kick",
        "end": END
    }
)

# If we kicked it, send it back to the optimizer to slide down the new valley
explorer_graph.add_edge("apply_kick", "local_optimization")

# Compile the Master App
master_app = explorer_graph.compile()


