# agents/optimizer.py
import json
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from tools.calc_tools import calculate_single_point, update_geometry

# 1. SETUP LLM
llm = ChatOllama(model="llama3.1", format="json", temperature=0)


def optimizer_agent_node(state):
    # reading the state
    current_xyz = state['xyz_string']

    # Default to 0.1 Angstrom if not set in memory yet
    current_step_size = state.get('step_size', 0.1)

    # Check if we are done
    if state.get('max_force', 100) < 0.05:
        return {"status": "converged"}

    # Bootstrapping: If no energy calculated yet, do it now
    if state.get('current_energy') is None:
        results = calculate_single_point(current_xyz)
        return {
            "current_energy": results['energy'],
            "max_force": results['max_force'],
            "forces": results['forces'],
            "step_size": 0.1
        }

    # Determining the move
    # Find the atom screaming the loudest (Highest Force)
    forces = np.array(state['forces'])
    magnitudes = np.linalg.norm(forces, axis=1)
    worst_atom = np.argmax(magnitudes)
    force_vec = forces[worst_atom]

    # Calculate direction (Normalized)
    force_norm = magnitudes[worst_atom]
    if force_norm < 1e-6:
        direction = np.zeros(3)
    else:
        direction = force_vec / force_norm

    # Attempting the move
    # We attempt to move by 'current_step_size'
    move_vec = direction * current_step_size

    print(f" TRIAL: Atom {worst_atom} -> Step {current_step_size:.4f}Å")

    # Create the new geometry TEMPORARILY
    trial_xyz = update_geometry(current_xyz, worst_atom, move_vec.tolist())
    trial_results = calculate_single_point(trial_xyz)

    # --- STEP 4: THE JUDGEMENT (Adaptive Logic) ---

    current_e = state['current_energy']
    trial_e = trial_results['energy']

    # CRITICAL CHECK: Did things get worse?
    if trial_e > current_e:
        # FAILURE: The step was too big. We jumped over the valley.
        new_step = current_step_size * 0.5  # <--- CUT SPEED IN HALF

        print(f" REJECT: Energy rose ({current_e:.3f} -> {trial_e:.3f})")
        print(f" BRAKING: Reducing step size to {new_step:.5f} Å")

        # We return the OLD xyz (ignoring the move) but update the step_size
        return {
            "step_size": new_step
            # Note: We do NOT return 'xyz_string', so the molecule stays put!
        }

    else:
        # SUCCESS: The energy went down. We keep this move.
        print(f" ACCEPT: Energy fell ({current_e:.3f} -> {trial_e:.3f})")
        print(f" FORCE: {state['max_force']:.3f} -> {trial_results['max_force']:.3f}")

        return {
            "xyz_string": trial_xyz,
            "current_energy": trial_e,
            "max_force": trial_results['max_force'],
            "forces": trial_results['forces'],
            "step_size": current_step_size  # Keep the same speed
        }