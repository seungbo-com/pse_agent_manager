# agents/controller_nodes.py
import numpy as np
import pandas as pd
from ase.io import read, write
from io import StringIO
from numpy.linalg import norm
from state.schemas import MoleculeState, ExplorerState
from tools.calc_tools import calculate_single_point
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Load the Map
PES_MAP = pd.read_csv("./data/true_energy.csv")
TARGET_ROW = PES_MAP.loc[PES_MAP['Energy'].idxmin()]
TARGET_R = TARGET_ROW['Bond_Length']
TARGET_THETA = TARGET_ROW['Bond_Angle']

def sensor_node(state: MoleculeState) -> dict:
    """
    SENSE: Measures geometry AND queries the real physics calculator for Energy.
    """
    step = state.get('step_count', 0)
    print(f"\n{'=' * 40}\n--- STEP {step} ---")

    xyz_str = state['xyz_string']
    atoms = read(StringIO(xyz_str), format="xyz")
    pos = atoms.get_positions(); atom_num = atoms.get_atomic_numbers()
    o_idx = np.where(atom_num==8)[0];h_idx = np.where(atom_num==1)[0]
    O, H1, H2 = pos[o_idx[0]], pos[h_idx[0]], pos[h_idx[1]]

    # 1. Measure Geometry
    v1, v2 = H1 - O, H2 - O
    r1, r2 = norm(v1), norm(v2)
    current_r = (r1 + r2) / 2.0

    cos_theta = np.dot(v1, v2) / (r1 * r2)
    current_theta = np.degrees(
        np.arccos(np.clip(cos_theta, -1.0, 1.0))
    )

    # 2. Measure REAL Energy (The Altimeter)
    calc_data = calculate_single_point(xyz_str)
    real_energy = calc_data['energy']

    return {
        "current_r": current_r,
        "current_theta": current_theta,
        "current_energy": real_energy
    }


def comparator_node(state: MoleculeState) -> dict:
    """
    COMPARE: Checks geometric error to target, AND validates Real vs Map Energy.
    """
    curr_r = state['current_r']
    curr_theta = state['current_theta']
    real_energy = state['current_energy']

    # 1. Geometric Error to Target
    error_r = TARGET_R - curr_r
    error_theta = TARGET_THETA - curr_theta

    # 2. Sanity Check: What did the map say the energy should be right here?
    # We find the closest point on the grid.
    # (Divide theta by 100 to roughly normalize the scale difference between Å and Degrees)
    distances = ((PES_MAP['Bond_Length'] - curr_r) ** 2 +
                 ((PES_MAP['Bond_Angle'] - curr_theta) / 100.0) ** 2)
    closest_idx = distances.idxmin()
    expected_energy = PES_MAP.loc[closest_idx, 'Energy']
    energy_diff = real_energy - expected_energy

    # --- TERMINAL READOUT ---
    print(f"Position : r={curr_r:.3f} Å, θ={curr_theta:.1f}°")
    print(f"Target   : r={TARGET_R:.3f} Å, θ={TARGET_THETA:.1f}°")
    print(f"Error    : Δr={error_r:+.3f} Å, Δθ={error_theta:+.1f}°")
    print("-" * 40)
    print(f"MAP Energy  : {expected_energy:.4f} eV (Expected from CSV)")
    print(f"REAL Energy : {real_energy:.4f} eV (Calculated just now)")
    print(f"Delta (E)   : {energy_diff:+.4f} eV")

    if abs(energy_diff) > 0.5:
        print("WARNING: Real energy and Map energy strongly disagree! Check your calculator/box settings.")

    return {
        "error_r": error_r,
        "error_theta": error_theta,
        "expected_energy": expected_energy
    }


def actuator_node(state: MoleculeState) -> dict:
    """
    ACTUATE: Applies exact spherical kinematics using ASE's built-in
    geometry solvers to prevent tangent-overshoot and pendulum oscillations.
    """
    gain = 0.3

    # 1. Calculate the exact new coordinates we want for THIS step
    new_r = state['current_r'] + (state['error_r'] * gain)
    new_theta = state['current_theta'] + (state['error_theta'] * gain)

    # 2. Load current atoms
    atoms = read(StringIO(state['xyz_string']), format="xyz")

    # 3. Apply exact transformations (O is 0, H1 is 1, H2 is 2)
    # set_distance(atom1, atom2, new_dist, fix=0) means keep atom 0 (Oxygen) locked in place
    atoms.set_distance(0, 1, new_r, fix=0)  # Update O-H1 bond
    atoms.set_distance(0, 2, new_r, fix=0)  # Update O-H2 bond

    # set_angle(atom1, vertex_atom, atom3, new_angle)
    atoms.set_angle(1, 0, 2, new_theta)  # Update H1-O-H2 angle securely

    # 4. Save back to XYZ string
    out_stream = StringIO()
    write(out_stream, atoms, format="xyz")

    return {
        "xyz_string": out_stream.getvalue(),
        "step_count": state.get('step_count', 0) + 1
    }

# SETUP LLM FOR THE OUTER GRAPH
supervisor_llm = ChatOllama(model="llama3.1", format="json", temperature=0)


def analyzer_node(state: ExplorerState) -> dict:
    print("\n[SUPERVISOR] LLM is evaluating the landscape...")

    current_e = state.get('current_energy', 0.0)
    current_xyz = state.get('current_xyz', '')
    current_frustration = state.get('frustration', 0)
    discovered_valleys = len(state.get('discovered_minima', []))
    history = "\n".join(
        state.get('history_log', [])[-3:])
    minima_list = state.get('discovered_minima', [])
    is_new_valley = True
    for m in minima_list:
        # If the energy is within 0.01 eV of a known valley, it's a duplicate!
        if abs(m['energy'] - current_e) < 0.01:
            is_new_valley = False
            break

    if is_new_valley:
        # Catalog the new discovery!
        minima_list.append({"energy": current_e, "xyz": current_xyz})
        calculated_frustration = 0  # Reset frustration
        print(f" MEMORY: New valley discovered at {current_e:.4f} eV! Cataloging...")
    else:
        calculated_frustration = current_frustration + 1
        print(f"MEMORY: Trapped in known valley. Frustration rising to {calculated_frustration}/3.")


    system_prompt = """You are an expert computational chemist AI supervising a geometry optimization engine.
The physics engine has just completed a local optimization and reached a local minimum on the Potential Energy Surface (PES).

Your job is to analyze the state and return a strictly formatted JSON object dictating the next macro-level action.

JSON Output Schema:
{
    "frustration_update": int,  // Add 1 to current frustration if stuck, or reset to 0 if this is a brand new valley.
    "recommended_kick": float,  // A suggested Monte Carlo kick strength in Angstroms (e.g., 0.5 to 1.5).
    "reasoning": str            // A brief, 1-sentence log of your thought process.
}"""

    human_prompt = f"Current Frustration: {calculated_frustration}. Total Unique Valleys: {len(minima_list)}. Generate JSON."

    try:
        # Invoke Ollama
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        response = supervisor_llm.invoke(messages)

        # Parse Decision
        decision = json.loads(response.content)
        new_kick = decision.get("recommended_kick", 0.5)
        thought_log = decision.get("reasoning", "LLM decided to proceed.")

    except Exception as e:
        print(f"LLM Error: {e}. Defaulting to safe values.")
        new_kick = 0.5 + (calculated_frustration * 0.2)  # Mathematical fallback
        thought_log = "Fallback due to LLM timeout/error."

    print(f"Supervisor Thought: {thought_log}")
    print(f"Recommended Kick: {new_kick} Å")

    # --- 3. Return the fully updated state ---
    return {
        "discovered_minima": minima_list,  # Ensure memory is saved!
        "frustration": calculated_frustration,
        "kick_strength": new_kick,
        "history_log": state.get('history_log', []) + [f"Supervisor: {thought_log}"]
    }
def perturbation_node(state: ExplorerState) -> dict:

    atoms = read(StringIO(state['current_xyz']),
                 format="xyz")


    current_kick = state.get('kick_strength', 0)

    # 2. The Monte Carlo "Kick"
    atoms.rattle(stdev=current_kick)

    # 3. Save new geometry back to string
    out_stream = StringIO()
    write(out_stream, atoms, format="xyz")

    print("Applied Monte Carlo rattle to escape local minimum.")

    return {
        "current_xyz": out_stream.getvalue(),
        'kick_strength': current_kick * 1.01,
        # "local_step_count": 0,  # Reset for the next optimization run
        "global_step_count": state.get('global_step_count', 0) + 1,
        "history_log": state['history_log'] + ["Applied 0.5A MC Kick."]
    }