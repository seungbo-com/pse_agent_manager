# agents/controller_nodes.py
import numpy as np
import pandas as pd
from ase.io import read, write
from io import StringIO
from numpy.linalg import norm
from state.schemas import MoleculeState, ExplorerState
from tools.calc_tools import calculate_single_point

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


def analyzer_node(state: ExplorerState) -> dict:

    new_energy = state['current_energy']
    memory = state['discovered_minima']

    # Check if this energy matches any previously found minimum
    is_novel = True
    for saved_valley in memory:
        # If energy is within 0.01 eV, we assume we fell into the same valley
        if abs(new_energy - saved_valley['energy']) < 0.01:
            is_novel = False
            break

    # Discovered new local minimum
    if is_novel:
        print(f"Discovered new minimum at {new_energy:.3f} eV!")
        # Add to memory
        new_record = {"energy": new_energy, "xyz": state['current_xyz']}
        return {
            "discovered_minima": memory + [new_record],
            "frustration": 0,  # Reset frustration
            "history_log": state['history_log'] + ["Found novel minimum."],
            "global_step_count":state.get('global_step_count', 0) + 1,
        }

    # Same minimum found before
    else:
        print("Fell into a known valley.")
        return {
            "frustration": state.get('frustration', 0) + 1,
            "history_log": state['history_log'] + ["Trapped in known minimum."]
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
        'kick_strength': current_kick*1.05,
        # "local_step_count": 0,  # Reset for the next optimization run
        "global_step_count": state.get('global_step_count', 0) + 1,
        "history_log": state['history_log'] + ["Applied 0.5A MC Kick."]
    }