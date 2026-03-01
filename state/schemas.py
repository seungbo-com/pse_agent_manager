from typing import TypedDict, List, Dict

class MoleculeState(TypedDict):
    # atoms_obj: object          # The ASE Atoms object (can't be serialized directly usually, but kept in memory)
    xyz_string: str            # String representation for the LLM to read
    # current_energy: float      # Latest Energy
    # forces: List[List[float]]  # Forces on each atom
    # max_force: float           # The convergence metric
    step_count: int            # To prevent infinite loops
    # trajectory: List[float]    # History of energies to track progress
    # status: str                # "running", "converged", "failed"
    # step_size: float
    current_r: float
    current_theta: float
    error_r: float
    error_theta: float
    current_energy: float
    expected_energy: float

# For exploring the PES surface
class ExplorerState(TypedDict):
    current_xyz: str              # The structure right now
    current_energy:float          # Current Energy
    discovered_minima: List[Dict] # Memory: [{"energy": -12.4, "xyz": "..."}]
    frustration: int              # How many times it found the SAME valley in a row
    kick_strength: float          # How hard to "rattle" the atoms next time
    history_log: List[str]        # LLM's scratchpad/log of actions
    global_step_count: int        # Total MC moves made