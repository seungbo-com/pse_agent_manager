from typing import TypedDict, List, Optional

class MoleculeState(TypedDict):
    atoms_obj: object          # The ASE Atoms object (can't be serialized directly usually, but kept in memory)
    xyz_string: str            # String representation for the LLM to read
    current_energy: float      # Latest Energy
    forces: List[List[float]]  # Forces on each atom
    max_force: float           # The convergence metric
    step_count: int            # To prevent infinite loops
    trajectory: List[float]    # History of energies to track progress
    status: str                # "running", "converged", "failed"