from ase.io import read
from ase.calculators.emt import EMT  # Toy potential (fast). SWAP for Gaussian/ORCA later.
from io import StringIO
import numpy as np


def calculate_single_point(xyz_data: str):
    """
    Takes an XYZ string, calculates Energy and Forces.
    """
    # 1. Convert string to ASE Atoms object
    xyz_file = StringIO(xyz_data)
    atoms = read(xyz_file, format="xyz")

    # 2. Attach Calculator (The Physics Engine)
    # NOTE: For real science, replace EMT() with a DFT calculator
    atoms.calc = EMT()

    # 3. Calculate
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    max_f = np.sqrt((forces ** 2).sum(axis=1).max())  # Max force magnitude

    return {
        "energy": energy,
        "max_force": max_f,
        "forces": forces.tolist()
    }


def update_geometry(xyz_data: str, atom_index: int, vector: list):
    """
    Moves an atom by a specific vector (Agent controls this).
    """
    xyz_file = StringIO(xyz_data)
    atoms = read(xyz_file, format="xyz")

    # Move the atom
    positions = atoms.get_positions()
    positions[atom_index] += np.array(vector)
    atoms.set_positions(positions)

    # Return new XYZ
    out_file = StringIO()
    atoms.write(out_file, format="xyz")
    return out_file.getvalue()