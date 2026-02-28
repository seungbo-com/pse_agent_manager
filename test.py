from ase.io import read
from ase.calculators.emt import EMT  # Toy potential (fast). SWAP for Gaussian/ORCA later.
from io import StringIO
import numpy as np

xyz_data = """3

O       0.000000000000000      0.047456582762629      0.270793713798385
H       0.000000000000000      0.825720161286046     -0.505635553243720
H       0.000000000000000     -0.764436809658985     -0.473816581056043""".strip()

# Converting string to ASE Atoms object
xyz_file = StringIO(xyz_data)
atoms = read(xyz_file, format="xyz")
atoms.calc = EMT()

# Collecting the three/two body correlation
symbols = atoms.get_chemical_symbols()
o_idx = symbols.index('O')
h_indices = [i for i, s in enumerate(symbols) if s == 'H']
r1 = atoms.get_distance(o_idx, h_indices[0])
r2 = atoms.get_distance(o_idx, h_indices[1])
avg_r = (r1 + r2) / 2.0
theta = atoms.get_angle(h_indices[0], o_idx, h_indices[1])
print(r1,r2)
# Calculate potential energy / force
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
max_f = np.sqrt((forces ** 2).sum(axis=1).max())  # Max force magnitude

print(avg_r, theta, energy)
