# main.py (Updated for Recording)
import os
from workflow.graph import app
from tools.calc_tools import calculate_single_point

# --- 1. SETUP THE "NOTEBOOK" ---
output_file = "optimization_movie.xyz"
log_file = "optimization_log.csv"

# Clear old files so we don't mix up experiments
with open(output_file, "w") as f:
    f.write("")
with open(log_file, "w") as f:
    f.write("Step,Energy,MaxForce\n")  # CSV Header

# --- 2. INITIALIZE ---
initial_xyz = """2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.8"""

initial_data = calculate_single_point(initial_xyz)

initial_state = {
    "xyz_string": initial_xyz,
    "current_energy": initial_data['energy'],
    "max_force": initial_data['max_force'],
    "forces": initial_data['forces']
}

# Write the starting frame
with open(output_file, "a") as f:
    f.write(initial_xyz)

print(f"Recording trajectory to: {os.path.abspath(output_file)}")

# --- 3. RUN AND RECORD ---
step_count = 0
for event in app.stream(initial_state):
    # 'event' contains the dictionary returned by the agent
    data = event.get('agent')  # 'agent' is the name of your node

    if data and 'xyz_string' in data:
        step_count += 1

        # A. Save the Geometry (The Structure)
        with open(output_file, "a") as f:
            f.write(data['xyz_string'])

        # B. Save the Numbers (The Data)
        e = data.get('current_energy', 0)
        f_max = data.get('max_force', 0)

        with open(log_file, "a") as f:
            f.write(f"{step_count},{e},{f_max}\n")

        print(f"💾 Saved Step {step_count} to disk.")