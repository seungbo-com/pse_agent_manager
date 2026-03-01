# main.py (Updated for Recording)
import os
from workflow.graph import master_app
from tools.calc_tools import calculate_single_point,get_bad_water_xyz
from tools.result_plot import plot_controller_journey
from explore_pes import overall_landscape
from state.schemas import MoleculeState
# Setting up the output log/xyz
output_file = "data/optimization_movie.xyz"
log_file = "data/optimization_log.csv"

# Clear old files so we don't mix up experiments
with open(output_file, "w") as f:
    f.write("")
with open(log_file, "w") as f:
    f.write("global_step,Step,r_angstrom,theta_deg,real_energy_ev,map_energy_ev,error_r,error_theta\n")


# # Initializing structure
# initial_xyz ="""3
# H2O molecule
# O          0.00000        0.00000        0.11779
# H          0.00000        0.75545       -0.47116
# H          0.00000       -0.75545       -0.47116
# """

print('Creating the true energy landscape')
overall_landscape(
    r_min=0.85,
    r_max=1.25,
    theta_min=100,
    theta_max=130,
    lin_space=50,
) # creating the dataset


# ----------------------------
# Setting up the initial data
# ----------------------------
initial_xyz = get_bad_water_xyz()
# initial_state = MoleculeState(
#         xyz_string=initial_xyz,
#         current_r=0.0,
#         current_theta=0.0,
#         current_energy=0.0,
#         expected_energy=0.0,
#         error_r=0.0,
#         error_theta=0.0,
#         step_count=0,
#     )

initial_explorer_state = {
    "current_xyz": initial_xyz,
    "current_energy": 0.0,
    "discovered_minima": [],
    "frustration": 0,
    "kick_strength": 0.5,
    "history_log": ["Agent initialized with starting geometry."],
    "global_step_count":1,
}

# Write the starting frame
with open(output_file, "a") as f:
    f.write(initial_xyz)

print(f"Recording trajectory to: {os.path.abspath(output_file)}")

for event in master_app.stream(initial_explorer_state):

    node_name = list(event.keys())[0]

    if node_name == 'check_memory':
        print(f'Agent Checking Memory: Current Frustration {event[node_name].get("frustration")}')

    if node_name == 'apply_kick':
        print(f'Agent Applying Kick: New Strength {event[node_name].get("kick_strength"):.2f}')
# # --- Initialize Local Tracker ---
# tracked_state = dict(initial_state)
# # print("Starting Map-Guided Optimization Loop...")
# # --- Running Recording ---
# for event in master_app.stream(initial_state):
#
#     # LangGraph yields a dictionary like: {'sensor': {'current_r': 1.0, ...}}
#     node_name = list(event.keys())[0]
#     node_data = event[node_name]
#
#     # Update our local tracker with whatever this specific node just changed
#     tracked_state.update(node_data)
#
#     if node_name == 'comparator':
#         step_count = tracked_state.get('step_count', 0)
#         # Extract the metrics
#         xyz = tracked_state['xyz_string']
#         r = tracked_state['current_r']
#         theta = tracked_state['current_theta']
#         e_real = tracked_state['current_energy']
#         e_map = tracked_state['expected_energy']
#         err_r = tracked_state['error_r']
#         err_theta = tracked_state['error_theta']
#
#         # Save the Geometry (The Structure)
#         with open(output_file, "a") as f:
#             # Ensure there's a newline between XYZ blocks
#             f.write(xyz.strip() + "\n")
#
#             # Save the Numbers (The Data)
#         with open(log_file, "a") as f:
#             f.write(f"{step_count},{r:.4f},{theta:.2f},{e_real:.6f},{e_map:.6f},{err_r:.4f},{err_theta:.2f}\n")
#
#         print(f"Logged Step {step_count} to disk.")

# # Running Recording
# step_count = 0
# for event in app.stream(initial_state):
#     # 'event' contains the dictionary returned by the agent
#     data = event.get('agent')  # 'agent' is the name of your node
#
#     if data and 'xyz_string' in data:
#         step_count += 1
#
#         # A. Save the Geometry (The Structure)
#         with open(output_file, "a") as f:
#             f.write(data['xyz_string'])
#
#         # B. Save the Numbers (The Data)
#         e = data.get('current_energy', 0)
#         f_max = data.get('max_force', 0)
#
#         with open(log_file, "a") as f:
#             f.write(f"{step_count},{e},{f_max}\n")
#
#         print(f" Saved Step {step_count} to disk.")

# print('Starting making PES plot!!')
# pes_plot()
#
# print('Creating the true energy landscape')
# overall_landscape() # creating the dataset

print('Plotting the true energy landscape + explored')
plot_controller_journey("data/true_energy.csv", log_file)
# plot_3d('./data/true_energy.csv','./data/pes_3d_map.csv')