# main.py (Updated for Recording)
import os
from workflow.graph import master_app
from tools.calc_tools import calculate_single_point,get_bad_water_xyz
from tools.result_plot import plot_controller_journey, plot_2d_pes
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
    r_min=0.65,
    r_max=2.25,
    theta_min=60,
    theta_max=150,
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


print('Plotting the true energy landscape + explored')
plot_controller_journey("data/true_energy.csv", log_file)
plot_2d_pes("data/true_energy.csv", log_file)
# plot_3d('./data/true_energy.csv','./data/pes_3d_map.csv')