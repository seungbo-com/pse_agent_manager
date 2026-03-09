import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.io import read
from scipy.interpolate import make_interp_spline
from itertools import combinations
from matplotlib.widgets import RadioButtons
from matplotlib import cm
import plotly.graph_objects as go
import webbrowser, os

csv_file = "./data/optimization_log.csv"
xyz_file = "./data/optimization_movie.xyz"

# def pes_plot():
#
#     print(" Outputing the final result")
#     print(" Loading data...")
#
#     try:
#         df = pd.read_csv(csv_file)
#         energies = df['Energy'].values
#         steps = df['Step'].values
#     except FileNotFoundError:
#         print(f" Error: Could not find {csv_file}. Run main.py first!")
#         return
#
#     # Load Geometries from XYZ to calculate Bond Lengths
#     try:
#         traj = read(xyz_file, index=':')  # Read all frames
#
#         atomic_num_list = traj[0].get_chemical_symbols() # gathering first index
#         comb_idx = list(combinations(np.arange(len(atomic_num_list)),2)) # all combination of indexes
#         distances = np.zeros((len(traj),len(comb_idx)),dtype=float)
#
#         ang_avgDist = np.zeros((len(traj),3),dtype=float)
#
#         # Calculate distance from every frame
#         for num_frame,atoms in enumerate(traj):
#
#             for num_comb, ea_comb in enumerate(comb_idx):
#
#                 distances[num_frame,num_comb] = atoms.get_distance(ea_comb[0], ea_comb[1])
#
#             # Collecting the three/two body correlation
#             symbols = atoms.get_chemical_symbols()
#             o_idx = symbols.index('O')
#             h_indices = [i for i, s in enumerate(symbols) if s == 'H']
#
#             r1 = atoms.get_distance(o_idx, h_indices[0])
#             r2 = atoms.get_distance(o_idx, h_indices[1])
#             avg_r = (r1 + r2) / 2.0
#             theta = atoms.get_angle(h_indices[0], o_idx, h_indices[1])
#             ang_avgDist[num_frame] = [avg_r, theta, energies[num_frame]]
#
#         # label checking
#         label_list = list(f'{traj[0].get_chemical_symbols()[ea_comb[0]]}{ea_comb[0]}  {traj[0].get_chemical_symbols()[ea_comb[1]]}{ea_comb[1]}' for ea_comb in comb_idx)
#
#         # Sync checks: Ensure CSV and XYZ have same number of points
#         min_len = min(len(energies), len(distances))
#         energies = energies[:min_len]
#         distances = distances[:min_len]
#         steps = steps[:min_len]
#
#     except FileNotFoundError:
#         print(f" Error: Could not find {xyz_file}."); exit()
#
#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plt.subplots_adjust(left=0.3)  # Make room for buttons on the left
#
#     # Initial Plot (First Pair)
#     current_pair_idx = 0
#
#     # Sort data for the initial line
#     x_data = distances[:, current_pair_idx]
#     y_data = energies
#
#     # Sort for clean line plotting
#     sort_idx = np.argsort(x_data)
#     scatter = ax.plot(x_data, y_data, 'o', color='red', label='Agent Steps', alpha=0.6)
#
#     # Try smoothing
#     try:
#         spline = make_interp_spline(x_data[sort_idx], y_data[sort_idx])
#         x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
#         y_smooth = spline(x_smooth)
#         line, = ax.plot(x_smooth, y_smooth, '-', color='blue', alpha=0.5, label='PES Slice')
#
#     except:
#         # Fallback if too few points
#         line, = ax.plot(x_data[sort_idx], y_data[sort_idx], '-', color='blue', alpha=0.5)
#
#     ax.set_xlabel("Interatomic Distance (Å)")
#     ax.set_ylabel("Potential Energy (eV)")
#     ax.set_title(f"PES Slice: {label_list[0]}")
#     ax.grid(True, linestyle='--', alpha=0.3)
#     ax.legend()
#
#     # Creating the button
#     ax_radio = plt.axes([0.02, 0.4, 0.2, 0.25], facecolor='#f0f0f0')
#     radio = RadioButtons(ax_radio, label_list)
#
#     def update(label):
#         # Find which index matches the clicked label
#         idx = label_list.index(label)
#         # Get new X data
#         x_new = distances[:, idx]
#         # 1. Update Scatter
#         # Note: 'plot' returns a list of lines. We need the first one.
#         scatter[0].set_data(x_new, y_data)
#
#         # 2. Update Line (Sort first)
#         sort_i = np.argsort(x_new)
#         x_sorted = x_new[sort_i]
#         y_sorted = y_data[sort_i]
#
#         try:
#             spline_new = make_interp_spline(x_sorted, y_sorted)
#             x_sm = np.linspace(x_sorted.min(), x_sorted.max(), 200)
#             y_sm = spline_new(x_sm)
#             line.set_data(x_sm, y_sm)
#         except:
#             line.set_data(x_sorted, y_sorted)
#
#         # Rescale axes to fit new data range
#         ax.relim()
#         ax.autoscale_view()
#         ax.set_title(f"PES Slice: {label}")
#         fig.canvas.draw_idle()
#
#     radio.on_clicked(update)
#
#     print(" Interactive Plot Generated. Check the popup window!")
#     # plt.show()
#     # Save to CSV
#     df = pd.DataFrame.from_dict({
#         "Bond_Length": ang_avgDist.T[0,:].tolist(),
#         "Bond_Angle": ang_avgDist.T[1,:].tolist(),
#         "Energy": ang_avgDist.T[2,:].tolist(),
#     })
#     df.to_csv('./data/pes_3d_map.csv', index=False)


def plot_controller_journey(pes_map_csv, controller_log_csv):
    """
    Plots the pre-calculated PES map and overlays the Math-Based Controller's
    trajectory using the 7-column log file.
    """

    # Loading the landscape
    print(f"Loading PES Map from {pes_map_csv}...")
    try:
        df_map = pd.read_csv(pes_map_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {pes_map_csv}")
        return

    # Pivot map data for Plotly (Angle must be index/rows, Length must be columns)
    pivot = df_map.pivot(index='Bond_Angle',
                         columns='Bond_Length',
                         values='Energy')
    X_grid = pivot.columns.values # 1D Array of Lengths (r)
    Y_grid = pivot.index.values   # 1D Array of Angles (theta)
    Z_grid = pivot.values         # 2D Matrix of Energies

    print(f"Loading Controller Trajectory from {controller_log_csv}...")
    try:
        df_ctrl = pd.read_csv(controller_log_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {controller_log_csv}")
        return

    fig = go.Figure()

    # Layer A (Trace 0): The Map Surface
    fig.add_trace(go.Surface(
        z=Z_grid, x=X_grid, y=Y_grid,
        colorscale='Viridis',
        opacity=0.8,
        name='PES Landscape',
        colorbar=dict(title='Energy (eV)', x=0.8),
        contours_z=dict(show=True, usecolormap=False, highlightcolor="limegreen", project_z=True)
    ))
    # Layer B (Trace 1): Base trace for the Line
    fig.add_trace(go.Scatter3d(
        x=[df_ctrl['r_angstrom'].iloc[0]], y=[df_ctrl['theta_deg'].iloc[0]], z=[df_ctrl['real_energy_ev'].iloc[0]],
        mode='lines', line=dict(color='white', width=6), name='Controller Path'
    ))

    # Layer C (Trace 2): Base trace for the Dots
    fig.add_trace(go.Scatter3d(
        x=[df_ctrl['r_angstrom'].iloc[0]], y=[df_ctrl['theta_deg'].iloc[0]], z=[df_ctrl['real_energy_ev'].iloc[0]],
        mode='markers', marker=dict(size=6, color='red'), name='Controller Updates'
    ))

    frames = []

    for i in range(1, len(df_ctrl)):
        cycle_data = df_ctrl[:i]
        X_a = cycle_data['r_angstrom'].values
        Y_a = cycle_data['theta_deg'].values
        Z_a = cycle_data['real_energy_ev'].values
        steps = cycle_data['Step'].values

        frame = go.Frame(
            data=[
                # Updates Trace 1 (The Line)
                go.Scatter3d(x=X_a, y=Y_a, z=Z_a),

                # Updates Trace 2 (The Dots)
                go.Scatter3d(
                    x=X_a, y=Y_a, z=Z_a,
                    marker=dict(size=6, color=steps, colorscale='Hot', showscale=False),
                    text=[f"Step {s}<br>r={r:.3f} Å<br>θ={t:.1f}°<br>E={e:.3f} eV"
                          for s, r, t, e in zip(steps, X_a, Y_a, Z_a)],
                    hoverinfo='text'
                )
            ],
            traces=[1, 2],
            name=f"frame_{i}"
        )
        frames.append(frame)

    fig.frames = frames

    # # Layer D: Start & End Markers
    # fig.add_trace(go.Scatter3d(
    #     x=[X_a[0]],
    #     y=[Y_a[0]],
    #     z=[Z_a[0]],
    #     mode='markers', marker=dict(size=12, color='lime', symbol='diamond'),
    #     name='Start Point'
    # ))
    # fig.add_trace(go.Scatter3d(
    #     x=[X_a[-1]],
    #     y=[Y_a[-1]],
    #     z=[Z_a[-1]],
    #     mode='markers', marker=dict(size=12, color='red', symbol='cross'),
    #     name='Final Geometry'
    # ))

    # Format the View
    fig.update_layout(
        title='Math-Based Feedback Controller Trajectory on PES',
        scene=dict(
            xaxis_title='Bond Length (H-O) (Å)',
            yaxis_title='Bond Angle (H-O-H) (Deg)',
            zaxis_title='Energy (eV)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200, height=800,
        template='plotly_dark'
    )

    fig.update_layout(
        showlegend=False,
        title="Agentic PES Exploration Movie",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0.9,  # Position on the screen
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 50, "redraw": True},  # 50ms per frame
                        "fromcurrent": True,
                        "transition": {"duration": 0}  # 0 transition stops it from smearing
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ]
        )]
    )

    output_html = "controller_journey.html"
    fig.write_html(output_html)
    print(f"Controller Journey plot saved to '{os.path.abspath(output_html)}'")

    # Auto-open in browser
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

def plot_2d_pes(pes_map_csv, controller_log_csv):

    # Loading the landscape
    print(f"Loading PES Map from {pes_map_csv}...")
    try:
        df_map = pd.read_csv(pes_map_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {pes_map_csv}")
        return

    # Pivot map data for Plotly (Angle must be index/rows, Length must be columns)
    pivot = df_map.pivot(index='Bond_Angle',
                         columns='Bond_Length',
                         values='Energy')
    X_grid = pivot.columns.values # 1D Array of Lengths (r)
    Y_grid = pivot.index.values   # 1D Array of Angles (theta)
    Z_grid = pivot.values         # 2D Matrix of Energies

    print(f"Loading Controller Trajectory from {controller_log_csv}...")
    try:
        df_ctrl = pd.read_csv(controller_log_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {controller_log_csv}")
        return


    fig = go.Figure()

    # Layer A (Trace 0): The 2D Contour Map
    fig.add_trace(go.Contour(
        z=Z_grid, x=X_grid, y=Y_grid,
        colorscale='Viridis',
        name='PES Landscape',
        colorbar=dict(title='Energy (eV)'),
        contours=dict(showlines=True)  # Shows the topological elevation lines
    ))

    fig.add_trace(go.Scatter(
        x=[df_ctrl['r_angstrom'].iloc[0]],
        y=[df_ctrl['theta_deg'].iloc[0]],
        mode='lines',
        line=dict(color='white', width=4),
        name='Agent Path'
    ))

    fig.add_trace(go.Scatter(
        x=[df_ctrl['r_angstrom'].iloc[0]],
        y=[df_ctrl['theta_deg'].iloc[0]],
        mode='markers',
        marker=dict(size=8, color='red', line=dict(width=1, color='black')),
        name='Updates'
    ))

    # 4. BUILD THE MOVIE FRAMES
    print("Generating 2D movie frames...")
    frames = []

    # Skip frames if the dataframe is huge so it doesn't crash the browser
    skip_rate = 2 if len(df_ctrl) > 500 else 1

    for i in range(1, len(df_ctrl), skip_rate):
        cycle_data = df_ctrl[:i]
        X_a = cycle_data['r_angstrom'].values
        Y_a = cycle_data['theta_deg'].values
        Z_a = cycle_data['real_energy_ev'].values
        steps = cycle_data['Step'].values

        frame = go.Frame(
            data=[
                # Updates Trace 1 (2D Line)
                go.Scatter(x=X_a, y=Y_a),

                # Updates Trace 2 (2D Dots)
                go.Scatter(
                    x=X_a, y=Y_a,
                    marker=dict(size=8, color=steps, colorscale='Hot', showscale=False),
                    text=[f"Step {s}<br>r={r:.3f} Å<br>θ={t:.1f}°<br>E={e:.3f} eV"
                          for s, r, t, e in zip(steps, X_a, Y_a, Z_a)]
                )
            ],
            traces=[1, 2],  # <--- Tells Plotly to only animate the Line and Dots, not the map
            name=f"frame_{i}"
        )
        frames.append(frame)

    fig.frames = frames

    fig.update_layout(
        showlegend=False,
        title="2D Agentic PES Exploration Movie",
        xaxis=dict(title='Bond Length (H-O) (Å)', range=[X_grid.min(), X_grid.max()]),
        yaxis=dict(title='Bond Angle (H-O-H) (Deg)', range=[Y_grid.min(), Y_grid.max()]),
        # xaxis_title='Bond Length (H-O) (Å)',
        # yaxis_title='Bond Angle (H-O-H) (Deg)',
        width=1000, height=800,
        template='plotly_dark',
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.1, y=1.05,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True,
                                  "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate",
                                    "transition": {"duration": 0}}])
            ]
        )]
    )

    output_html = "controller_journey_2d.html"
    fig.write_html(output_html)
    print(f"✅ 2D Controller Journey plot saved to '{os.path.abspath(output_html)}'")
    webbrowser.open(f"file://{os.path.abspath(output_html)}")