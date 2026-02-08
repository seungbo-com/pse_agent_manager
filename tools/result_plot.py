import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.io import read
from scipy.interpolate import make_interp_spline
from itertools import combinations
from matplotlib.widgets import RadioButtons
from matplotlib import cm
import plotly.graph_objects as go
import webbrowser

csv_file = "./data/optimization_log.csv"
xyz_file = "./data/optimization_movie.xyz"

def pes_plot():

    print(" Outputing the final result")
    print(" Loading data...")

    try:
        df = pd.read_csv(csv_file)
        energies = df['Energy'].values
        steps = df['Step'].values
    except FileNotFoundError:
        print(f" Error: Could not find {csv_file}. Run main.py first!")
        return

    # Load Geometries from XYZ to calculate Bond Lengths
    try:
        traj = read(xyz_file, index=':')  # Read all frames

        atomic_num_list = traj[0].get_chemical_symbols() # gathering first index
        comb_idx = list(combinations(np.arange(len(atomic_num_list)),2)) # all combination of indexes
        distances = np.zeros((len(traj),len(comb_idx)),dtype=float)

        ang_avgDist = np.zeros((len(traj),3),dtype=float)

        # Calculate distance from every frame
        for num_frame,atoms in enumerate(traj):

            for num_comb, ea_comb in enumerate(comb_idx):

                distances[num_frame,num_comb] = atoms.get_distance(ea_comb[0], ea_comb[1])

            # Collecting the three/two body correlation
            symbols = atoms.get_chemical_symbols()
            o_idx = symbols.index('O')
            h_indices = [i for i, s in enumerate(symbols) if s == 'H']

            r1 = atoms.get_distance(o_idx, h_indices[0])
            r2 = atoms.get_distance(o_idx, h_indices[1])
            avg_r = (r1 + r2) / 2.0
            theta = atoms.get_angle(h_indices[0], o_idx, h_indices[1])
            ang_avgDist[num_frame] = [avg_r, theta, energies[num_frame]]

        # label checking
        label_list = list(f'{traj[0].get_chemical_symbols()[ea_comb[0]]}{ea_comb[0]}  {traj[0].get_chemical_symbols()[ea_comb[1]]}{ea_comb[1]}' for ea_comb in comb_idx)

        # Sync checks: Ensure CSV and XYZ have same number of points
        min_len = min(len(energies), len(distances))
        energies = energies[:min_len]
        distances = distances[:min_len]
        steps = steps[:min_len]

    except FileNotFoundError:
        print(f" Error: Could not find {xyz_file}."); exit()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.3)  # Make room for buttons on the left

    # Initial Plot (First Pair)
    current_pair_idx = 0

    # Sort data for the initial line
    x_data = distances[:, current_pair_idx]
    y_data = energies

    # Sort for clean line plotting
    sort_idx = np.argsort(x_data)
    scatter = ax.plot(x_data, y_data, 'o', color='red', label='Agent Steps', alpha=0.6)

    # Try smoothing
    try:
        spline = make_interp_spline(x_data[sort_idx], y_data[sort_idx])
        x_smooth = np.linspace(x_data.min(), x_data.max(), 200)
        y_smooth = spline(x_smooth)
        line, = ax.plot(x_smooth, y_smooth, '-', color='blue', alpha=0.5, label='PES Slice')

    except:
        # Fallback if too few points
        line, = ax.plot(x_data[sort_idx], y_data[sort_idx], '-', color='blue', alpha=0.5)

    ax.set_xlabel("Interatomic Distance (Å)")
    ax.set_ylabel("Potential Energy (eV)")
    ax.set_title(f"PES Slice: {label_list[0]}")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

    # Creating the button
    ax_radio = plt.axes([0.02, 0.4, 0.2, 0.25], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, label_list)

    def update(label):
        # Find which index matches the clicked label
        idx = label_list.index(label)
        # Get new X data
        x_new = distances[:, idx]
        # 1. Update Scatter
        # Note: 'plot' returns a list of lines. We need the first one.
        scatter[0].set_data(x_new, y_data)

        # 2. Update Line (Sort first)
        sort_i = np.argsort(x_new)
        x_sorted = x_new[sort_i]
        y_sorted = y_data[sort_i]

        try:
            spline_new = make_interp_spline(x_sorted, y_sorted)
            x_sm = np.linspace(x_sorted.min(), x_sorted.max(), 200)
            y_sm = spline_new(x_sm)
            line.set_data(x_sm, y_sm)
        except:
            line.set_data(x_sorted, y_sorted)

        # Rescale axes to fit new data range
        ax.relim()
        ax.autoscale_view()
        ax.set_title(f"PES Slice: {label}")
        fig.canvas.draw_idle()

    radio.on_clicked(update)

    print(" Interactive Plot Generated. Check the popup window!")
    # plt.show()
    # Save to CSV
    df = pd.DataFrame.from_dict({
        "Bond_Length": ang_avgDist.T[0,:].tolist(),
        "Bond_Angle": ang_avgDist.T[1,:].tolist(),
        "Energy": ang_avgDist.T[2,:].tolist(),
    })
    df.to_csv('./data/pes_3d_map.csv', index=False)

def plot_3d(csv_file, csv_file_agent):

    '''
    Plotting 3D surface plot

    :param
        df - Pandas DataFrame from collected data
        df_agent - Pandas DataFrame of agentic automatic exploration

    :return: Saving into png file.
    '''

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f" Error: Could not find {csv_file}.")
        return

    try:
        df_agent = pd.read_csv(csv_file_agent)
    except FileNotFoundError:
        print(f" Error: Could not find {csv_file_agent}.")
        return

    print("Generating 3D Plot...")

    # Prepare Data for Plotting
    # X = df['Bond_Length'].values
    # Y = df['Bond_Angle'].values
    # Z = df['Energy'].values
    #
    X_a = df_agent['Bond_Length'].values
    Y_a = df_agent['Bond_Angle'].values
    Z_a = df_agent['Energy'].values

    print(df,df_agent)
    # shape = (len(r_steps), len(theta_steps))

    # pivot the data
    pivot = df.pivot(index='Bond_Length',
                     columns='Bond_Angle',
                     values='Energy')

    X_grid, Y_grid = np.meshgrid(pivot.index,pivot.columns)
    Z_grid = pivot.values
    pivot_a = df_agent.pivot(index='Bond_Length',
                     columns='Bond_Angle',
                     values='Energy')
    X_gridA, Y_gridA = np.meshgrid(pivot_a.index, pivot_a.columns);Z_gridA = pivot_a.values
    steps = np.arange(X_grid.shape[0])
    fig = go.Figure()

    # Layer A: The Surface
    fig.add_trace(go.Surface(
        z=Z_grid, x=X_grid, y=Y_grid,
        colorscale='Viridis',
        opacity=0.8,
        name='PES Landscape',
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
    ))

    # Layer B: The Agent Path (Line)
    fig.add_trace(go.Scatter3d(
        x=X_a, y=Y_a, z=Z_a,
        mode='lines',
        line=dict(color='white', width=5),
        name='Agent Path'
    ))

    # Layer C: The Agent Steps (Dots)
    fig.add_trace(go.Scatter3d(
        x=X_a, y=Y_a, z=Z_a,
        mode='markers',
        marker=dict(
            size=5,
            color=steps,
            colorscale='Hot',
            showscale=False
        ),
        text=[f"Step {s}<br>r={r:.3f}<br>θ={t:.1f}<br>Pot E={e:.3f}"  for s, r, t, e in zip(steps, X_a, Y_a, Z_a)],
        hoverinfo='text',
        name='Optimization Steps'
    ))

    # Layer D: Start & End Markers
    fig.add_trace(go.Scatter3d(
        x=[X_a[0]], y=[Y_a[0]], z=[Z_a[0]],
        mode='markers', marker=dict(size=10, color='lime', symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[X_a[-1]], y=[Y_a[-1]], z=[Z_a[-1]],
        mode='markers', marker=dict(size=10, color='red', symbol='cross'),
        name='End'
    ))

    fig.update_layout(
        title='Interactive Agent Optimization on PES',
        scene=dict(
            xaxis_title='Bond Length (Å)',
            yaxis_title='Bond Angle (Deg)',
            zaxis_title='Energy (eV)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))  # Initial View Angle
        ),
        width=1200, height=800,
        template='plotly_dark'  # Makes the colors pop
    )

    output_html = "interactive_pes.html"
    # Save and Show
    fig.write_html(output_html)
    print(f"Interactive plot saved to '{output_html}'")

    # Try to open automatically in browser
    webbrowser.open(output_html)

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # # Plot the Surface
    # surf = ax.plot_surface(X_grid, Y_grid, Z_grid,
    #                        cmap=cm.viridis,
    #                        linewidth=0,
    #                        antialiased=False,
    #                        alpha=0.8)
    #
    # # Start Point (Green)
    # ax.scatter(X_gridA[0], Y_gridA[0], Z_gridA[0],
    #            color='lime', s=100, edgecolors='k', label='Start', zorder=11)
    # # End Point (Red Star)
    # ax.scatter(X_gridA[-1], Y_gridA[-1], Z_gridA[-1],
    #            color='red', marker='*', s=200, edgecolors='k', label='End',zorder=11)
    # # Intermediate Steps (Black dots)
    # ax.scatter(X_gridA[1:-1], Y_gridA[1:-1], Z_gridA[1:-1],
    #            color='black', s=20, zorder=11)
    #
    #
    # # Add Contour (Floor projection) to see the "Map"
    # ax.contour(X_grid, Y_grid, Z_grid,
    #            zdir='z',
    #            offset=Z_grid.min(),
    #            cmap=cm.viridis)
    #
    # # Labels
    # ax.set_xlabel('Bond Length (Å)')
    # ax.set_ylabel('Bond Angle (Degrees)')
    # ax.set_zlabel('Potential Energy (eV)')
    # ax.set_title('Water Molecule Potential Energy Surface')
    #
    # # Colorbar
    # fig.colorbar(surf, shrink=0.5, aspect=5, label='Energy (eV)')
    #
    # plt.savefig("pes_3d_plot.png", dpi=300)
    # print("3D Plot saved to 'pes_3d_plot.png'")
    # plt.show()