import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ase.io import read
from scipy.interpolate import make_interp_spline
from itertools import combinations
from matplotlib.widgets import RadioButtons

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



        # Calculate distance from every frame
        for num_frame,atoms in enumerate(traj):
            for num_comb, ea_comb in enumerate(comb_idx):
                distances[num_frame,num_comb] = atoms.get_distance(ea_comb[0], ea_comb[1])

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
    plt.show()

# if __name__ == "__main__":
#     main()