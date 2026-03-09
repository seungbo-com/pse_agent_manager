import numpy as np
import pandas as pd
from tools.calc_tools import calculate_single_point

# Define the resolution of your map
output_file = "./data/true_energy.csv"

def get_water_geometry(r,theta_deg):
    """
    Creates H2O geometry from internal coordinates.
    O is at (0,0,0). H1 is on X-axis. H2 is rotated by theta.
    """
    theta_rad = np.radians(theta_deg)

    # H1 Position
    h1_x = r
    h1_y = 0.0

    # H2 Position
    h2_x = r * np.cos(theta_rad)
    h2_y = r * np.sin(theta_rad)

    return f"""3
    Grid Point r={r:.2f}, theta={theta_deg:.2f}
    O  0.000 0.000 0.000
    H  {h1_x:.3f} {h1_y:.3f} 0.000
    H  {h2_x:.3f} {h2_y:.3f} 0.000
    """.strip()


def overall_landscape(r_min, r_max,theta_min,theta_max, lin_space):
    """
    This function creates the landscape geometry and saves it in a csv file.
    :argument:
        r_min (minimm - bond length / Angstrom),
        r_max (maximum - bond length / Angstrom),
        theta_min (minimm - bond angle / Angstrom),
        theta_max (maximum - bond angle / Angstrom),
        lin_space (bin width)

    """
    # More points = smoother map, but takes longer
    r_steps = np.linspace(r_min,r_max,lin_space)  # Bond Length
    theta_steps = np.linspace(theta_min,theta_max,lin_space)  # Bond Angle

    print("Starting creating energy landscape!!")
    print(f"Grid Size: {len(r_steps)} x {len(theta_steps)} = {len(r_steps) * len(theta_steps)} points")

    results = []
    total = len(r_steps) * len(theta_steps)
    count = 0

    for r in r_steps:

        for theta in theta_steps:

            count += 1

            # Create the hypothetical molecule
            xyz_string = get_water_geometry(r,theta)
            # calculate the energy (Single Point)
            data = calculate_single_point(xyz_string)
            # theta = atoms.get_angle(h_indices[0], o_idx, h_indices[1])

            energy = data['energy']

            # 3. Log it
            results.append({
                "Bond_Length": r,
                "Bond_Angle": theta,
                "Energy": energy
            })

            # Progress Bar
            if count % 500 == 0:
                print(f"[{count}/{total}] r={r:.2f}Å, θ={theta:.1f}°, E={energy:.3f} eV")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f" Data saved to {output_file}")

    # plot_3d(df)
