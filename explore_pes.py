import numpy as np
import pandas as pd
from tools.calc_tools import calculate_single_point

# Define the resolution of your map
# More points = smoother map, but takes longer
r_steps = np.linspace(0.8, 1.4, 15)  # Bond Length: 0.8 to 1.4 Angstroms
theta_steps = np.linspace(90, 130, 15)  # Bond Angle: 90 to 130 Degrees

output_file = "./data/true_energy.csv"

def get_water_geometry(r, theta_deg):
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
    Grid Point r={r:.2f} theta={theta_deg:.2f}
    O  0.000 0.000 0.000
    H  {h1_x:.3f} {h1_y:.3f} 0.000
    H  {h2_x:.3f} {h2_y:.3f} 0.000
    """.strip()


def overall_landscape():

    print("Starting creating energy landscape!!")
    print(f"Grid Size: {len(r_steps)} x {len(theta_steps)} = {len(r_steps) * len(theta_steps)} points")

    results = []
    total = len(r_steps) * len(theta_steps)
    count = 0

    for r in r_steps:
        for theta in theta_steps:
            count += 1

            # Create the hypothetical molecule
            xyz_string = get_water_geometry(r, theta)
            # calculate the energy (Single Point)
            data = calculate_single_point(xyz_string)
            energy = data['energy']

            # 3. Log it
            results.append({
                "Bond_Length": r,
                "Bond_Angle": theta,
                "Energy": energy
            })

            # Progress Bar
            if count % 10 == 0:
                print(f"   [{count}/{total}] r={r:.2f}Å, θ={theta:.1f}°, E={energy:.3f} eV")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f" Data saved to {output_file}")

    # plot_3d(df)

#
# if __name__ == "__main__":
#     main()