import csv
import math
import numpy as np
from matplotlib import pyplot as plt

# Update paths to match the contact data
BASE_FOLDER = ""
SAVE_PLOT_FOLDER = BASE_FOLDER + ""
DATA_FILE = SAVE_PLOT_FOLDER + "contact_mean_std_sph_p_rod.txt"

plt.rcParams['font.size'] = 20  # Default font size for all text
plt.rcParams['axes.titlesize'] = 42  # Title font size
plt.rcParams['axes.labelsize'] = 36  # Axis labels (x, y)
plt.rcParams['xtick.labelsize'] = 30  # X-tick labels
plt.rcParams['ytick.labelsize'] = 30  # Y-tick labels
plt.rcParams['legend.fontsize'] = 20  # Legend font size

def parse_contact_data(filename):
    """
    Parses the contact data file with format:
    # type
    n_spheres mean std
    Returns dict: {type: {'x': [...], 'mean': [...], 'std': [...]}}
    """
    data = {}
    current_type = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# "):
                    # Extract type from comment line
                    parts = line.split()
                    if len(parts) >= 2:
                        current_type = parts[1].lower()
                        if current_type not in data:
                            data[current_type] = {'x': [], 'mean': [], 'std': []}
                continue

            # Parse data line
            if current_type is not None:
                parts = line.split()
                if len(parts) >= 3:
                    n_spheres = int(parts[0])
                    mean = float(parts[1])
                    std = float(parts[2])

                    data[current_type]['x'].append(n_spheres)
                    data[current_type]['mean'].append(mean)
                    data[current_type]['std'].append(std)

    return data

def N_to_furrow_param(series):
    xs = series["x"]  # a Python list of N values
    alphas = []
    for N in xs:
        # validity checks
        if not isinstance(N, (int, float)) or not math.isfinite(N) or N <= 1:
            alphas.append(float("nan"))
            continue
        #furrow relative depth      (1.0 - (2.0 / (N - 1.0)))**0.5 --> choose this one
        #furrow relative volume     (1.0 - (2.0 / (N - 1.0)))
        inside = (1.0 - (2.0 / (N - 1.0)))**(0.5)
        if inside < 0.0:
            inside = 0.0
        alphas.append(math.sqrt(inside))
    series["x"] = alphas  # overwrite with alpha

def sphere_volume(radius):
    """Calculate the volume of a sphere given its radius."""
    return (4/3) * np.pi * (radius ** 3)

def change_normalization(shape_type, n_minimal_sphere=3, r=0.0025):
    """Calculate normalization factor based on shape type"""
    individual_volume = sphere_volume(r)
    steinmetz2_volume = (16.0/3.0) * (r**3)
    steinmetz3_volume = 8.0 * (2.0 - np.sqrt(2.0)) * (r**3)
    rod_volume = np.pi * (r**2) * ((n_minimal_sphere - 1) * 2.0 * r) + individual_volume

    if shape_type == "rod":
        denom = rod_volume
    elif shape_type == "cross":
        denom = (2.0 * rod_volume - steinmetz2_volume)
    elif shape_type == "star":
        denom = (3.0 * rod_volume - 3.0 * steinmetz2_volume + steinmetz3_volume)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    return denom/individual_volume

def renormalize_values(y, std, type_name, n_minimal_sphere=3, r=0.0025):
    """
    Renormalize mean and std values using the appropriate normalization factor.

    Parameters:
    - y: array of mean values
    - std: array of standard deviation values
    - type_name: string indicating the shape type ('rod', 'cross', or 'star')
    - n_minimal_sphere: number of minimal spheres (default 3)
    - r: radius of spheres (default 0.0025)

    Returns:
    - tuple of (renormalized_y, renormalized_std)
    """
    # Calculate normalization factor
    norm_factor = change_normalization(type_name, n_minimal_sphere, r)

    # Renormalize all values
    renormalized_y = y * norm_factor
    renormalized_std = std * norm_factor

    return renormalized_y, renormalized_std

def plot_contact_data(data, colors):
    """
    Plots the contact data with error bars showing standard deviation
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    # Linear scale is more appropriate for contact data
    # ax.set_yscale('symlog', linthresh=1e-10)

    # Define the order we want to plot the types
    plot_order = ['rod', 'cross', 'star']

    for type_name in plot_order:
        if type_name not in data:
            continue

        color = colors[type_name]
        series = data[type_name]

        N_to_furrow_param(series)
        x = 1-np.asarray(series['x'])

        y = np.asarray(series['mean'])
        std = np.asarray(series['std'])

        # Plot with error bars
        ax.errorbar(x, y,
                   yerr=std,
                   fmt='o', linestyle='',
                   label=type_name,
                   markersize=10,
                   markeredgewidth=1.5,
                   markerfacecolor=color,
                   markeredgecolor='black',
                   elinewidth=1.5,
                   ecolor=color,
                   capsize=3,
                   capthick=1.5,
                   zorder=3,
                   alpha=1)

    ax.set_xlabel("Meso roughness")
    ax.set_ylabel("Coordination Number")  # normalized with sphere volume
    #ax.set_title("Normalized Contacts vs Furrow Filling")
    ax.grid(alpha=0.25, which='both')
    ax.set_xscale('log')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys(), fontsize=52, loc=[0.2375,0.21])

    # Save the plot
    output_path = SAVE_PLOT_FOLDER + "contact_plot.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    # Define colors for each type
    colors = {
        'cross': 'purple',
        'rod': 'green',
        'star': 'orange'
    }

    # Parse the data
    data = parse_contact_data(DATA_FILE)

    # Plot the data
    plot_contact_data(data, colors)

if __name__ == "__main__":
    main()
