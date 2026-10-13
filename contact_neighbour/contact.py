import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

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
    # <family><friction>   (e.g. "rod0.4")
    n_spheres mean std
    Returns dict: {series_name: {'x': [...], 'mean': [...], 'std': [...]}}
    """
    data = {}
    current_type = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    current_type = parts[1].lower()
                    data.setdefault(current_type, {'x': [], 'mean': [], 'std': []})
                continue
            if current_type is None:
                continue
            parts = line.split()
            if len(parts) >= 3:
                n_spheres, mean, std = float(parts[0]), float(parts[1]), float(parts[2])
                data[current_type]['x'].append(n_spheres)
                data[current_type]['mean'].append(mean)
                data[current_type]['std'].append(std)

    return data


def n_to_mu_meso(n_values):
    """Convert sphere-per-rod counts N into meso roughness µ_meso = 1 - furrow_parameter."""
    mu_meso = []
    for n in n_values:
        if not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 1:
            mu_meso.append(float("nan"))
            continue
        inside = max(0.0, 1.0 - (2.0 / (n - 1.0)) ** 2)
        furrow_parameter = math.sqrt(inside)
        mu_meso.append(1.0 - furrow_parameter)
    return mu_meso


def split_series_name(name):
    """Split a series name like "cross0.4" into ("cross", "0.4")."""
    for family in ("rod", "cross", "star"):
        if name.startswith(family):
            return family, name[len(family):]
    return name, ""


def add_semantic_legends(ax, colors, markers):
    """Add separate legends for particle type and microscopic friction."""
    type_legend = ax.legend(
        handles=[
            Line2D([], [], color=colors["rod"], marker="s", linestyle="none", label="Rod"),
            Line2D([], [], color=colors["cross"], marker="s", linestyle="none", label="Cross"),
            Line2D([], [], color=colors["star"], marker="s", linestyle="none", label="Star"),
        ],
        title="Type", loc="lower left", bbox_to_anchor=(0.00, -0.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )
    ax.add_artist(type_legend)

    ax.legend(
        handles=[
            Line2D([], [], color="black", marker=markers["0.0"], linestyle="none", label="0.0"),
            Line2D([], [], color="black", marker=markers["0.4"], linestyle="none", label="0.4"),
            Line2D([], [], color="black", marker=markers["1.0"], linestyle="none", label="1.0"),
        ],
        title="Friction", loc="lower left", bbox_to_anchor=(0.16, -0.01),
        fontsize=18, title_fontsize=20, frameon=False,
    )


def plot_contact_data(data, colors, markers, series_order):
    """
    Plots the contact data with error bars showing standard deviation
    """
    fig, ax = plt.subplots(figsize=(16, 9))

    for name in series_order:
        if name not in data:
            continue

        family, level = split_series_name(name)
        color = colors[family]
        marker = markers[level]
        series = data[name]

        x = np.asarray(n_to_mu_meso(series['x']))
        y = np.asarray(series['mean'])
        std = np.asarray(series['std'])

        # Plot with error bars
        ax.errorbar(x, y,
                   yerr=std,
                   fmt=marker, linestyle='',
                   label="_nolegend_",
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
    ax.set_ylabel("Neighbouring Particles Count")
    # ax.grid(alpha=0.25, which='both')
    ax.set_xscale('log')
    add_semantic_legends(ax, colors, markers)

    output_path = SAVE_PLOT_FOLDER + "contact_plot.pdf"
    fig.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

def main():
    colors = {
        'cross': 'purple',
        'rod': 'green',
        'star': 'orange'
    }
    markers = {
        '0.0': 'o',
        '0.4': 's',
        '1.0': '^',
    }
    series_order = [
        "rod0.0", "cross0.0", "star0.0",
        "rod0.4", "cross0.4", "star0.4",
        "rod1.0", "cross1.0", "star1.0",
    ]

    data = parse_contact_data(DATA_FILE)
    plot_contact_data(data, colors, markers, series_order)

if __name__ == "__main__":
    main()
